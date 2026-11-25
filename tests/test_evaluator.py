# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for the Evaluator: loading checkpoints of compiled models, the
per-seed history CSV (with and without state_norm), and an end-to-end CPU
evaluation with an in-memory dataset (model prepared once, throughput and
state-norm bookkeeping, eval_max_batch_size)."""
import csv
import json
import math

import pytest
import torch

from benchmarks.evaluators import evaluator as evaluator_module
from benchmarks.evaluators.evaluator import Evaluator, HistorySeed
from benchmarks.utils import config
from minmax_rnc.model import MinMaxRNC_LM

pytestmark = pytest.mark.group('evaluator')

VOCAB  = 6
LENGTH = 16
CHUNK  = 4


def _model(name='m', iosize=VOCAB):
    return MinMaxRNC_LM(name, iosize=iosize, n_layers=1, d_model=16, d_state=4)


class TestStripCompilePrefix:

    def test_strips_prefix_of_a_compiled_module(self):
        model = _model()
        compiled = torch.compile(model)   # lazy: nothing is compiled until called
        sd = compiled.state_dict()
        assert all(k.startswith('_orig_mod.') for k in sd)
        stripped = Evaluator.strip_compile_prefix(sd)
        assert stripped.keys() == model.state_dict().keys()
        _model().load_state_dict(stripped, strict=True)

    def test_plain_state_dict_is_unchanged(self):
        sd = _model().state_dict()
        assert Evaluator.strip_compile_prefix(sd) is sd


class TestHistorySeed:

    def _history(self):
        return HistorySeed(step=[4, 8], acc=[1.0, 0.5], acc_cum=[1.0, 0.75],
                           runtime=[0.1, 0.2], timelimit=['', ''], state_norm=[2.0, 3.0])

    def test_round_trip(self, tmp_path):
        path = tmp_path / 'h.csv'
        self._history().write_to_csv(path)
        loaded = HistorySeed.load_from_csv(path)
        assert loaded.get('state_norm') == [2.0, 3.0]
        assert loaded.get('acc_cum') == [1.0, 0.75]

    def test_loads_histories_without_state_norm(self, tmp_path):
        path = tmp_path / 'h.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'acc', 'acc_cum', 'runtime', 'timelimit'])
            writer.writerow([1024, 1.0, 1.0, 0.5, ''])
        loaded = HistorySeed.load_from_csv(path)
        assert loaded.get('acc') == [1.0]
        assert math.isnan(loaded.get('state_norm')[0])

    def test_rejects_unknown_header(self, tmp_path):
        path = tmp_path / 'h.csv'
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'accuracy'])
        with pytest.raises(Exception):
            HistorySeed.load_from_csv(path)


# ---------------------------------------------------------------------------
# End-to-end evaluation on CPU
# ---------------------------------------------------------------------------

class _InMemoryEvalDataset:
    def __init__(self, seed, n=3):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(0, VOCAB, (n, LENGTH), generator=g)

    def __len__(self):
        return self.x.size(0)

    def get_loader(self, batch_size):
        return [
            {'inputs': self.x[i:i + batch_size], 'outputs': self.x[i:i + batch_size].clone(),
             'mask': torch.ones_like(self.x[i:i + batch_size], dtype=torch.bool)}
            for i in range(0, len(self), batch_size)
        ]


class _FakeDatasets:
    def get_length(self, name):
        return LENGTH

    def get_iosize(self, name):
        return VOCAB

    def instantiate(self, name, seed):
        return _InMemoryEvalDataset(seed)

    def str(self, name, seed):
        return f'in-memory(seed={seed})'


class _CountingModels:
    def __init__(self):
        self.n_instantiated = 0

    def instantiate(self, name, iosize):
        self.n_instantiated += 1
        torch.manual_seed(0)
        return _model(name, iosize)

    def str(self, name):
        return name


@pytest.fixture
def fake_models(monkeypatch):
    fake = _CountingModels()
    monkeypatch.setattr(evaluator_module, 'datasets', _FakeDatasets())
    monkeypatch.setattr(evaluator_module, 'models', fake)
    return fake


def _evaluator(tmp_path, state_dict, **kwargs):
    return Evaluator('Evaluator', model='m', metric='ChunkedTokenAccuracy', dataset='eval',
                     csv_log_path=str(tmp_path), model_state_dict=state_dict,
                     chunk_size=CHUNK, **kwargs)


PROTOCOL = {'ci_min_n': 2, 'ci_max_n': 2, 'ci_seed': 0}


class TestEvaluateEndToEnd:

    def test_evaluates_and_writes_throughput_and_state_norm(self, tmp_path, fake_models):
        torch.manual_seed(0)
        state_dict = torch.compile(_model()).state_dict()   # '_orig_mod.' keys
        with config.overrides(PROTOCOL | {'eval_max_batch_size': 2}):
            _evaluator(tmp_path, state_dict).evaluate()

        # prepared once for all seeds
        assert fake_models.n_instantiated == 1

        with open(tmp_path / 'eval_summary.json') as f:
            summary = json.load(f)
        assert summary['n_seeds'] == 2

        for seed in (1, 2):
            with open(tmp_path / f'runinfo__seed_{seed}.json') as f:
                runinfo = json.load(f)
            assert runinfo['tokens'] == 3 * LENGTH
            assert runinfo['throughput'] > 0
            assert runinfo['batch_size'] == 2
            history = HistorySeed.load_from_csv(tmp_path / f'eval_log__seed_{seed}.csv')
            assert len(history.get('state_norm')) == LENGTH // CHUNK
            assert all(v > 0 for v in history.get('state_norm'))

    def test_completed_seeds_are_not_reevaluated(self, tmp_path, fake_models):
        state_dict = _model().state_dict()
        with config.overrides(PROTOCOL):
            _evaluator(tmp_path, state_dict).evaluate()
            _evaluator(tmp_path, state_dict).evaluate()
        # the second run loads both seeds from disk and never builds a model
        assert fake_models.n_instantiated == 1

    def test_compile_flag(self, tmp_path, fake_models):
        sd = _model().state_dict()
        assert _evaluator(tmp_path, sd).compile_model is False
        assert _evaluator(tmp_path, sd, compile_model=True).compile_model is True
        with config.overrides({'compile_model': True}):
            assert _evaluator(tmp_path, sd).compile_model is True
            assert _evaluator(tmp_path, sd, compile_model=False).compile_model is False
