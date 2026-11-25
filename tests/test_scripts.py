# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""scripts/ (majority-class baseline, event-gap statistics, gradient probe)
on tiny CPU inputs; the tiny mode of `reproduce`; `generate` with a runtime
config and protocol overrides; the `reproduce` driver. No saved datasets."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.datasets.sequences import Sequences
from benchmarks.runnables import generate, reproduce
from benchmarks.utils import config
from benchmarks.utils import tiny
from benchmarks.utils.meta import Meta

pytestmark = pytest.mark.group('scripts')

SCRIPTS = Path(__file__).resolve().parent.parent / 'scripts'


def _script(name):
    spec = importlib.util.spec_from_file_location(f'_script_{name}', SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sequences(p_set=0.02, length=200, size=20, seed=1):
    return Sequences('seq', size=size, length=[length, length], seed=seed, kind='val',
                     n_ordered_seq=2, n_distinct_seqs=2, multiplicity=2, n_other_tokens=10,
                     p_set=p_set, p_reset=0.0)


# ---------------------------------------------------------------------------
# Majority-class baseline
# ---------------------------------------------------------------------------

class TestMajorityBaseline:

    mb = _script('majority_baseline')

    def test_counts_and_baseline_match_a_direct_count(self):
        ds = _sequences()
        samples = [ds[i] for i in range(len(ds))]
        counts = self.mb.label_counts(samples, ds.vocab_size)
        empty_id = ds.vocab.token_to_id['Completed_[]']
        labels = torch.cat([s['outputs'].long() for s in samples])
        assert sum(counts) == labels.numel()
        assert counts[empty_id] == int((labels == empty_id).sum())
        result = self.mb.baseline_from_counts(counts, ds.vocab.id_to_token, empty_id)
        assert result['majority_label'] == 'Completed_[]'
        assert result['baseline_accuracy'] == pytest.approx(result['empty_set_fraction'])
        assert 0.5 < result['baseline_accuracy'] < 1.0

    def test_masked_steps_are_ignored(self):
        sample = {'outputs': torch.tensor([1, 1, 2, 2, 2]),
                  'mask': torch.tensor([True, True, True, False, False])}
        counts = self.mb.label_counts([sample], 4)
        assert counts == [0, 2, 1, 0]
        assert self.mb.baseline_from_counts(counts)['baseline_accuracy'] == pytest.approx(2 / 3)

    def test_max_samples(self):
        sample = {'outputs': torch.tensor([0]), 'mask': torch.tensor([True])}
        assert self.mb.label_counts([sample] * 5, 2, max_samples=2) == [2, 0]

    def test_gain(self):
        assert self.mb.gain(1.0, 0.94) == pytest.approx(1.0)
        assert self.mb.gain(0.94, 0.94) == pytest.approx(0.0)
        assert self.mb.gain(0.97, 0.94) == pytest.approx(0.5)

    def test_best_val_acc_per_seed(self, tmp_path):
        for seed, accs in ((1, [0.9, 0.99, 0.95]), (2, [0.5, 0.6])):
            with open(tmp_path / f'train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_{seed}.csv', 'w',
                      newline='') as f:
                w = csv.writer(f)
                w.writerow(['epoch', 'val_acc'])
                for e, a in enumerate(accs, 1):
                    w.writerow([e, a])
        assert self.mb.best_val_acc_per_seed(str(tmp_path)) == {1: 0.99, 2: 0.6}


# ---------------------------------------------------------------------------
# Event gap / retention span statistics
# ---------------------------------------------------------------------------

class TestGapStats:

    gs = _script('sequences_gap_stats')

    def test_dataset_params_from_meta(self):
        p = self.gs.dataset_params('sequences_grad_train_(1)')
        assert (p['p_set'], p['T_min'], p['T_max']) == (1.5259e-5, 131072, 131072)
        assert self.gs.dataset_params('sequences_train_(1)')['T_max'] == 512

    def test_rejects_other_datasets(self):
        with pytest.raises(ValueError):
            self.gs.dataset_params('latching_train_(1)')

    def test_sequences_2_training_statistics(self):
        s = self.gs.stats('sequences_train_(1)', 300, np.random.default_rng(0))
        assert s['matched_fraction'] == 1.0
        assert 2.0 < s['event_gap']['mean'] < 8.0          # the paper: 4.0
        assert 250 < s['retention_span']['mean'] < 480      # the paper: 367

    def test_simulation_matches_the_generator_on_dense_data(self):
        # With dense set tokens, the fraction of steps after the first match
        # of either sequence is comparable between the simulation and the
        # generator's labels: check the match rate.
        ds = _sequences(p_set=0.3, length=64, size=200)
        empty_id = ds.vocab.token_to_id['Completed_[]']
        gen_matched = np.mean([bool((ds[i]['outputs'] != empty_id).any()) for i in range(len(ds))])
        _, _, _, sim_matched = self.gs.simulate(0.3, 64, 64, 2, 2, 2, 2000, np.random.default_rng(0))
        assert abs(gen_matched - sim_matched) < 0.08

    def test_cli_writes_json(self, tmp_path):
        out = tmp_path / 'gaps.json'
        self.gs.main(['--datasets', 'sequences_val_(1)', '--n-sequences', '50', '--out', str(out)])
        assert 'sequences_val_(1)' in json.loads(out.read_text())


# ---------------------------------------------------------------------------
# Gradient probe (untrained model, tiny probes)
# ---------------------------------------------------------------------------

class TestGradientProbe:

    gp = _script('gradient_probe')

    def test_probe_tokens(self):
        self.gp.bootstrap()
        vocab = self.gp.build_vocab()
        x, A, B = self.gp.probe_tokens(vocab, 64, 10, filler_seed=0)
        assert x.shape == (1, 64) and (A, B) == (53, 63)
        assert vocab.id_to_token[int(x[0, A])].startswith('Set_0_0_')
        assert vocab.id_to_token[int(x[0, B])].startswith('Set_0_1_')
        others = set(vocab.token_to_id[t] for t in vocab.other_tokens)
        assert all(int(v) in others for i, v in enumerate(x[0]) if i not in (A, B))

    def test_fp32_model_differs_only_in_precision(self):
        self.gp.bootstrap()
        train = self.gp.instantiate(self.gp.MODEL, 'train')
        fp32 = self.gp.instantiate(self.gp.MODEL, 'fp32')
        assert train._backbone_cfg.amp_dtype == 'bf16'
        assert train._backbone_cfg.recurrence_dtype == 'fp16'
        assert fp32._backbone_cfg.amp_dtype is None
        assert fp32._backbone_cfg.recurrence_dtype == 'fp32'
        fp32.load_state_dict(train.state_dict(), strict=True)

    def test_tiny_run_writes_csvs(self, tmp_path):
        out = self.gp.run(checkpoints=('init',), init_seeds=(1,), gaps=[1, 10, 100], length=256,
                          dtypes=('train', 'fp32'), filler_seeds=(0,), out=tmp_path / 'probe',
                          device='cpu')
        with open(out / 'probe_ratio.csv') as f:
            rows = list(csv.DictReader(f))
        # 2 dtypes x (3 gaps + null probe) x 2 objectives
        assert len(rows) == 16
        assert {r['gap'] for r in rows} == {'1', '10', '100', 'none'}
        assert all(float(r['g_B']) > 0 for r in rows)
        with open(out / 'state_jacobian.csv') as f:
            assert len(list(csv.DictReader(f))) > 0
        env = json.loads((out / 'probe_env.json').read_text())
        assert env['args']['model'] == self.gp.MODEL


# ---------------------------------------------------------------------------
# Tiny mode
# ---------------------------------------------------------------------------

class TestTinyMode:

    def test_datasets_are_shrunk(self):
        d = tiny.shrink('datasets', {'class': 'Sequences', 'args': {
            'kind': 'train', 'size': 20000, 'length': [131072, 131072], 'p_set': 0.1}})
        assert d['args'] == {'kind': 'train', 'size': 16, 'length': [32, 32], 'p_set': 0.1}
        d = tiny.shrink('datasets', {'args': {'kind': 'val', 'size': 1000, 'length': [1024, 2048]}})
        assert d['args']['length'] == [40, 48] and d['args']['size'] == 8
        d = tiny.shrink('datasets', {'args': {'kind': 'evaluation', 'size': 1000, 'length': 1024000}})
        assert d['args']['length'] == 256 and d['args']['size'] == 4

    def test_experiments_keep_their_protocol(self):
        e = tiny.shrink('experiments', {'args': {'epochs': 100, 'protocol': {'stopper__enabled': False}}})
        assert e['args']['epochs'] == 2
        assert e['args']['protocol']['stopper__enabled'] is False
        assert e['args']['protocol']['ci_max_n'] == 1
        with config.overrides(e['args']['protocol']):
            pass

    def test_trainers_and_evaluators(self):
        t = tiny.shrink('trainers', {'args': {'train_batch_size': 64, 'max_unroll_steps': 65536,
                                              'grad_accum_steps': 2, 'compile_model': True}})
        assert t['args'] == {'train_batch_size': 4, 'max_unroll_steps': 16,
                             'grad_accum_steps': 2, 'compile_model': True}
        assert tiny.shrink('evaluators', {'args': {'chunk_size': 1024}})['args']['chunk_size'] == 64

    def test_other_scopes_and_the_source_are_unchanged(self):
        entry = {'args': {'d_model': 90}}
        assert tiny.shrink('models', entry) == entry
        meta = Meta({'x': {'args': {'kind': 'train', 'size': 100, 'length': [10, 20]}}})
        view = tiny.TinyMeta(meta, 'datasets')
        assert view['x']['args']['size'] == 16
        assert meta['x']['args']['size'] == 100


# ---------------------------------------------------------------------------
# generate with a runtime config; reproduce driver
# ---------------------------------------------------------------------------

class _FakeDataset:
    def __init__(self, record, name, seed):
        record.append((name, seed, config.get('ci_max_n')))

    def saved_exists(self):
        return True


class TestGenerateWithConfig:

    def test_selected_names_and_protocol_seed_counts(self, monkeypatch):
        record = []
        monkeypatch.setattr(generate.datasets, 'preinstantiate',
                            lambda name, seed: _FakeDataset(record, name, seed))
        monkeypatch.setattr(generate.datasets, 'repr', lambda d, s: 'fake')
        generate.run(config={'experiments': ['longrange_sequences_(1)', 'main_latching_(1)'],
                             'evaluations': ['longrange_sequences_(1)']})
        by_dataset = {}
        for name, seed, n in record:
            by_dataset.setdefault(name, []).append((seed, n))
        # long-range: ci_max_n 5 from the experiment's protocol, also for its evaluation
        for name in ('sequences_grad_train_(1)', 'sequences_grad_val_(1)', 'sequences_grad_evaluation_(1)'):
            assert by_dataset[name] == [(s, 5) for s in range(1, 6)]
        assert by_dataset['latching_train_(1)'] == [(s, 10) for s in range(1, 11)]
        assert 'latching_evaluation_(1)' not in by_dataset


class TestReproduceDriver:

    def test_runs_the_steps_in_order_and_writes_a_record(self, tmp_path, monkeypatch):
        calls = []
        for step in ('generate', 'train', 'evaluate'):
            monkeypatch.setattr(getattr(reproduce, step), 'run',
                                lambda *a, step=step, **kw: calls.append((step, kw['config'])))
        monkeypatch.setattr(reproduce, 'OUTPUT_DIR', str(tmp_path))
        reproduce.run('groups', '--steps', 'evaluate', 'train', '--select', 'groups_s4')
        assert [c[0] for c in calls] == ['train', 'evaluate']
        assert calls[0][1] == {'experiments': ['groups_s4'], 'evaluations': ['groups_s4']}
        (path,) = (tmp_path / 'reproduce').glob('groups__*.json')
        record = json.loads(path.read_text())
        assert record['status'] == 'completed'
        assert record['environment']['torch'] == torch.__version__

    def test_unknown_family(self):
        with pytest.raises(SystemExit):
            reproduce.run('ablation')

    def test_tiny_flag_requires_main(self):
        with pytest.raises(SystemExit):
            reproduce.run('main', '--tiny')
