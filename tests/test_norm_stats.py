# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for gradient-/state-norm helpers and per-seed extraction."""
import csv
import math

import pytest
import torch

from benchmarks.utils.norms import flatten_tensors, sum_of_squares
from benchmarks.stats.norms import (
    training_seed_grad_norms,
    training_seed_state_norms,
    evaluation_seed_state_norms,
)
from benchmarks.experiments import utils as experiment_utils
from benchmarks.evaluations import utils as evaluation_utils

pytestmark = pytest.mark.group('stats')

EXPERIMENT = 'latching_(1)'
EVALUATION = 'latching_(1)'
MODEL      = 'minmaxrnc__d_(90)__l_(2)__s_(40)__og_false__cv_basic'
TRAINER    = 'trainer_default'


# ---------------------------------------------------------------------------
# benchmarks.utils.norms (pure tensor math, no I/O)
# ---------------------------------------------------------------------------

class TestFlattenTensors:

    def test_single_tensor(self):
        t = torch.ones(3)
        assert list(flatten_tensors(t)) == [t]

    def test_nested_list_of_dicts(self):
        a, b = torch.ones(2), torch.zeros(3)
        state = [{'neuron': a, 'conv': b}]
        found = list(flatten_tensors(state))
        assert found == [a, b]

    def test_none_yields_nothing(self):
        assert list(flatten_tensors(None)) == []

    def test_non_tensor_scalar_yields_nothing(self):
        assert list(flatten_tensors(42)) == []


class TestSumOfSquares:

    def test_single_tensor(self):
        t = torch.tensor([3.0, 4.0])
        assert sum_of_squares(t).item() == pytest.approx(25.0)

    def test_combines_multiple_tensors_additively(self):
        state = [{'neuron': torch.tensor([3.0, 4.0]), 'conv': torch.tensor([1.0])}]
        assert sum_of_squares(state).item() == pytest.approx(26.0)

    def test_none_returns_none(self):
        assert sum_of_squares(None) is None

    def test_empty_structure_returns_none(self):
        assert sum_of_squares([]) is None

    def test_result_is_detached(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = sum_of_squares(t)
        assert result.requires_grad is False


# ---------------------------------------------------------------------------
# benchmarks.stats.norms -- per-seed extraction from disk
# ---------------------------------------------------------------------------

TRAIN_CSV_HEADER = [
    'epoch', 'lr_0', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
    'runtime', 'timelimit', 'earlystop', 'tokens', 'epoch_time', 'stats_time', 'throughput',
    'grad_norm_mean', 'grad_norm_max', 'state_norm_mean', 'state_norm_max',
]


def _write_train_csv(path, rows, header=TRAIN_CSV_HEADER):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


class TestTrainingSeedGradNorms:

    def test_averages_grad_norm_mean_across_epochs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        path = out_dir + '/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_1.csv'
        _write_train_csv(path, [
            [1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '', 1000, 2.0, 0.01, 500.0, 2.0, 3.0, 5.0, 6.0],
            [2, 0.001, 0.4, 0.95, 0.4, 0.95, 2.0, '', '', 1000, 2.0, 0.01, 500.0, 4.0, 5.0, 7.0, 8.0],
        ])

        values = training_seed_grad_norms(EXPERIMENT, MODEL, TRAINER)
        assert values == pytest.approx([3.0])  # mean(2.0, 4.0)

    def test_no_files_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        assert training_seed_grad_norms(EXPERIMENT, MODEL, TRAINER) == []


class TestTrainingSeedStateNorms:

    def test_averages_state_norm_mean_across_epochs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        path = out_dir + '/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_1.csv'
        _write_train_csv(path, [
            [1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '', 1000, 2.0, 0.01, 500.0, 2.0, 3.0, 5.0, 6.0],
            [2, 0.001, 0.4, 0.95, 0.4, 0.95, 2.0, '', '', 1000, 2.0, 0.01, 500.0, 4.0, 5.0, 9.0, 10.0],
        ])

        values = training_seed_state_norms(EXPERIMENT, MODEL, TRAINER)
        assert values == pytest.approx([7.0])  # mean(5.0, 9.0)

    def test_skips_nan_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        path = out_dir + '/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_1.csv'
        _write_train_csv(path, [
            [1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '', 1000, 2.0, 0.01, 500.0, 2.0, 3.0, float('nan'), float('nan')],
            [2, 0.001, 0.4, 0.95, 0.4, 0.95, 2.0, '', '', 1000, 2.0, 0.01, 500.0, 4.0, 5.0, 9.0, 10.0],
        ])

        values = training_seed_state_norms(EXPERIMENT, MODEL, TRAINER)
        assert values == pytest.approx([9.0])  # only the non-NaN row

    def test_all_nan_gives_no_value_for_that_seed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        path = out_dir + '/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_1.csv'
        _write_train_csv(path, [
            [1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '', 1000, 2.0, 0.01, 500.0, 2.0, 3.0, float('nan'), float('nan')],
        ])

        assert training_seed_state_norms(EXPERIMENT, MODEL, TRAINER) == []


class TestEvaluationSeedStateNorms:

    def _write_eval_csv(self, path, rows):
        header = ['step', 'acc', 'acc_cum', 'runtime', 'timelimit', 'state_norm']
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

    def test_averages_state_norm_across_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluation_utils, 'OUTPUT_DIR', str(tmp_path))
        out_dir = evaluation_utils.get_output_dir(EVALUATION, MODEL, TRAINER)

        path = out_dir + '/eval_log__seed_1.csv'
        self._write_eval_csv(path, [
            [1024, 0.9, 0.9, 1.0, '', 5.0],
            [2048, 0.8, 0.85, 2.0, '', 7.0],
        ])

        values = evaluation_seed_state_norms(EVALUATION, MODEL, TRAINER)
        assert values == pytest.approx([6.0])

    def test_skips_nan_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluation_utils, 'OUTPUT_DIR', str(tmp_path))
        out_dir = evaluation_utils.get_output_dir(EVALUATION, MODEL, TRAINER)

        path = out_dir + '/eval_log__seed_1.csv'
        self._write_eval_csv(path, [
            [1024, 0.9, 0.9, 1.0, '', float('nan')],
            [2048, 0.8, 0.85, 2.0, '', 8.0],
        ])

        values = evaluation_seed_state_norms(EVALUATION, MODEL, TRAINER)
        assert values == pytest.approx([8.0])

    def test_no_files_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluation_utils, 'OUTPUT_DIR', str(tmp_path))
        assert evaluation_seed_state_norms(EVALUATION, MODEL, TRAINER) == []
