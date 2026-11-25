# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for per-seed throughput extraction (benchmarks.stats.throughput).
Aggregation itself is covered by tests/test_aggregation.py."""
import csv
import json

import pytest

from benchmarks.stats.throughput import (
    training_seed_throughputs,
    evaluation_seed_throughputs,
)
from benchmarks.experiments import utils as experiment_utils
from benchmarks.evaluations import utils as evaluation_utils

pytestmark = pytest.mark.group('stats')

# Real names from meta/ -- repr()/get_output_dir() only need the meta entry
# to exist, not the model class to be importable.
EXPERIMENT = 'latching_(1)'
EVALUATION = 'latching_(1)'
MODEL      = 'minmaxrnc__d_(90)__l_(2)__s_(40)__og_false__cv_basic'
TRAINER    = 'trainer_default'

TRAIN_CSV_HEADER = [
    'epoch', 'lr_0', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
    'runtime', 'timelimit', 'earlystop', 'tokens', 'epoch_time', 'throughput',
    'grad_norm_mean', 'grad_norm_max', 'state_norm_mean', 'state_norm_max',
]


def _write_train_csv(path, rows, header=TRAIN_CSV_HEADER):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


class TestTrainingSeedThroughputs:

    def test_sums_tokens_and_time_across_epochs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        path = out_dir + '/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_1.csv'
        _write_train_csv(path, [
            [1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '', 1000, 2.0, 500.0, 1.0, 1.0, 2.0, 2.0],
            [2, 0.001, 0.4, 0.95, 0.4, 0.95, 2.0, '', '', 1000, 2.0, 500.0, 1.0, 1.0, 2.0, 2.0],
        ])

        values = training_seed_throughputs(EXPERIMENT, MODEL, TRAINER)
        assert values == pytest.approx([2000 / 4.0])

    def test_one_value_per_seed_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        for seed in (1, 2, 3):
            path = out_dir + f'/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_{seed}.csv'
            _write_train_csv(path, [[1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '', 1000, 1.0, 1000.0, 1.0, 1.0, 2.0, 2.0]])

        assert len(training_seed_throughputs(EXPERIMENT, MODEL, TRAINER)) == 3

    def test_skips_csv_without_throughput_columns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        experiment_utils.create_output_dir(EXPERIMENT, MODEL, TRAINER)
        out_dir = experiment_utils.get_output_dir(EXPERIMENT, MODEL, TRAINER)

        path = out_dir + '/train_log__lr_0.001__bs_64__opt_adam__sc_step__seed_1.csv'
        old_header = [
            'epoch', 'lr_0', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'runtime', 'timelimit', 'earlystop',
        ]
        _write_train_csv(path, [[1, 0.001, 0.5, 0.9, 0.5, 0.9, 1.0, '', '']], header=old_header)

        assert training_seed_throughputs(EXPERIMENT, MODEL, TRAINER) == []

    def test_no_files_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        assert training_seed_throughputs(EXPERIMENT, MODEL, TRAINER) == []


class TestEvaluationSeedThroughputs:

    def test_reads_throughput_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluation_utils, 'OUTPUT_DIR', str(tmp_path))
        out_dir = evaluation_utils.get_output_dir(EVALUATION, MODEL, TRAINER)

        path = out_dir + '/runinfo__seed_1.json'
        with open(path, 'w') as f:
            json.dump({'runtime': 1.0, 'runtime_pretty': '1.00s',
                       'tokens': 1000, 'throughput': 1000.0}, f)

        assert evaluation_seed_throughputs(EVALUATION, MODEL, TRAINER) == [1000.0]

    def test_skips_json_without_throughput_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluation_utils, 'OUTPUT_DIR', str(tmp_path))
        out_dir = evaluation_utils.get_output_dir(EVALUATION, MODEL, TRAINER)

        path = out_dir + '/runinfo__seed_1.json'
        with open(path, 'w') as f:
            json.dump({'runtime': 1.0, 'runtime_pretty': '1.00s'}, f)

        assert evaluation_seed_throughputs(EVALUATION, MODEL, TRAINER) == []

    def test_no_files_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(evaluation_utils, 'OUTPUT_DIR', str(tmp_path))
        assert evaluation_seed_throughputs(EVALUATION, MODEL, TRAINER) == []
