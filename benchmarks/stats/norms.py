# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""
Per-seed gradient-norm / state-norm extraction, read back from the
fine-grained statistics recorded by Trainer and Evaluator (see
benchmarks/trainers/trainer.py and benchmarks/evaluators/evaluator.py).
Aggregation itself lives in benchmarks.stats.aggregation.

There is no gradient norm for evaluation -- it's inference-only, no backward
pass -- so only training_seed_grad_norms exists for that metric.
"""

import os
import csv
import glob
import math

from ..experiments import utils as experiment_utils
from ..evaluations import utils as evaluation_utils


def _mean_finite(values):
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else None


def _training_seed_column_means(experiment_name, model_name, trainer_name, column):
    """One value per completed (hyperparameter combo, seed) run: the mean of
    `column` across that run's epochs, skipping NaN rows (e.g. a model that
    doesn't return a state)."""
    out_dir = experiment_utils.get_output_dir(experiment_name, model_name, trainer_name)
    values = []
    pattern = os.path.join(out_dir, 'train_log__*__seed_*.csv')
    for path in sorted(glob.glob(pattern)):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or column not in reader.fieldnames:
                continue
            epoch_values = [float(row[column]) for row in reader]
        seed_mean = _mean_finite(epoch_values)
        if seed_mean is not None:
            values.append(seed_mean)
    return values


def training_seed_grad_norms(experiment_name, model_name, trainer_name):
    return _training_seed_column_means(experiment_name, model_name, trainer_name, 'grad_norm_mean')


def training_seed_state_norms(experiment_name, model_name, trainer_name):
    return _training_seed_column_means(experiment_name, model_name, trainer_name, 'state_norm_mean')


def evaluation_seed_state_norms(evaluation_name, model_name, trainer_name):
    """One value per completed seed: the mean state_norm across that seed's
    chunk positions, skipping NaN rows."""
    out_dir = evaluation_utils.get_output_dir(evaluation_name, model_name, trainer_name)
    values = []
    pattern = os.path.join(out_dir, 'eval_log__seed_*.csv')
    for path in sorted(glob.glob(pattern)):
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'state_norm' not in reader.fieldnames:
                continue
            chunk_values = [float(row['state_norm']) for row in reader]
        seed_mean = _mean_finite(chunk_values)
        if seed_mean is not None:
            values.append(seed_mean)
    return values
