# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""
Per-seed tokens/sec extraction, read back from the fine-grained statistics
recorded by Trainer and Evaluator (see benchmarks/trainers/trainer.py and
benchmarks/evaluators/evaluator.py). Aggregation itself lives in
benchmarks.stats.aggregation -- this module only knows how to turn a
(instance, model, trainer) triple into a list of per-seed throughput values.
"""

import os
import csv
import json
import glob

from ..experiments import utils as experiment_utils
from ..evaluations import utils as evaluation_utils


def training_seed_throughputs(experiment_name, model_name, trainer_name):
    """One throughput value per completed (hyperparameter combo, seed) run:
    total tokens / total wall time across that run's epochs."""
    out_dir = experiment_utils.get_output_dir(experiment_name, model_name, trainer_name)
    values = []
    pattern = os.path.join(out_dir, 'train_log__*__seed_*.csv')
    for path in sorted(glob.glob(pattern)):
        tokens_total, time_total = 0.0, 0.0
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if 'tokens' not in reader.fieldnames or 'epoch_time' not in reader.fieldnames:
                continue
            for row in reader:
                tokens_total += float(row['tokens'])
                time_total   += float(row['epoch_time'])
        if time_total > 0:
            values.append(tokens_total / time_total)
    return values


def evaluation_seed_throughputs(evaluation_name, model_name, trainer_name):
    """One throughput value per completed seed, read from runinfo__seed_*.json."""
    out_dir = evaluation_utils.get_output_dir(evaluation_name, model_name, trainer_name)
    values = []
    pattern = os.path.join(out_dir, 'runinfo__seed_*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        if 'throughput' in data:
            values.append(data['throughput'])
    return values
