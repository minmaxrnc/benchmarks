# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""
Metric-agnostic aggregation shared by every statistic reported by
`throughput_report` (tokens/sec, gradient norm, state norm, ...).

Three levels, computed by feeding a per-seed value extractor in:

  per_run  -- for one (experiment/evaluation instance, model, trainer): mean
              and standard error of the per-seed values.
  family   -- for one benchmark family (e.g. 'latching', spanning
              latching_(1)..latching_(8)): mean and SE across that family's
              per-run means.
  global   -- for one model/trainer: mean and SE across its family means.

Family membership is derived from the instance name itself: any name of the
form '<family>_(<N>)' belongs to family '<family>'; anything else is its own
singleton family.
"""

import re

import numpy as np


_FAMILY_RE = re.compile(r'^(.*)_\(-?\d+\)$')


def family_of(instance_name: str) -> str:
    m = _FAMILY_RE.match(instance_name)
    return m.group(1) if m else instance_name


def mean_se(values):
    """Sample mean and standard error of the mean. SE is None for n < 2."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return None, None
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return mean, se


def collect_per_run(instance_names, get_models_trainers, seed_values_fn):
    """{instance_name: {model_name: {trainer_name: {...}}}}, skipping any
    (instance, model, trainer) with no recorded seeds yet.

    seed_values_fn(instance_name, model_name, trainer_name) -> list[float]
    """
    per_run = {}
    for instance_name in instance_names:
        for mt in get_models_trainers(instance_name):
            model_name, trainer_name = mt['model'], mt['trainer']
            values = seed_values_fn(instance_name, model_name, trainer_name)
            if not values:
                continue
            mean, se = mean_se(values)
            per_run.setdefault(instance_name, {}) \
                    .setdefault(model_name, {})[trainer_name] = {
                'n_seeds': len(values),
                'mean': mean,
                'se': se,
                'values': values,
            }
    return per_run


def aggregate_by_family(per_run):
    """{instance: {model: {trainer: stats}}} -> {family: {model: {trainer: stats}}},
    averaging each family's per-instance means (each instance weighted equally,
    regardless of how many seeds it happened to use)."""
    groups = {}
    for instance_name, models_dict in per_run.items():
        family = family_of(instance_name)
        for model_name, trainers_dict in models_dict.items():
            for trainer_name, stats in trainers_dict.items():
                groups.setdefault(family, {}) \
                      .setdefault(model_name, {}) \
                      .setdefault(trainer_name, []) \
                      .append(stats['mean'])

    family_stats = {}
    for family, models_dict in groups.items():
        family_stats[family] = {}
        for model_name, trainers_dict in models_dict.items():
            family_stats[family][model_name] = {}
            for trainer_name, means in trainers_dict.items():
                mean, se = mean_se(means)
                family_stats[family][model_name][trainer_name] = {
                    'n_instances': len(means),
                    'mean': mean,
                    'se': se,
                }
    return family_stats


def aggregate_global(family_stats):
    """{family: {model: {trainer: stats}}} -> {model: {trainer: stats}},
    averaging each model/trainer's family means (each family weighted equally)."""
    groups = {}
    for family, models_dict in family_stats.items():
        for model_name, trainers_dict in models_dict.items():
            for trainer_name, stats in trainers_dict.items():
                groups.setdefault(model_name, {}) \
                      .setdefault(trainer_name, []) \
                      .append(stats['mean'])

    global_stats = {}
    for model_name, trainers_dict in groups.items():
        global_stats[model_name] = {}
        for trainer_name, means in trainers_dict.items():
            mean, se = mean_se(means)
            global_stats[model_name][trainer_name] = {
                'n_families': len(means),
                'mean': mean,
                'se': se,
            }
    return global_stats


def levels(per_run):
    family = aggregate_by_family(per_run)
    return {'per_run': per_run, 'family': family, 'global': aggregate_global(family)}


def build_levels(instance_names, get_models_trainers, seed_values_fn):
    """collect_per_run + levels in one call -- the common case."""
    per_run = collect_per_run(instance_names, get_models_trainers, seed_values_fn)
    return levels(per_run)
