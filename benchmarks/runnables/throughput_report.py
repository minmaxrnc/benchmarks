# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""
Aggregates the fine-grained tokens/sec, gradient-norm, and state-norm
statistics recorded by `train` and `evaluate` into mean +/- SE across seeds,
per benchmark family, and globally per model. Reads outputs already on disk
-- run `train`/`evaluate` first.
"""

import os
import json
import argparse

from ..utils import meta
from ..utils import runtime_config
from ..models.models import models
from ..trainers.trainers import trainers
from ..experiments.experiments import experiments
from ..evaluations.evaluations import evaluations

from ..definitions import OUTPUT_DIR

from ..stats.aggregation import build_levels
from ..stats import throughput as throughput_stats
from ..stats import norms as norm_stats


def _evaluation_models_trainers(evaluation_name):
    args = evaluations.get_meta(evaluation_name)['args']
    if 'models_trainers' in args:
        return args['models_trainers']
    return experiments.get_models_trainers(args['experiment'])


def build_report(experiment_names, evaluation_names):
    return {
        'throughput': {
            'training': build_levels(
                experiment_names, experiments.get_models_trainers,
                throughput_stats.training_seed_throughputs
            ),
            'evaluation': build_levels(
                evaluation_names, _evaluation_models_trainers,
                throughput_stats.evaluation_seed_throughputs
            ),
        },
        'gradient_norm': {
            'training': build_levels(
                experiment_names, experiments.get_models_trainers,
                norm_stats.training_seed_grad_norms
            ),
        },
        'state_norm': {
            'training': build_levels(
                experiment_names, experiments.get_models_trainers,
                norm_stats.training_seed_state_norms
            ),
            'evaluation': build_levels(
                evaluation_names, _evaluation_models_trainers,
                norm_stats.evaluation_seed_state_norms
            ),
        },
    }


def run(*args, config=None):
    out_path = os.path.join(OUTPUT_DIR, 'throughput_report.json')

    if config is not None:
        cfg = config
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-c', default=None, metavar='PATH',
                            help='path to runtime config YAML (default: config/run.yaml)')
        parser.add_argument('--out', '-o', default=None, metavar='PATH',
                            help='output JSON path (default: outputs/throughput_report.json)')
        parsed = parser.parse_args(args)
        cfg = runtime_config.load(parsed.config)
        if parsed.out is not None:
            out_path = parsed.out

    if cfg is not None and 'experiments' in cfg:
        experiment_names = cfg['experiments']
    else:
        experiment_names = list(meta.load('experiments', only_enabled=True).keys())

    if cfg is not None and 'evaluations' in cfg:
        evaluation_names = cfg['evaluations']
    else:
        evaluation_names = list(meta.load('evaluations', only_enabled=True).keys())

    report = build_report(experiment_names, evaluation_names)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_summary(report)
    print(f"\nFull report written to {out_path}")

    return report


def _fmt(mean, se, n, n_label, unit):
    if mean is None:
        return 'n/a'
    line = f"{mean:,.3g} {unit}"
    if se is not None:
        line += f" +/- {se:,.3g}"
    return f"{line}  ({n_label}={n})"


def _print_phase(title, phase_levels, unit):
    print(f"\n# {title}")

    family = phase_levels['family']
    glob_  = phase_levels['global']

    if not family:
        print("  (no data on disk yet -- nothing to report)")
        return

    print("\n  By family:")
    for family_name in sorted(family.keys()):
        for model_name, trainers_dict in family[family_name].items():
            for trainer_name, stats in trainers_dict.items():
                model_str   = models.str(model_name)
                trainer_str = trainers.str(trainer_name)
                line = _fmt(stats['mean'], stats['se'], stats['n_instances'], 'instances', unit)
                print(f"    {family_name:<20} {model_str}[{trainer_str}]: {line}")

    print("\n  Global (across families):")
    for model_name, trainers_dict in glob_.items():
        for trainer_name, stats in trainers_dict.items():
            model_str   = models.str(model_name)
            trainer_str = trainers.str(trainer_name)
            line = _fmt(stats['mean'], stats['se'], stats['n_families'], 'families', unit)
            print(f"    {model_str}[{trainer_str}]: {line}")


def _print_summary(report):
    print("\n" + "=" * 60)
    print("THROUGHPUT (tokens/sec)")
    print("=" * 60)
    _print_phase('Training', report['throughput']['training'], 'tok/s')
    _print_phase('Evaluation', report['throughput']['evaluation'], 'tok/s')

    print("\n" + "=" * 60)
    print("GRADIENT NORM (L2, post-backward, pre-clip)")
    print("=" * 60)
    _print_phase('Training', report['gradient_norm']['training'], '')

    print("\n" + "=" * 60)
    print("STATE NORM (L2 over the whole recurrent state)")
    print("=" * 60)
    _print_phase('Training', report['state_norm']['training'], '')
    _print_phase('Evaluation', report['state_norm']['evaluation'], '')
