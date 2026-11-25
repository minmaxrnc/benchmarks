# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import argparse
from ..utils.tqdm import tqdm

from ..utils import meta
from ..datasets.datasets import datasets
from ..experiments.experiments import experiments
from ..evaluations.evaluations import evaluations

from ..utils.seed import Seeds
from ..utils import config as config_module
from ..utils import runtime_config



def run(*argv, config=None):
    """Generate and save datasets. Without flags, for every enabled experiment
    and evaluation. A runtime config (-c, or the `config` keyword argument as
    for train/evaluate) selects the experiments/evaluations by name instead.
    Each dataset is generated for ci_max_n seeds, as overridden by the
    `protocol:` block of its experiment/evaluation, if any."""
    parser = argparse.ArgumentParser(
        description="Generate datasets. Without flags generates all datasets."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-x", "--experiments",
        action='store_true',
        help="Generate only experiment datasets (train + validation)"
    )
    group.add_argument(
        "-e", "--evaluations",
        action='store_true',
        help="Generate only evaluation datasets"
    )
    parser.add_argument(
        "-s", "--select",
        metavar="PREFIX",
        default=None,
        help="Generate only datasets for experiments/evaluations whose name starts with PREFIX"
    )
    parser.add_argument(
        '--config', '-c', default=None, metavar='PATH',
        help='runtime config YAML listing experiments/evaluations (default: config/run.yaml)'
    )

    args = parser.parse_args(argv)
    cfg = config if config is not None else runtime_config.load(args.config)

    def _names(scope):
        if cfg is not None and scope in cfg:
            names = list(cfg[scope] or [])
        else:
            names = list(meta.load(scope, only_enabled=True).keys())
        if args.select:
            names = [n for n in names if n.startswith(args.select)]
        return names

    def _generate_dataset(dataset_name, long_seqs):
        n_seeds = config_module.get('ci_max_n')
        for seed in Seeds(n_seeds):
            dataset = datasets.preinstantiate(dataset_name, seed=seed)
            header = "{:2d}/{}: ".format(seed, n_seeds)
            if dataset.saved_exists():
                print(f"{header}Dataset '{datasets.repr(dataset, seed)}' -> already saved")
            else:
                dataset.generate_and_save(print_header=header, long_seqs=long_seqs)
                print(f"{header}Dataset '{datasets.repr(dataset, seed)}' -> generated and saved")

    def _generate_for_experiment(experiment):
        experiment_args = experiments.get_property(experiment, 'args')
        with config_module.overrides(experiments.get_protocol(experiment)):
            print(f"\n# Generating *training* datasets for experiment '{experiments.repr(experiment)}'")
            _generate_dataset(experiment_args['train_dataset'], long_seqs=False)
            print(f"\n# Generating *validation* datasets for experiment '{experiments.repr(experiment)}'")
            _generate_dataset(experiment_args['val_dataset'], long_seqs=False)

    def _generate_for_evaluation(evaluation):
        evaluation_args = evaluations.get_property(evaluation, 'args')
        protocol = experiments.get_protocol(evaluation_args['experiment'])
        protocol |= dict(evaluation_args.get('protocol') or {})
        with config_module.overrides(protocol):
            print(f"\n# Generating *evaluation* dataset for evaluation '{evaluations.repr(evaluation)}'")
            _generate_dataset(evaluation_args['dataset'], long_seqs=True)

    experiment_names = [] if args.evaluations else _names('experiments')
    evaluation_names = [] if args.experiments else _names('evaluations')

    print('\nWill generate datasets for the following:')
    if not args.evaluations:
        print('  Experiments:')
        for experiment in experiment_names:
            print(f"    - {experiments.str(experiment)}")
    if not args.experiments:
        print('  Evaluations:')
        for evaluation in evaluation_names:
            print(f"    - {evaluations.str(evaluation)}")

    for experiment in experiment_names:
        _generate_for_experiment(experiment)

    for evaluation in evaluation_names:
        _generate_for_evaluation(evaluation)
