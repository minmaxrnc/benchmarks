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
from ..utils import config
import zipfile



def run(*argv):
    parser = argparse.ArgumentParser(
        description="Provide exactly one of: --experiment NAME, or --evaluation NAME."
    )

    # Mutually exclusive options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-x", "--experiment",
        metavar="NAME",
        help="Experiment name"
    )
    group.add_argument(
        "-e", "--evaluation",
        metavar="NAME",
        help="Evaluation name"
    )
    group.add_argument(
        '-a', '--all',
        action='store_true',
        help='Generate all datasets'
    )

    args = parser.parse_args(argv)

    # Enforce exactly one of the three
    chosen = sum([
        args.experiment is not None,
        args.evaluation is not None,
        args.all
    ])
    if chosen != 1:
        parser.error("Provide exactly one of: --experiment NAME, --evaluation NAME, or --all")

    def _generate_dataset(dataset_name, long_seqs):
        n_seeds = config.get('ci_max_n')
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
        print(f"\n# Generating *training* datasets for experiment '{experiments.repr(experiment)}'")
        _generate_dataset(experiment_args['train_dataset'], long_seqs=False)
        print(f"\n# Generating *validation* datasets for experiment '{experiments.repr(experiment)}'")
        _generate_dataset(experiment_args['val_dataset'], long_seqs=False)

    def _generate_for_evaluation(evaluation):
        evaluation_args = evaluations.get_property(evaluation, 'args')
        print(f"\n# Generating *evaluation* dataset for evaluation '{experiments.repr(evaluation)}'")
        _generate_dataset(evaluation_args['dataset'], long_seqs=True)

    if args.experiment is not None:
        _generate_for_experiment(args.experiment)
    elif args.evaluation is not None:
        _generate_for_experiment(args.evaluation)
    else:
        experiments_meta = meta.load('experiments', only_enabled=True)
        evaluations_meta = meta.load('evaluations', only_enabled=True)

        print('\nWill generate datasets for the following:')
        print('  Experiments:')
        for experiment in experiments_meta:
            print(f"    - {experiments.str(experiment)}")

        print('  Evaluations:')
        for evaluation in evaluations_meta:
            print(f"    - {evaluations.str(evaluation)}")

        for experiment in experiments_meta:
            _generate_for_experiment(experiment)

        for evaluation in evaluations_meta:
            _generate_for_evaluation(evaluation)


