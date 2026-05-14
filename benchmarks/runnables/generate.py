# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import argparse
from tqdm.auto import tqdm

from ..utils import meta
from ..datasets.datasets import datasets
from ..experiments.experiments import experiments
from ..evaluations.evaluations import evaluations

from ..utils.seed import Seeds
from ..utils import config
import zipfile



def run(*argv):
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

    args = parser.parse_args(argv)

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

    if args.experiments:
        experiments_meta = meta.load('experiments', only_enabled=True)
        if args.select:
            experiments_meta = [e for e in experiments_meta if e.startswith(args.select)]

        print('\nWill generate datasets for the following experiments:')
        for experiment in experiments_meta:
            print(f"    - {experiments.str(experiment)}")

        for experiment in experiments_meta:
            _generate_for_experiment(experiment)

    elif args.evaluations:
        evaluations_meta = meta.load('evaluations', only_enabled=True)
        if args.select:
            evaluations_meta = [e for e in evaluations_meta if e.startswith(args.select)]

        print('\nWill generate datasets for the following evaluations:')
        for evaluation in evaluations_meta:
            print(f"    - {evaluations.str(evaluation)}")

        for evaluation in evaluations_meta:
            _generate_for_evaluation(evaluation)

    else:
        experiments_meta = meta.load('experiments', only_enabled=True)
        evaluations_meta = meta.load('evaluations', only_enabled=True)
        if args.select:
            experiments_meta = [e for e in experiments_meta if e.startswith(args.select)]
            evaluations_meta = [e for e in evaluations_meta if e.startswith(args.select)]

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


