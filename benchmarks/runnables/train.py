# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import sys
import argparse
from datetime import datetime

from ..utils import meta
from ..utils import runtime_config
from ..experiments.experiments import experiments

from ..definitions import EXPERIMENTS_OUTPUT_DIR as OUTPUT_DIR

from ..utils.device import device


def run(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=None, metavar='PATH',
                        help='path to runtime config YAML (default: config/run.yaml)')
    parsed = parser.parse_args(args)

    cfg = runtime_config.load(parsed.config)
    if cfg is not None and 'experiments' in cfg:
        META = meta.load('experiments')
        names = cfg['experiments']
    else:
        META = meta.load('experiments', only_enabled=True)
        names = list(META.keys())

    for experiment_name in names:
        print(f"\n# Experiment: {experiments.str(experiment_name)}\n")
        print(f"Device: {device}")
        experiment = experiments.instantiate(experiment_name)
        experiment.run()


