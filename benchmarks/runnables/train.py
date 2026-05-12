# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import sys
from datetime import datetime

from ..utils import meta
from ..experiments.experiments import experiments

from ..definitions import EXPERIMENTS_OUTPUT_DIR as OUTPUT_DIR

from ..utils.device import device

META = meta.load('experiments', only_enabled=True)


def run():
    for experiment_name, experiment_entry in META.items():
        print(f"\n# Experiment: {experiments.str(experiment_name)}\n")
        print(f"Device: {device}")
        experiment = experiments.instantiate(experiment_name)
        experiment.run()


