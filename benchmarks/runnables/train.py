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


