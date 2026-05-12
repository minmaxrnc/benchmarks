# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os
from ..definitions import EXPERIMENTS_OUTPUT_DIR as OUTPUT_DIR
from datetime import datetime
from pathlib import Path
from .experiments import experiments
from ..models.models import models
from ..trainers.trainers import trainers
import json


def get_output_dir(experiment, model, trainer):
    experiment_repr = experiments.repr(experiment)
    model_repr = models.repr(model)
    trainer_repr = trainers.repr(trainer)
    return os.path.join(OUTPUT_DIR, experiment_repr, model_repr + '__' + trainer_repr)

def exists_output_dir(experiment_name, model_name, trainer_name):
    return Path(get_output_dir(experiment_name, model_name, trainer_name)).exists()

def create_output_dir(experiment_name, model_name, trainer_name):
    os.makedirs(
            get_output_dir(experiment_name, model_name, trainer_name),
            exist_ok=True
            )

def get_best_model_path(experiment_name, model_name, trainer_name):
    return os.path.join(
            get_output_dir(experiment_name, model_name, trainer_name),
            'best_model.pt'
            )

def get_completed_path(experiment_name, model_name, trainer_name):
    return os.path.join(
            get_output_dir(experiment_name, model_name, trainer_name),
            'completed.txt'
            )

def get_runinfo_path(experiment_name, model_name, trainer_name):
    return os.path.join(
            get_output_dir(experiment_name, model_name, trainer_name),
            'run_info.json'
            )

def has_already_been_completed(experiment_name, model_name, trainer_name):
    return Path(get_completed_path(experiment_name, model_name, trainer_name)).exists()


def write_completed(experiment_name, model_name, trainer_name, runtime):
        completion_time = datetime.now().astimezone().isoformat(timespec="seconds")

        runinfo_path = get_runinfo_path(experiment_name, model_name, trainer_name)
        runinfo_data = {
            'completion_time': completion_time,
            'runtime': runtime
        }
        with open(runinfo_path, "w", encoding="utf-8") as f:
            json.dump(runinfo_data, f, ensure_ascii=False, indent=2)

        completed_path = get_completed_path(experiment_name, model_name, trainer_name)
        completed_text = f"Experiment for model '{model_name}' with trainer '{trainer_name}' completed at the following time:\n{completion_time}\n"
        Path(completed_path).write_text(completed_text, encoding="utf-8")


