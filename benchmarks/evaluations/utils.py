# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os
from datetime import datetime
from pathlib import Path
import json

import torch

from ..definitions import EVALUATIONS_OUTPUT_DIR as OUTPUT_DIR

from ..experiments import utils as experiments_utils
from ..utils.device import device

from .evaluations import evaluations
from ..experiments import experiments
from ..models.models import models
from ..trainers.trainers import trainers


def get_output_dir(evaluation, model, trainer):
    evaluation_repr = evaluations.repr(evaluation)
    model_repr = models.repr(model)
    trainer_repr = trainers.repr(trainer)
    output_dir = os.path.join(OUTPUT_DIR, evaluation_repr, model_repr + '__' + trainer_repr)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_completed_path(evaluation, model, trainer):
    return os.path.join(
            get_output_dir(evaluation, model, trainer),
            'completed.txt'
            )

def get_runinfo_path(evaluation, model, trainer):
    return os.path.join(
            get_output_dir(evaluation, model, trainer),
            'run_info.json'
            )

def has_already_been_completed(evaluation, model, trainer):
    return Path(get_completed_path(evaluation, model, trainer)).exists()


def write_completed(evaluation, model, trainer, runtime):
        completion_time = datetime.now().astimezone().isoformat(timespec="seconds")

        runinfo_path = get_runinfo_path(evaluation, model, trainer)
        runinfo_data = {
            'completion_time': completion_time,
            'runtime': runtime
        }
        with open(runinfo_path, "w", encoding="utf-8") as f:
            json.dump(runinfo_data, f, ensure_ascii=False, indent=2)

        completed_path = get_completed_path(evaluation, model, trainer)
        completed_text = f"Evaluation for model '{model}' with trainer '{trainer}' completed at the following time:\n{completion_time}\n"
        Path(completed_path).write_text(completed_text, encoding="utf-8")


def get_trained_model_path(experiment, model, trainer):
    return experiments_utils.get_best_model_path(experiment, model, trainer)

def trained_model_exists(experiment, model, trainer):
    return Path(get_trained_model_path(experiment, model, trainer)).exists()

def load_model_state_dict(experiment, model, trainer):
    model_path = get_trained_model_path(experiment, model, trainer)
    ckpt = torch.load(model_path, map_location=device)
    return ckpt.get("model", ckpt)

