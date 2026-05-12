# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from ..trainers.trainers import trainers
from ..models.models import models
from ..evaluators.evaluators import evaluators
from ..experiments.experiments import experiments

from .utils import (
        has_already_been_completed,
        write_completed,
        get_completed_path,
        get_output_dir,
        get_trained_model_path,
        trained_model_exists,
        load_model_state_dict
        )

from ..utils.seed import seed_everything
from ..utils.timer import Timer


class Evaluation:

    def __init__(
            self,
            name:           str,
            experiment:     str,
            metric:         str,
            dataset:        str,
            models_trainers = None
            ):

        self.name = name
        self.experiment = experiment
        self.metric = metric
        self.dataset = dataset

        if models_trainers is not None:
            self.models_trainers = models_trainers
        else:
            self.models_trainers = experiments.get_models_trainers(self.experiment)


    def run_model(self, model, trainer):

        timer = Timer()
        timer.start()

        model_str = models.str(model)
        trainer_str = trainers.str(trainer)

        print(f"\n### Model-Trainer: {model_str}[{trainer_str}]\n")

        if has_already_been_completed(self.name, model, trainer):
            print(
                f"Evaluation has already been completed.",
                f"Checkfile: {get_completed_path(self.name, model, trainer)}\n"
            )
            return

        if not trained_model_exists(self.experiment, model, trainer):
            model_path = get_trained_model_path(self.experiment, model, trainer)
            print(
                f"Evaluation not executed.",
                f"The trained model is not present at the following path: {model_path}\n"
            )
            return

        log_path = get_output_dir(self.name, model, trainer)

        model_state_dict = load_model_state_dict(self.experiment, model, trainer)

        evaluator = evaluators.instantiate(
                'Evaluator',
                model            = model,
                model_state_dict = model_state_dict,
                metric           = self.metric,
                dataset          = self.dataset,
                csv_log_path     = log_path
                )
        evaluator.evaluate()

        timer.stop()

        write_completed(self.name, model, trainer, timer.get_elapsed())


    def run(self):
        seed_everything()

        if len(self.models_trainers) == 0:
            print('No models to be considered.')
            return

        print("Model-trainer pairs in this evaluation:")
        for mt in self.models_trainers:
            print(f"  {models.str(mt['model'])}[{trainers.str(mt['trainer'])}]")

        if all([
            has_already_been_completed(self.name, mt['model'], mt['trainer'])
            for mt in self.models_trainers
        ]):
            print('Evaluation for all models has already been completed.')
            return

        for mt in self.models_trainers:
            self.run_model(mt['model'], mt['trainer'])

