from typing import List, Optional, Dict
import time

from ..trainers.trainers import trainers
from ..models.models import models

from .utils import (
        create_output_dir,
        get_output_dir,
        get_best_model_path,
        write_completed,
        get_completed_path,
        has_already_been_completed
        )

from ..utils.seed import seed_everything
from ..utils.timer import Timer


class Experiment:

    def __init__(
            self,
            name: str,
            models_trainers: List[Dict[str, str]],
            metric: str,
            loss: str,
            train_dataset: str,
            val_dataset: str,
            epochs: int
            ):

        self.name            = name
        self.models_trainers = models_trainers
        self.loss            = loss
        self.metric          = metric
        self.train_dataset   = train_dataset
        self.val_dataset     = val_dataset
        self.epochs          = epochs


    def run_model(self, model_name, trainer_name):

        timer = Timer()
        timer.start()

        model_str   = models.str(model_name)
        trainer_str = trainers.str(trainer_name)

        if has_already_been_completed(self.name, model_name, trainer_name):
            print(f"Experiment for model-trainer {model_str}-{trainer_str} has already been completed.")
            print(f"Checkfile: {get_completed_path(self.name, model_name, trainer_name)}")
            return

        create_output_dir(self.name, model_name, trainer_name)
        best_model_path = get_best_model_path(self.name, model_name, trainer_name)
        train_log_path = get_output_dir(self.name, model_name, trainer_name)

        print('')
        print(f"Model: {model_str}")
        print(f"Trainer: {trainer_str}")

        trainer = trainers.instantiate(
            trainer_name,
            model           = model_name,
            loss            = self.loss,
            metric          = self.metric,
            train_dataset   = self.train_dataset,
            val_dataset     = self.val_dataset,
            epochs          = self.epochs,
            csv_log_path    = train_log_path,
            best_model_path = best_model_path
        )

        trainer.fit()

        timer.stop()
        write_completed(self.name, model_name, trainer_name, timer.get_elapsed())


    def run(self):
        seed_everything()

        if len(self.models_trainers) == 0:
            print('No model-trainer pairs to be considered.')
            return

        print("Model-trainer pairs in this experiment:")
        for mt in self.models_trainers:
            model_str   = models.str(mt['model'])
            trainer_str = trainers.str(mt['trainer'])
            print(f"  {model_str}[{trainer_str}]")

        if all([
            has_already_been_completed(self.name, mt['model'], mt['trainer'])
            for mt in self.models_trainers
        ]):
            print(f'Experiment for all models-trainers has already been completed.')
            return

        for mt in self.models_trainers:
            self.run_model(
                mt['model'],
                mt['trainer']
            )


