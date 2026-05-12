import sys
from pathlib import Path
from datetime import datetime

from ..utils import meta
from ..evaluations.evaluations import evaluations

META = meta.load('evaluations', only_enabled=True)

from ..definitions import EXPERIMENTS_OUTPUT_DIR
from ..definitions import EVALUATIONS_OUTPUT_DIR

from ..utils.device import device


def run():
    for evaluation_name, evaluation_entry in META.items():
        check_file = Path(
                EVALUATIONS_OUTPUT_DIR,
                evaluation_name,
                'completed.txt'
                )
        print(f"\n# Evaluation: {evaluations.str(evaluation_name)}\n")
        print(f"Device: {device}")
        evaluation = evaluations.instantiate(evaluation_name)
        evaluation.run()

