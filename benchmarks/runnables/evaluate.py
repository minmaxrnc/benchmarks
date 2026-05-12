# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

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

