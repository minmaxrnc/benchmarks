# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os, sys

# try:
#     PROJECT_ROOT = os.environ['PROJECT_ROOT']
# except KeyError:
#     print("Virtual environment has not been activated")
#     print("Excecute the following command from the project's root")
#     print("source venv_active")

PROJECT_ROOT="."

META_MAIN_FILENAME  = 'meta/meta.yaml'
META_IMPORT_LITERAL = '__import__'
META_ALL_LITERAL    = '__all__'
META_AS_LITERAL     = '__as__'
META_TYPE_SEPARATOR = ':'
META_DATE_FORMAT    = 'dd/mm/yyyy'

OUTPUT_DIR = os.path.join(
        PROJECT_ROOT,
        "outputs"
        )

EXPERIMENTS_OUTPUT_DIR = os.path.join(
        OUTPUT_DIR,
        "experiments"
        )

EVALUATIONS_OUTPUT_DIR = os.path.join(
        OUTPUT_DIR,
        "evaluations"
        )

SAVED_DATASET_OUTPUT_DIR = os.path.join(
        "datasets"
        )


DATA_DIR = os.path.join(
        PROJECT_ROOT,
        "data"
        )

IGNORE_INDEX = -1

SAVED_DATASETS_THRESHOLD = 50 * 1024**3  # 50 GiB (53,687,091,200 bytes)


TQDM_MININTERVAL = 0.5
TQDM_MININTERVAL_LONG = 300


INIT_CONSTANT = 1

