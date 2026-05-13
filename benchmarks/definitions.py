# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os, sys
from pathlib import Path

META_MAIN_FILENAME  = 'meta/meta.yaml'
META_IMPORT_LITERAL = '__import__'
META_ALL_LITERAL    = '__all__'
META_AS_LITERAL     = '__as__'
META_TYPE_SEPARATOR = ':'
META_DATE_FORMAT    = 'dd/mm/yyyy'


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = os.environ.get("DATA_DIR", ".")

OUTPUT_DIR = os.path.join(
    DATA_DIR,
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
    DATA_DIR,
    "datasets"
)


IGNORE_INDEX = -1

SAVED_DATASETS_THRESHOLD = 50 * 1024**3  # 50 GiB (53,687,091,200 bytes)

TQDM_MININTERVAL = 0.5
TQDM_MININTERVAL_LONG = 300

TQDM_NOTEBOOK = os.environ.get("TQDM_NOTEBOOK", "0") == "1"
if TQDM_NOTEBOOK:
    from tqdm.notebook import tqdm, trange
else:
    from tqdm.auto import tqdm, trange

