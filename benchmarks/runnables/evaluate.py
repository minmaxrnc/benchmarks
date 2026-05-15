# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import sys
import argparse
from pathlib import Path
from datetime import datetime

from ..utils import meta
from ..utils import runtime_config
from ..evaluations.evaluations import evaluations

from ..definitions import EXPERIMENTS_OUTPUT_DIR
from ..definitions import EVALUATIONS_OUTPUT_DIR

from ..utils.device import device


def run(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default=None, metavar='PATH',
                        help='path to runtime config YAML (default: config/run.yaml)')
    parsed = parser.parse_args(args)

    cfg = runtime_config.load(parsed.config)
    if cfg is not None and 'evaluations' in cfg:
        META = meta.load('evaluations')
        names = cfg['evaluations']
    else:
        META = meta.load('evaluations', only_enabled=True)
        names = list(META.keys())

    for evaluation_name in names:
        check_file = Path(
                EVALUATIONS_OUTPUT_DIR,
                evaluation_name,
                'completed.txt'
                )
        print(f"\n# Evaluation: {evaluations.str(evaluation_name)}\n")
        print(f"Device: {device}")
        evaluation = evaluations.instantiate(evaluation_name)
        evaluation.run()

