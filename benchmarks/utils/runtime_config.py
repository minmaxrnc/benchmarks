# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import yaml
from pathlib import Path

from ..definitions import RUNTIME_CONFIG_PATH


def load(path=None):
    """Load the runtime config YAML, returning None if the file does not exist.

    path: override the default config/run.yaml location.
    """
    resolved = Path(path) if path is not None else RUNTIME_CONFIG_PATH
    if not resolved.exists():
        return None
    with open(resolved) as f:
        return yaml.safe_load(f) or {}
