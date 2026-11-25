# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from contextlib import contextmanager
from copy import deepcopy
from . import meta

META = meta.load('config')

# Stack of active overrides (see `overrides`): later entries take precedence.
_OVERRIDES = []


def get(key):
    for layer in reversed(_OVERRIDES):
        if key in layer:
            return deepcopy(layer[key])
    return deepcopy(META[key])


@contextmanager
def overrides(values=None):
    """Temporarily override keys of meta/meta.config.yaml.

    Used for the per-experiment `protocol:` block (see
    benchmarks/experiments/experiment.py). Nested calls stack, the innermost
    taking precedence. Only keys that exist in meta/meta.config.yaml can be
    overridden, so that a typo fails loudly instead of being ignored.
    """
    values = dict(values or {})
    unknown = [k for k in values if k not in META.keys()]
    if unknown:
        raise KeyError(f"Unknown config keys in protocol override: {unknown}")
    _OVERRIDES.append(values)
    try:
        yield
    finally:
        _OVERRIDES.pop()
