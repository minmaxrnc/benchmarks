# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Software and hardware environment of a run (versions, GPU)."""

import platform
from importlib import metadata

import torch

PACKAGES = ('minmaxrnc', 'numpy', 'triton', 'pyyaml')


def environment():
    info = {
        'python':   platform.python_version(),
        'platform': platform.platform(),
        'torch':    torch.__version__,
        'cuda':     torch.version.cuda,
        'gpu':      torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    for pkg in PACKAGES:
        try:
            info[pkg] = metadata.version(pkg)
        except Exception:
            info[pkg] = None
    return info
