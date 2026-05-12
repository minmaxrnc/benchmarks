# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import sys
import os
import importlib
import torch
import signal

import benchmarks.runnables
from benchmarks.runnables import *


def handle_sigint(signum, frame):
    sys.exit(0)


def load_models():
    """Import all packages found in the models/ directory.

    Each subdirectory of models/ that contains an __init__.py is imported.
    Packages are responsible for calling benchmarks.register_model(cls)
    for every model class they provide.
    """
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.isdir(models_dir):
        return

    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    for name in sorted(os.listdir(models_dir)):
        package_dir = os.path.join(models_dir, name)
        if os.path.isdir(package_dir) and os.path.isfile(os.path.join(package_dir, '__init__.py')):
            try:
                importlib.import_module(name)
            except ImportError as e:
                print(f"Warning: could not import model package '{name}': {e}")


def startup():
    signal.signal(signal.SIGINT, handle_sigint)
    load_models()


def print_available_runnables():
    print('Available runnables')
    for x in sorted(benchmarks.runnables.__all__):
        print(' ', x)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Expecting runnable module to execute as argument, or 'ls' to list them")
    elif sys.argv[1] == 'ls':
        print_available_runnables()
    else:
        startup()
        runnable_name = sys.argv[1]
        if hasattr(sys.modules[__name__], runnable_name):
            runnable = getattr(sys.modules[__name__], runnable_name)
            runnable.run(*sys.argv[2:])
        else:
            print(f"The runnable '{runnable_name}' does not exist")
            print_available_runnables()
