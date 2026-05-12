# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os
import sys
import importlib
import yaml
import pytest

# All meta paths are relative to the project root (".")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load_model_plugins():
    """Mirror main.py load_models(): import every package under models/."""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    if not os.path.isdir(models_dir):
        return
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    for name in sorted(os.listdir(models_dir)):
        pkg_dir = os.path.join(models_dir, name)
        if os.path.isdir(pkg_dir) and os.path.isfile(os.path.join(pkg_dir, '__init__.py')):
            try:
                importlib.import_module(name)
            except ImportError as e:
                print(f"Warning: could not import model plugin '{name}': {e}", file=sys.stderr)


_load_model_plugins()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "group(name): test group as defined in meta/meta.tests.yaml",
    )


def pytest_collection_modifyitems(config, items):
    meta_path = os.path.join(PROJECT_ROOT, 'meta', 'meta.tests.yaml')
    with open(meta_path) as f:
        test_meta = yaml.safe_load(f)

    groups = test_meta.get('groups', {})

    for item in items:
        for marker in item.iter_markers('group'):
            group_name = marker.args[0]
            group_cfg = groups.get(group_name, {})
            if not group_cfg.get('enabled', True):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"group '{group_name}' is disabled in meta/meta.tests.yaml"
                    )
                )
                break
