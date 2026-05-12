# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for dataset item generation and DataLoader collation.

This group is DISABLED by default in meta/meta.tests.yaml because it requires
pre-generated datasets saved to disk (run the generation runnables first).
Enable it by setting  datasets.enabled: true  in meta/meta.tests.yaml.
"""
import pytest

pytestmark = pytest.mark.group('datasets')


class TestDatasetPlaceholder:
    """Placeholder — replace with real tests once datasets are generated."""

    def test_placeholder(self):
        pytest.skip(
            "Dataset tests require saved datasets on disk. "
            "Generate them first, then set datasets.enabled: true "
            "in meta/meta.tests.yaml."
        )
