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
