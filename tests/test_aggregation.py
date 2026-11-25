# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for the generic aggregation core (benchmarks.stats.aggregation),
shared by every statistic reported by throughput_report."""
import math

import pytest

from benchmarks.stats.aggregation import (
    family_of,
    mean_se,
    collect_per_run,
    aggregate_by_family,
    aggregate_global,
    levels,
    build_levels,
)

pytestmark = pytest.mark.group('stats')


# ---------------------------------------------------------------------------
# family_of
# ---------------------------------------------------------------------------

class TestFamilyOf:

    def test_strips_instance_suffix(self):
        assert family_of('latching_(3)') == 'latching'

    def test_strips_negative_instance_suffix(self):
        assert family_of('foo_(-1)') == 'foo'

    def test_name_without_suffix_is_its_own_family(self):
        assert family_of('bare_name') == 'bare_name'


# ---------------------------------------------------------------------------
# mean_se
# ---------------------------------------------------------------------------

class TestMeanSE:

    def test_basic_mean_and_se(self):
        mean, se = mean_se([1.0, 2.0, 3.0])
        assert mean == pytest.approx(2.0)
        assert se == pytest.approx(1.0 / math.sqrt(3))

    def test_single_value_has_no_se(self):
        mean, se = mean_se([5.0])
        assert mean == pytest.approx(5.0)
        assert se is None

    def test_empty_returns_none(self):
        assert mean_se([]) == (None, None)


# ---------------------------------------------------------------------------
# collect_per_run
# ---------------------------------------------------------------------------

class TestCollectPerRun:

    def test_builds_nested_dict_from_seed_values(self):
        def get_models_trainers(instance_name):
            return [{'model': 'm', 'trainer': 't'}]

        def seed_values_fn(instance_name, model_name, trainer_name):
            return {'x_(1)': [10.0, 20.0], 'x_(2)': [30.0]}[instance_name]

        per_run = collect_per_run(['x_(1)', 'x_(2)'], get_models_trainers, seed_values_fn)

        assert per_run['x_(1)']['m']['t']['n_seeds'] == 2
        assert per_run['x_(1)']['m']['t']['mean'] == pytest.approx(15.0)
        assert per_run['x_(2)']['m']['t']['n_seeds'] == 1
        assert per_run['x_(2)']['m']['t']['mean'] == pytest.approx(30.0)

    def test_skips_instances_with_no_values(self):
        def get_models_trainers(instance_name):
            return [{'model': 'm', 'trainer': 't'}]

        def seed_values_fn(instance_name, model_name, trainer_name):
            return []

        per_run = collect_per_run(['x_(1)'], get_models_trainers, seed_values_fn)
        assert per_run == {}

    def test_multiple_models_trainers_per_instance(self):
        def get_models_trainers(instance_name):
            return [
                {'model': 'm1', 'trainer': 't'},
                {'model': 'm2', 'trainer': 't'},
            ]

        def seed_values_fn(instance_name, model_name, trainer_name):
            return [1.0] if model_name == 'm1' else [2.0]

        per_run = collect_per_run(['x_(1)'], get_models_trainers, seed_values_fn)
        assert per_run['x_(1)']['m1']['t']['mean'] == pytest.approx(1.0)
        assert per_run['x_(1)']['m2']['t']['mean'] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# aggregate_by_family / aggregate_global
# ---------------------------------------------------------------------------

class TestAggregateByFamily:

    def test_averages_instance_means_equally_regardless_of_seed_count(self):
        per_run = {
            'latching_(1)': {'m': {'t': {'mean': 100.0, 'se': None, 'n_seeds': 1, 'values': [100.0]}}},
            'latching_(2)': {'m': {'t': {'mean': 200.0, 'se': None, 'n_seeds': 10, 'values': [200.0] * 10}}},
        }
        family = aggregate_by_family(per_run)
        assert family['latching']['m']['t']['mean'] == pytest.approx(150.0)
        assert family['latching']['m']['t']['n_instances'] == 2

    def test_separates_distinct_families(self):
        per_run = {
            'latching_(1)':  {'m': {'t': {'mean': 100.0, 'se': None, 'n_seeds': 1, 'values': [100.0]}}},
            'sequences_(1)': {'m': {'t': {'mean': 300.0, 'se': None, 'n_seeds': 1, 'values': [300.0]}}},
        }
        family = aggregate_by_family(per_run)
        assert set(family.keys()) == {'latching', 'sequences'}
        assert family['latching']['m']['t']['mean'] == pytest.approx(100.0)
        assert family['sequences']['m']['t']['mean'] == pytest.approx(300.0)


class TestAggregateGlobal:

    def test_averages_family_means_equally(self):
        family_stats = {
            'latching':  {'m': {'t': {'mean': 100.0, 'se': None, 'n_instances': 8}}},
            'sequences': {'m': {'t': {'mean': 300.0, 'se': None, 'n_instances': 5}}},
        }
        glob_ = aggregate_global(family_stats)
        assert glob_['m']['t']['mean'] == pytest.approx(200.0)
        assert glob_['m']['t']['n_families'] == 2


class TestLevelsAndBuildLevels:

    def test_levels_wraps_per_run_family_global(self):
        per_run = {
            'latching_(1)': {'m': {'t': {'mean': 100.0, 'se': None, 'n_seeds': 1, 'values': [100.0]}}},
        }
        result = levels(per_run)
        assert set(result.keys()) == {'per_run', 'family', 'global'}
        assert result['per_run'] is per_run
        assert result['family']['latching']['m']['t']['mean'] == pytest.approx(100.0)
        assert result['global']['m']['t']['mean'] == pytest.approx(100.0)

    def test_build_levels_end_to_end(self):
        def get_models_trainers(instance_name):
            return [{'model': 'm', 'trainer': 't'}]

        def seed_values_fn(instance_name, model_name, trainer_name):
            return {'latching_(1)': [10.0], 'sequences_(1)': [30.0]}[instance_name]

        result = build_levels(['latching_(1)', 'sequences_(1)'], get_models_trainers, seed_values_fn)
        assert result['global']['m']['t']['mean'] == pytest.approx(20.0)
        assert result['global']['m']['t']['n_families'] == 2

    def test_empty_instance_list_gives_empty_report(self):
        result = build_levels([], lambda n: [], lambda i, m, t: [])
        assert result == {'per_run': {}, 'family': {}, 'global': {}}
