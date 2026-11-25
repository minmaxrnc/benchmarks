# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for permutation-group Cayley table construction.

Pure math checks -- no saved datasets required.
"""
import numpy as np
import pytest

from benchmarks.datasets.utils.permutation_groups import (
    parse_group,
    group_order,
    build_group_table,
    build_group,
)

pytestmark = pytest.mark.group('permutation_groups')


class TestParseGroup:

    def test_parses_symmetric(self):
        assert parse_group('S5') == ('S', 5)

    def test_parses_alternating(self):
        assert parse_group('A5') == ('A', 5)

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValueError):
            parse_group('Z5')

    def test_rejects_invalid_number(self):
        with pytest.raises(ValueError):
            parse_group('Sx')


class TestGroupOrder:

    @pytest.mark.parametrize('name,expected', [
        ('S3', 6), ('S4', 24), ('S5', 120),
        ('A5', 60),
    ])
    def test_order(self, name, expected):
        kind, n = parse_group(name)
        assert group_order(kind, n) == expected


class TestBuildGroupTable:

    GROUPS = ['S3', 'S4', 'A5', 'S5']

    @pytest.mark.parametrize('name', GROUPS)
    def test_table_shape_matches_order(self, name):
        table, order = build_group(name)
        assert table.shape == (order, order)

    @pytest.mark.parametrize('name', GROUPS)
    def test_identity_is_index_zero(self, name):
        table, order = build_group(name)
        # e . j == j  and  i . e == i
        assert np.array_equal(table[0, :], np.arange(order))
        assert np.array_equal(table[:, 0], np.arange(order))

    @pytest.mark.parametrize('name', GROUPS)
    def test_latin_square_property(self, name):
        """Every row and column is a permutation of 0..N-1 (group closure + cancellation)."""
        table, order = build_group(name)
        expected = np.arange(order)
        assert np.array_equal(np.sort(table, axis=1), np.tile(expected, (order, 1)))
        assert np.array_equal(np.sort(table, axis=0), np.tile(expected, (order, 1)).T)

    @pytest.mark.parametrize('name', GROUPS)
    def test_associativity(self, name):
        """(i . j) . k == i . (j . k) for all i, j, k (vectorised over the full table)."""
        table, _ = build_group(name)
        lhs = table[table]        # lhs[i,j,k] = table[table[i,j], k]
        rhs = table[:, table]     # rhs[i,j,k] = table[i, table[j,k]]
        assert np.array_equal(lhs, rhs)

    def test_cached_across_calls(self):
        t1, _ = build_group('S4')
        t2, _ = build_group('S4')
        assert t1 is t2  # lru_cache should return the same array object

    def test_rejects_unsupported_kind(self):
        with pytest.raises(ValueError):
            build_group_table('Z', 4)
