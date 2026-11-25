# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Cayley tables for finite permutation groups (symmetric and alternating).

Used by the GroupStateTracking dataset to generate the state-tracking task of
predicting the running left-to-right product of a random sequence of group
elements (see Siems et al., "DeltaProduct", NeurIPS 2025, Figure 5).
"""

import itertools
import math
from functools import lru_cache
from typing import Tuple

import numpy as np


def parse_group(name: str) -> Tuple[str, int]:
    """Parse a group identifier such as 'S3' or 'A5' into (kind, n)."""
    kind = name[0]
    if kind not in ('S', 'A'):
        raise ValueError(f"Unsupported group kind '{kind}' in '{name}'. Must be 'S' or 'A'.")
    try:
        n = int(name[1:])
    except ValueError:
        raise ValueError(f"Invalid group identifier '{name}'. Expected e.g. 'S3', 'A5'.")
    return kind, n


def group_order(kind: str, n: int) -> int:
    """Return |S_n| = n! or |A_n| = n!/2."""
    order = math.factorial(n)
    if kind == 'A':
        order //= 2
    elif kind != 'S':
        raise ValueError(f"Unsupported group kind '{kind}'. Must be 'S' or 'A'.")
    return order


def _compose(p: tuple, q: tuple) -> tuple:
    """Return p∘q, i.e. (p∘q)[i] = p[q[i]] (apply q then p)."""
    return tuple(p[q[i]] for i in range(len(q)))


def _is_even_permutation(p: tuple) -> bool:
    """Parity of `p` via inversion count (even <=> element of A_n)."""
    n = len(p)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inversions += 1
    return inversions % 2 == 0


@lru_cache(maxsize=None)
def build_group_table(kind: str, n: int) -> np.ndarray:
    """
    Build the Cayley (multiplication) table of S_n or A_n.

    Elements are permutations of range(n), with the identity fixed at index 0
    (for interpretability only -- the choice of indexing is otherwise
    arbitrary and does not affect the group structure).

    Returns
    -------
    table : (N, N) int64 array. table[i, j] = index of elements[i] ∘ elements[j],
        where N = group_order(kind, n).
    """
    if kind not in ('S', 'A'):
        raise ValueError(f"Unsupported group kind '{kind}'. Must be 'S' or 'A'.")

    identity = tuple(range(n))
    all_perms = [p for p in itertools.permutations(range(n)) if p != identity]
    all_perms = [identity] + all_perms

    if kind == 'S':
        elements = all_perms
    else:
        elements = [p for p in all_perms if _is_even_permutation(p)]

    index = {p: i for i, p in enumerate(elements)}
    N = len(elements)
    table = np.empty((N, N), dtype=np.int64)
    for i, p in enumerate(elements):
        for j, q in enumerate(elements):
            table[i, j] = index[_compose(p, q)]
    return table


def build_group(name: str) -> Tuple[np.ndarray, int]:
    """Parse `name` (e.g. 'S3', 'A5') and return (table, group_order)."""
    kind, n = parse_group(name)
    table = build_group_table(kind, n)
    return table, table.shape[0]
