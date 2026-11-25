# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import numpy as np

from .utils.vocabulary import Vocab
from .utils.permutation_groups import build_group

from .dataset import Dataset, DatasetKindType, DatasetLengthType


class GroupStateTracking(Dataset):
    """
    State-tracking task on a finite permutation group (S_n or A_n): the word
    problem Word(G) of the paper (Appendix app:experiments-groups).

    The inputs sigma_1, ..., sigma_T are group elements drawn i.i.d. uniformly
    at random, and the label at every step t is the running product

        gamma_t = gamma_{t-1} * sigma_t = sigma_1 * sigma_2 * ... * sigma_t,

    with gamma_0 the identity, where p * q is the composition p o q of
    permutation_groups.build_group_table, (p o q)[i] = p[q[i]] (apply q, then
    p). Every step is labelled (no masking). Token i is group element i, with
    the identity at index 0; the vocabulary has |G| + 2 tokens (padding and
    an unused special token).

    The task follows the state-tracking benchmark of Siems et al.,
    "DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
    Products", NeurIPS 2025 (Figure 5). Whether its generator multiplies in the
    same order has not been checked: the other order, sigma_t * ... * sigma_1,
    is the inverse of the running product of the inverse elements, so the two
    tasks are equivalent, but their labels differ.
    """

    def __init__(
        self,
        name:       str,
        size:       int,
        length:     DatasetLengthType,
        seed:       int,
        kind:       DatasetKindType,
        # -- Loaded args below
        group:      str,
        all_labels: bool = True,
    ):

        super().__init__(
            name,
            size,
            length,
            seed,
            kind
        )

        self.group       = group
        self.table, self.group_order = build_group(group)
        self.vocab       = Vocab.build(self.group_order)
        self.all_labels  = all_labels


    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab.pad_id

    @staticmethod
    def get_iosize(**kwargs):
        _, order = build_group(kwargs['group'])
        return Vocab.get_size(order)


    def _makeitem(self, idx, seed) -> dict:
        rng = np.random.default_rng(seed=seed)

        T     = self._rand_len(rng)
        table = self.table
        order = self.group_order

        elem_indices = rng.integers(low=0, high=order, size=T)

        acc = 0
        output_indices = []
        for x in elem_indices:
            acc = int(table[acc, x])
            output_indices.append(acc)

        inputs  = [self.vocab.token_to_id[self.vocab.tokens[i]] for i in elem_indices]
        outputs = [self.vocab.token_to_id[self.vocab.tokens[i]] for i in output_indices]

        if self.all_labels:
            mask = [True] * T
        else:
            mask     = [False] * T
            mask[-1] = True

        stats = {
            'length': T,
            'range_dependency': T - 1,
        }

        return {
            'inputs':  inputs,
            'outputs': outputs,
            'mask':    mask,
            'stats':   stats
        }

    def _rand_len(self, rng) -> int:
        """A random length in [min_len, max_len) (the upper bound is
        excluded, as in the other generators); all the configured datasets
        have min_len == max_len."""
        if self.min_len == self.max_len:
            return self.min_len
        else:
            return int(rng.integers(low=self.min_len, high=self.max_len))
