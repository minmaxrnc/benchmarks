# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass

import numpy as np
import math
import random

from typing import Union, Sequence, List, Tuple, Dict, Optional, Literal

from ..utils.device import pin_memory
from ..utils.device import device
from ..utils.misc import str_class

from .utils.vocabulary import Vocab
from .utils.collate import make_collate_fn

from itertools import chain, combinations

from .dataset import Dataset, DatasetKindType, DatasetLengthType


class Latching(Dataset):

    def __init__(
        self,
        name:       str,
        size:       int,
        length:     DatasetLengthType,
        seed:       int,
        kind:       DatasetKindType,
        # -- Loaded args below
        vocab_size: int,
        n_mem:      int,
        disjoint:   bool,
        all_labels: bool = True,
        p_wrong:    float = 0.0
    ):

        super().__init__(
            name,
            size,
            length,
            seed,
            kind
        )
        self.n_mem      = n_mem
        self.disjoint   = disjoint
        self.all_labels = all_labels
        self.vocab      = Vocab.build(vocab_size)
        self.p_wrong    = p_wrong


    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab.pad_id

    @staticmethod
    def get_iosize(**kwargs):
        return Vocab.get_size(
            kwargs['vocab_size']
        )


    def _makeitem(self, idx, seed) -> dict:
        rng = np.random.default_rng(seed=seed)

        T = self._rand_len(rng)
        initial_token_id = self._rand_initial_token_id(rng)

        inputs  = [initial_token_id] + [self._rand_token_id(rng) for _ in range(T-1)]

        if self._rand_wrong(rng, self.p_wrong):
            wrong_token_id = self._rand_wrong_token_id(rng, initial_token_id)
            outputs = [wrong_token_id] * T
        else:
            outputs = [initial_token_id] * T

        if self.all_labels:
            mask = [True] * T
        else:
            mask = [False] * T
            mask[-1] = True

        stats = {
            'length': T,
            'range_dependency': T-1,
        }

        return {
            'inputs':  inputs,
            'outputs': outputs,
            'mask':    mask,
            'stats':   stats
        }

    def _rand_len(self, rng) -> int:
        """Return a random length in [min_len, max_len]."""
        if self.min_len == self.max_len:
            return self.min_len
        else:
            return int(rng.integers(low=self.min_len, high=self.max_len))

    def _rand_wrong(self, rng, p) -> str:
        return rng.random() < p

    def _rand_initial_token_id(self, rng) -> str:
        """Return a random token among the first n_mem."""
        token_idx = rng.integers(low=0, high=self.n_mem)
        return self.vocab.token_to_id[self.vocab.tokens[token_idx]]

    def _rand_wrong_token_id(self, rng, correct_token_id) -> str:
        token_idx = rng.choice(
            list(range(0, correct_token_id)) + list(range(correct_token_id+1, self.n_mem))
        )
        return self.vocab.token_to_id[self.vocab.tokens[token_idx]]

    def _rand_token_id(self, rng) -> str:
        """Return a random token."""
        if self.disjoint:
            token_idx = rng.integers(
                low=self.n_mem,
                high=len(self.vocab.tokens)
            )
        else:
            token_idx = rng.integers(
                low=0,
                high=len(self.vocab.tokens)
            )
        return self.vocab.token_to_id[self.vocab.tokens[token_idx]]

