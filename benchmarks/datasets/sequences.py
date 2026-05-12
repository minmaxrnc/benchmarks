
# TODO add tokens to retrieve result for specific sequence

import torch
from dataclasses import dataclass
import numpy as np

from collections import deque

from typing import List, Dict, Union, Tuple, Optional, Literal

from ..utils.misc import powerset

from .utils.vocabulary import generate_letter_ids
from .utils.collate import make_collate_fn

from .dataset import Dataset, DatasetKindType, DatasetLengthType


class Sequences(Dataset):

    def __init__(
        self,
        name:             str,
        size:             int,
        length:           DatasetLengthType,
        seed:             int,
        kind:             DatasetKindType,
        # -- Loaded args below
        n_ordered_seq:    int,
        n_distinct_seqs:  int,
        multiplicity:     int,
        n_other_tokens:   int,
        p_set:            float,
        p_reset:          float,
        all_labels:       bool = True
    ):

        if p_set + p_reset > 1.0:
            raise ValueError('Required: p_set + p_reset <= 1.0')


        super().__init__(
            name,
            size,
            length,
            seed,
            kind
        )

        self.all_labels = all_labels

        self.vocab = Vocab.build(
            n_ordered_seq,
            n_distinct_seqs,
            multiplicity,
            n_other_tokens
        )

        self.n_ordered_seq    = n_ordered_seq
        self.n_distinct_seqs  = n_distinct_seqs
        self.multiplicity     = multiplicity
        self.p_set            = p_set
        self.p_reset          = p_reset
        self.n_other_tokens   = n_other_tokens

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab.pad_id

    @staticmethod
    def get_iosize(**kwargs):
        return Vocab.get_size(
            n_ordered_seq   = kwargs['n_ordered_seq'],
            n_distinct_seqs = kwargs['n_distinct_seqs'],
            multiplicity    = kwargs['multiplicity'],
            n_other_tokens  = kwargs['n_other_tokens']
        )


    def _makeitem(self, idx, seed) -> dict:
        rand = _Random(
            seed,
            set_tokens   = self.vocab.set_tokens,
            reset_tokens = self.vocab.reset_tokens,
            other_tokens = self.vocab.other_tokens,
            p_set        = self.p_set,
            p_reset      = self.p_reset
        )

        completion_status       = [0     for _ in range(self.n_distinct_seqs)]
        completion_status_first = [0     for _ in range(self.n_distinct_seqs)]
        completed               = [False for _ in range(self.n_distinct_seqs)]
        range_dependency        = 0

        inputs  = []
        outputs = []

        T = rand.len(self.min_len, self.max_len)

        for t in range(T):

            token_kind  = rand.token_kind()
            input_token = rand.token(token_kind)

            if token_kind == 'set':
                seq_id = self.vocab.get_seq_id(input_token)
                pos_id = self.vocab.get_pos_id(input_token)

                if completion_status[seq_id] == pos_id:
                    completion_status[seq_id] += 1
                if completion_status[seq_id] == self.n_distinct_seqs:
                    completed[seq_id] = True

            elif token_kind == 'reset':
                seq_id = self.vocab.get_seq_id(input_token)
                completion_status[seq_id] = 0
                completed[seq_id] = False

            output_token = self.vocab.get_output_token(completed)

            inputs.append(self.vocab.token_to_id[input_token])
            outputs.append(self.vocab.token_to_id[output_token])

            # Compute stats
            if token_kind != 'other':
                # Update first match
                if completion_status[seq_id] == 1 and pos_id == 0:
                    # First match or re-match
                    completion_status_first[seq_id] = t
                elif completion_status[seq_id] == 0:
                    # No match yet
                    completion_status_first[seq_id] = t
            # Update ranges
            for seq_id in range(self.n_distinct_seqs):
                cur_range = t - completion_status_first[seq_id]
                range_dependency = max(range_dependency, cur_range)

        stats = {
            'length': T,
            'range_dependency': range_dependency
        }

        if self.all_labels:
            mask = [True] * T
        else:
            mask     = [False] * T
            mask[-1] = True

        return {
            'inputs':  inputs,
            'outputs': outputs,
            'mask':    mask,
            'stats':   stats
        }



_VOCAB_SEED = 42

PAD_TOKEN        = "_"
RESET_PREFIX     = "Reset"
SET_PREFIX       = "Set"
COMPLETED_PREFIX = "Completed"
OTHER_PREFIX     = "Other"

@dataclass(frozen=True)
class Vocab:
    pad_token:                str             = PAD_TOKEN
    reset_prefix:             str             = RESET_PREFIX
    set_prefix:               str             = SET_PREFIX
    other_prefix:             str             = OTHER_PREFIX
    completed_prefix:         str             = COMPLETED_PREFIX
    set_tokens:               List[str]       = None
    reset_tokens:             List[str]       = None
    set_token_ids:            List[int]       = None
    reset_token_ids:          List[int]       = None
    other_tokens:             List[str]       = None
    output_tokens:            List[str]       = None
    token_to_id:              Dict[str, int]  = None
    id_to_token:              List[str]       = None
    set_token_to_seq_id:      Dict[str,int]   = None
    set_token_to_pos_id:      Dict[str,int]   = None
    reset_token_to_seq_id:    Dict[str,int]   = None
    output_token_to_seq_idxs: Dict[str,str]   = None
    seq_idxs_to_output_token: Dict[str,str]   = None


    def __len__(self):
        return len(self.id_to_token)

    @staticmethod
    def get_size(**kwargs):
        """Same args as build."""
        return len(Vocab.build(**kwargs))

    @staticmethod
    def _get_special_tokens():
        return [PAD_TOKEN]

    @staticmethod
    def build(n_ordered_seq: int,
              n_distinct_seqs: int,
              multiplicity: int,
              n_other_tokens
              ) -> "Vocab":

        specials = Vocab._get_special_tokens()

        # Generate tokens for sets and resets

        sr_token_ids = generate_letter_ids(multiplicity)

        reset_token_to_seq_idx = {}
        reset_tokens = []
        for seq_idx in range(n_distinct_seqs):
            for mult_idx, letter_id in enumerate(sr_token_ids):
                reset_token = '_'.join([
                    RESET_PREFIX,
                    str(seq_idx),
                    letter_id
                    ])
                reset_token_to_seq_idx[reset_token] = seq_idx
                reset_tokens.append(reset_token)

        set_token_to_seq_idx = {}
        set_token_to_pos_idx = {}
        set_tokens = []
        for seq_idx in range(n_distinct_seqs):
            for pos_idx in range(n_ordered_seq):
                for mult_idx, letter_id in enumerate(sr_token_ids):
                    set_token = '_'.join([
                        SET_PREFIX,
                        str(seq_idx),
                        str(pos_idx),
                        letter_id
                        ])
                    set_token_to_seq_idx[set_token] = seq_idx
                    set_token_to_pos_idx[set_token] = pos_idx
                    set_tokens.append(set_token)

        # Generate 'completed' tokens
        completed_token_to_seq_idxs = {}
        completed_tokens = []
        seq_idxs_to_token = {}
        for seq_idxs in powerset(range(n_distinct_seqs)):
            completed_token = '_'.join([
                COMPLETED_PREFIX,
                str(seq_idxs)
                ])
            completed_token_to_seq_idxs[completed_token] = seq_idxs
            seq_idxs_to_token[str(seq_idxs)] = completed_token
            completed_tokens.append(completed_token)

        # Generate 'other' tokens
        other_token_ids = generate_letter_ids(n_other_tokens)
        other_tokens = [OTHER_PREFIX + '_' + token_id for token_id in other_token_ids]

        # Shuffle tokens all_tokens
        output_tokens = completed_tokens
        all_tokens = set_tokens + reset_tokens + other_tokens + output_tokens + specials
        rng = np.random.default_rng(seed=_VOCAB_SEED)
        rng.shuffle(all_tokens)
        token_to_id = {t: i for i, t in enumerate(all_tokens)}
        id_to_token = all_tokens

        return Vocab(
            pad_token                = specials[0],
            token_to_id              = token_to_id,
            id_to_token              = id_to_token,
            set_tokens               = set_tokens,
            set_token_ids            = [token_to_id[tk] for tk in set_tokens],
            reset_token_ids          = [token_to_id[tk] for tk in reset_tokens],
            reset_tokens             = reset_tokens,
            other_tokens             = other_tokens,
            output_tokens            = output_tokens,
            set_token_to_seq_id      = set_token_to_seq_idx,
            set_token_to_pos_id      = set_token_to_pos_idx,
            reset_token_to_seq_id    = reset_token_to_seq_idx,
            output_token_to_seq_idxs = completed_token_to_seq_idxs,
            seq_idxs_to_output_token = seq_idxs_to_token
        )

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    def get_seq_id(self, token) -> int:
        if token.startswith(self.reset_prefix):
            return self.reset_token_to_seq_id[token]
        elif token.startswith(self.set_prefix):
            return self.set_token_to_seq_id[token]
        else:
            raise ValueError("`token` must be a valid 'set' or 'reset' token")

    def get_output_token(self, completed) -> int:
        seq_idxs = []
        for i, completed_i in enumerate(completed):
            if completed_i:
                seq_idxs.append(i)
        return self.seq_idxs_to_output_token[str(seq_idxs)]

    def get_pos_id(self, token) -> int:
        if token.startswith(self.set_prefix):
            return self.set_token_to_pos_id[token]
        else:
            raise ValueError("`token` must be a valid 'set' token")

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)


class _Random:
    """All random choices needed to generate the dataset."""

    def __init__(self, seed, set_tokens, reset_tokens, other_tokens, p_set, p_reset):
        self.set_tokens   = set_tokens
        self.reset_tokens = reset_tokens
        self.other_tokens = other_tokens
        self.p_set        = p_set
        self.p_reset      = p_reset

        self.rng = np.random.default_rng(seed=seed)

    def len(self, min_len, max_len) -> int:
        """Return a random length in [min_len, max_len]."""
        if min_len == max_len:
            return min_len
        else:
            return int(self.rng.integers(low=min_len, high=max_len))

    def token_kind(self) -> str:
        """Return the kind of token to generate."""
        r = self.rng.random()
        if r < self.p_set:
            return 'set'
        elif r < self.p_reset:
            return 'reset'
        else:
            return 'other'

    def token(self, token_kind) -> str:
        """Return a random token index."""
        if token_kind == 'set':
            index = self.rng.integers(len(self.set_tokens))
            return self.set_tokens[index]
        elif token_kind == 'reset':
            index = self.rng.integers(len(self.reset_tokens))
            return self.reset_tokens[index]
        else:
            index = self.rng.integers(len(self.other_tokens))
            return self.other_tokens[index]


