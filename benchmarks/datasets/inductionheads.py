import torch
from dataclasses import dataclass
import numpy as np

from typing import List, Dict, Union, Tuple, Optional

from .utils.vocabulary import generate_letter_ids

from .dataset import Dataset, DatasetKindType, DatasetLengthType


class InductionHeads(Dataset):
    """
    Dataset from Mamba paper.
    """

    def __init__(
        self,
        name:     str,
        size:     int,
        length:   DatasetLengthType,
        seed:     int,
        kind:     DatasetKindType,
        # -- Loaded args below
        n_tokens: int,
        t_mem:    List[int],
        masked:   bool
    ):

        super().__init__(
            name,
            size,
            length,
            seed,
            kind
        )

        if len(t_mem) != 2:
            raise ValueError('Required: t_mem = [t_mem_min, t_mem_max]')

        self.n_tokens = n_tokens
        self.t_mem    = t_mem
        self.masked   = masked

        self.vocab = Vocab.build(
            self.n_tokens
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab.pad_id

    @staticmethod
    def get_iosize(**kwargs):
        return Vocab.get_size(
            n_tokens = kwargs['n_tokens']
        )

    def _makeitem(self, idx, seed) -> dict:
        rand = _Random(
            seed,
            self.vocab.tokens,
            self.min_len,
            self.max_len
        )

        T = rand.len()

        min_t_mem = self.t_mem[0]
        max_t_mem = min(T-1, self.t_mem[1])-3 # 2 positions left for storing and 1 for retrieving

        t_mem = rand.t_mem(min_t_mem, max_t_mem)
        mem_token_id = self.vocab.token_to_id[rand.token()]

        inputs = [self.vocab.token_to_id[rand.token()] for _ in range(T)]
        inputs[t_mem]   = self.vocab.return_id
        inputs[t_mem+1] = mem_token_id
        inputs[-1]      = self.vocab.return_id

        outputs = [self.vocab.pad_id] * (T-1) + [mem_token_id]
        mask    = [not self.masked]   * (T-1) + [True]

        stats = {
            'length':           T,
            'range_dependency': T - t_mem - 1
        }

        return {
            'inputs':  inputs,
            'outputs': outputs,
            'mask':    mask,
            'stats':   stats
        }

    @staticmethod
    def get_stats_kinds():
        return ['length', 'range_dependency']


_VOCAB_SEED = 42

PAD_TOKEN    = "_"
RETURN_TOKEN = "@"

@dataclass(frozen=True)
class Vocab:
    tokens:       List[str]
    token_to_id:  Dict[str, int]
    id_to_token:  List[str]
    pad_token:    str = PAD_TOKEN
    return_token: str = RETURN_TOKEN

    @property
    def return_id(self) -> int:
        return self.token_to_id[self.return_token]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    def __len__(self):
        return len(self.id_to_token)


    @staticmethod
    def get_size(**kwargs):
        """Same args as build."""
        return len(Vocab.build(**kwargs))

    @staticmethod
    def _get_special_tokens():
        return [PAD_TOKEN, RETURN_TOKEN]

    @staticmethod
    def build(n_tokens: int) -> "Vocab":

        specials = Vocab._get_special_tokens()
        tokens   = generate_letter_ids(n_tokens)

        # Shuffle all_tokens
        all_tokens = tokens + specials
        rng = np.random.default_rng(seed=_VOCAB_SEED)
        rng.shuffle(all_tokens)

        # Assign id's
        token_to_id = {t: i for i, t in enumerate(all_tokens)}
        id_to_token = all_tokens

        return Vocab(
            tokens       = tokens,
            token_to_id  = token_to_id,
            id_to_token  = id_to_token
        )


    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)


class _Random:
    """All random choices needed to generate the dataset."""

    def __init__(self, seed, tokens, min_len, max_len):
        self.tokens  = tokens
        self.min_len = min_len
        self.max_len = max_len
        self.rng     = np.random.default_rng(seed=seed)

    def len(self) -> int:
        """Return a random length in [min_len, max_len]."""
        if self.min_len == self.max_len:
            return self.min_len
        else:
            return int(self.rng.integers(low=self.min_len, high=self.max_len))

    def token(self) -> str:
        """Return a random token."""
        return self.rng.choice(self.tokens)

    def t_mem(self, min_t, max_t) -> str:
        """Return a random other token."""
        return int(self.rng.integers(low=min_t, high=max_t+1))



