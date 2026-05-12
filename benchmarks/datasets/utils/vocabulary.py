# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from dataclasses import dataclass
from typing import Union, Sequence, List, Tuple, Dict

def generate_letter_ids(n: int):
    return [to_base26(i) for i in range(n)]


def to_base26(n: int) -> str:
    digits = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if n < 0:
        raise ValueError("Parameter `n` must be non-negative")
    if n == 0:
        return digits[0]
    out = []
    while n:
        n, r = divmod(n, 26)
        out.append(digits[r])
    return "".join(out)



PAD_TOKEN       = "_"
UNDEFINED_TOKEN = "?"


@dataclass(frozen=True)
class Vocab:
    pad_token: str = PAD_TOKEN
    undefined_token: str = UNDEFINED_TOKEN
    token_to_id: Dict[str, int] = None
    id_to_token: List[str] = None
    tokens: List[str] = None

    def __len__(self):
        return len(self.id_to_token)

    @staticmethod
    def get_size(vocab_size: int):
        """Same args as build."""
        return len(Vocab.build(vocab_size))

    @staticmethod
    def build(vocab_size: int) -> "Vocab":
        specials = [PAD_TOKEN, UNDEFINED_TOKEN]
        tokens = [to_base26(i) for i in range(vocab_size)]
        all_tokens = tokens + specials
        token_to_id = {t: i for i, t in enumerate(all_tokens)}
        id_to_token = all_tokens
        return Vocab(
            pad_token=specials[0],
            undefined_token=specials[1],
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            tokens=tokens
        )

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def undefined_id(self) -> int:
        return self.token_to_id[self.undefined_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)



