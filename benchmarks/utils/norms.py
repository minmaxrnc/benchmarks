# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""
Model-agnostic helpers for reading a norm out of a recurrent state.

A model's `state` (per the Model.forward(..., return_state=True) contract)
can be any nesting of tensors/dicts/lists -- e.g. MinMax RNC returns
list[layer]{'neuron': Tensor, 'conv': Tensor}, VanillaRNN returns a plain
Tensor. sum_of_squares() flattens whatever it's given and returns the total
sum of squares of every element found, as a 0-dim tensor left on its
original device -- callers accumulate these across a batch/epoch/chunk loop
and only call .item()/.cpu() once at the end, so no extra device sync is
introduced mid-loop.
"""

import torch


def flatten_tensors(x):
    """Yield every torch.Tensor found in a (possibly nested list/dict/tuple) structure."""
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from flatten_tensors(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from flatten_tensors(v)


def sum_of_squares(x):
    """Sum of squares across every tensor found in x. 0-dim tensor on x's
    device, or None if x is None / contains no tensors."""
    total = None
    for t in flatten_tensors(x):
        sq = t.detach().float().pow(2).sum()
        total = sq if total is None else total + sq
    return total
