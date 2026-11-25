# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Smoke-test ("tiny") versions of the meta entries.

With BENCHMARKS_TINY=1 (set by `python main.py reproduce <family> --tiny`),
every factory reads its meta through TinyMeta, which shrinks what makes a run
expensive and keeps everything else (models, trainers' recipes, compilation,
precision, chunked training, protocol overrides) as configured:

  * datasets: a few sequences of a few dozen steps (evaluation: 256 steps);
  * experiments: 2 epochs, one seed, no DataLoader workers, short time limits;
  * trainers: batches of at most 4 sequences, training chunks of at most 16
    steps when max_unroll_steps is set (so that chunked training still runs);
  * evaluators: chunks of 64 steps.

The numbers it produces are meaningless; it only checks that the pipeline
(generate -> train -> evaluate) runs end to end.
"""

from copy import deepcopy


TINY_DATASET_SIZE   = {'train': 16, 'val': 8, 'evaluation': 4}
TINY_DATASET_LENGTH = {'train': (24, 32), 'val': (40, 48), 'evaluation': (256, 256)}
TINY_EPOCHS         = 2
TINY_BATCH_SIZE     = 4
TINY_UNROLL_STEPS   = 16
TINY_CHUNK_SIZE     = 64
TINY_PROTOCOL = {
    'ci_min_n': 1,
    'ci_max_n': 1,
    'num_workers': 0,
    'max_train_runtime': 1800,
    'max_train_runtime__per_params': 1800,
    'max_train_runtime__per_seed': 1800,
    'max_eval_runtime': 1800,
}


def _tiny_length(kind, length):
    low, high = TINY_DATASET_LENGTH[kind]
    if isinstance(length, int):
        return high
    if length[0] == length[1]:
        return [high, high]
    return [low, high]


def _scale_list(value, fn):
    if isinstance(value, list):
        return [fn(v) for v in value]
    return fn(value)


def shrink(scope, entry):
    """The tiny version of a compiled meta entry of the given scope."""
    entry = deepcopy(entry)
    args = entry.get('args')
    if not isinstance(args, dict):
        return entry
    if scope == 'datasets' and 'kind' in args:
        kind = args['kind']
        args['size'] = min(args['size'], TINY_DATASET_SIZE[kind])
        args['length'] = _tiny_length(kind, args['length'])
    elif scope == 'experiments':
        args['epochs'] = min(args['epochs'], TINY_EPOCHS)
        args['protocol'] = dict(args.get('protocol') or {}) | TINY_PROTOCOL
    elif scope == 'trainers':
        if 'train_batch_size' in args:
            args['train_batch_size'] = _scale_list(args['train_batch_size'],
                                                   lambda b: min(b, TINY_BATCH_SIZE))
        if args.get('max_unroll_steps') is not None:
            args['max_unroll_steps'] = min(args['max_unroll_steps'], TINY_UNROLL_STEPS)
    elif scope == 'evaluators' and 'chunk_size' in args:
        args['chunk_size'] = min(args['chunk_size'], TINY_CHUNK_SIZE)
    return entry


class TinyMeta:
    """Read-only view of a benchmarks.utils.meta.Meta whose entries are
    shrunk by `shrink` (only item access is used by the factories)."""

    def __init__(self, meta, scope):
        self._meta  = meta
        self._scope = scope

    def __getitem__(self, name):
        return shrink(self._scope, self._meta[name])

    def keys(self):
        return self._meta.keys()

    def __iter__(self):
        return iter(self._meta)

    def __len__(self):
        return len(self._meta)
