# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Event gap, retention span and range dependency of Sequences datasets
(Monte Carlo; Appendix tab:longrange_stats).

Simulates the token distribution of the generator
(benchmarks/datasets/sequences.py, class Sequences) with the parameters of the
given dataset configurations (meta/meta.datasets.yaml): at each step a set
token is drawn with probability p_set, uniformly among the
n_distinct_seqs * n_ordered_seq * multiplicity set tokens. As in the
generator, reset tokens are never drawn when p_reset <= p_set (which holds for
every configured dataset; other configurations are rejected), and a sequence
counts as matched once n_distinct_seqs of its events have arrived in order
(docs/known-issues.md). The datasets themselves are not generated and their
seeds are not used: the statistics are those of the same distribution.

For each match (sequence s completed at step B):
  event gap      = B - A, with A the step of the latest first event of s
                   before B (the "first event" tracked by the generator,
                   which updates it when a first event re-occurs);
  retention span = T - 1 - B, the steps after the match up to the end of the
                   input sequence.
Range dependency: as computed by the generator (max over steps and sequences
of the steps since the tracked first event, or since the last relevant token
/ the start if no match is in progress; it keeps growing after a match).

Usage (from the repository root):
    python scripts/sequences_gap_stats.py [--n-sequences 20000] [--out stats.json]
    python scripts/sequences_gap_stats.py --datasets 'sequences_train_(1)' 'sequences_grad_train_(1)'
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent

# The datasets of tab:longrange_stats: Sequences(2) training, the long-range
# training/validation (same parameters), and the evaluation (same parameters
# for both benchmarks).
DEFAULT_DATASETS = ['sequences_train_(1)', 'sequences_val_(1)', 'sequences_grad_train_(1)',
                    'sequences_evaluation_(1)']


def bootstrap():
    os.chdir(REPO_DIR)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))


def dataset_params(name):
    from benchmarks.datasets.datasets import datasets
    meta = datasets.get_meta(name)
    if meta['class'] != 'Sequences':
        raise ValueError(f"{name} is not a Sequences dataset")
    args = meta['args']
    if args['p_reset'] > args['p_set']:
        raise NotImplementedError(f"{name}: p_reset > p_set (reset tokens would be drawn)")
    length = args['length']
    tmin, tmax = (length, length) if isinstance(length, int) else tuple(length)
    return {
        'p_set': args['p_set'], 'T_min': tmin, 'T_max': tmax,
        'n_distinct_seqs': args['n_distinct_seqs'], 'n_ordered_seq': args['n_ordered_seq'],
        'multiplicity': args['multiplicity'],
    }


def simulate(p_set, T_min, T_max, n_distinct_seqs, n_ordered_seq, multiplicity, n, rng):
    """Monte Carlo over n input sequences. T is drawn in [T_min, T_max) as by
    the generator (T_min if they are equal). Returns (event gaps, retention
    spans, range dependencies, fraction of sequences with a match)."""
    n_set = n_distinct_seqs * n_ordered_seq * multiplicity
    gaps, rets, rdeps, matched = [], [], [], 0
    for _ in range(n):
        T = T_min if T_min == T_max else int(rng.integers(T_min, T_max))
        if T <= 5000:                                # dense: one Bernoulli draw per step
            pos = np.flatnonzero(rng.random(T) < p_set)
        else:                                        # sparse: binomial count, uniform positions
            pos = np.unique(rng.integers(T, size=rng.binomial(T, p_set)))
        kind = rng.integers(n_set, size=len(pos))
        seq = kind // (n_ordered_seq * multiplicity)  # Vocab order: sequence, position, letter
        idx = (kind // multiplicity) % n_ordered_seq
        status = [0] * n_distinct_seqs
        first = [0] * n_distinct_seqs
        rdep, any_match = 0, False
        for t, s, i in zip(pos.tolist(), seq.tolist(), idx.tolist()):
            for q in range(n_distinct_seqs):         # ranges up to step t-1
                rdep = max(rdep, t - 1 - first[q])
            if status[s] == i:
                status[s] += 1
                if status[s] == n_distinct_seqs:     # the generator's completion check
                    gaps.append(t - first[s]); rets.append(T - 1 - t); any_match = True
            if status[s] == 1 and i == 0:            # the generator's tracked first event
                first[s] = t
            elif status[s] == 0:
                first[s] = t
        for q in range(n_distinct_seqs):
            rdep = max(rdep, T - 1 - first[q])
        rdeps.append(rdep)
        matched += any_match
    return np.array(gaps), np.array(rets), np.array(rdeps), matched / n


def _q(a, x):
    return float(np.quantile(a, x)) if len(a) else None


def _summary(a):
    if not len(a):
        return {'mean': None, 'median': None, 'p10': None, 'p90': None}
    return {'mean': float(a.mean()), 'median': _q(a, .5), 'p10': _q(a, .1), 'p90': _q(a, .9)}


def stats(name, n_sequences, rng, n_long=None):
    """Statistics of one dataset configuration. Datasets with sequences of
    200,000 steps or more use n_long sequences (default n_sequences // 4, at
    least 1,000), as for the evaluation row of the paper's table."""
    params = dataset_params(name)
    n = n_sequences
    if params['T_min'] >= 200000:
        n = n_long if n_long is not None else max(n_sequences // 4, 1000)
    g, r, rd, m = simulate(n=n, rng=rng, **params)
    gap = _summary(g)
    if len(g):
        gap |= {'min': int(g.min()), 'max': int(g.max()),
                'frac_below_1000': float((g < 1000).mean()), 'frac_below_100': float((g < 100).mean())}
    return {
        'n_sequences': n, **params, 'matched_fraction': m, 'n_matches': int(len(g)),
        'event_gap': gap, 'retention_span': _summary(r),
        'range_dependency_mean': float(rd.mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--n-sequences', type=int, default=20000)
    ap.add_argument('--n-long', type=int, default=None,
                    help='sequences for datasets of length >= 200,000 (default: n-sequences // 4, at least 1,000)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='JSON output path (default: stdout)')
    a = ap.parse_args(argv)
    out = Path(a.out).resolve() if a.out else None
    bootstrap()
    rng = np.random.default_rng(a.seed)
    result = {name: stats(name, a.n_sequences, rng, a.n_long) for name in a.datasets}
    text = json.dumps(result, indent=1)
    if out is None:
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return result


if __name__ == '__main__':
    main()
