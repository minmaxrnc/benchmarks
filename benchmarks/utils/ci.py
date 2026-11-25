# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import numpy as np
from .singleton import SingletonMeta
from . import config


class ConfidenceIntervals(metaclass=SingletonMeta):

    def __init__(self):
        self.rng = np.random.default_rng()

    def _get_rng(self, seed):
        """The bootstrap RNG. With seed=None, the `ci_seed` config key decides:
        None (the default) keeps one unseeded RNG for the whole process, as in
        the original runs; an int gives a fresh RNG seeded with it on every
        call, so that a CI depends only on the scores it is computed from."""
        if seed is None:
            seed = config.get('ci_seed')
        if seed is None:
            return self.rng
        return np.random.default_rng(seed)

    def mean_t(self, scores, delta):
        """Student-t CI."""
        x = np.asarray(scores, dtype=float)
        n = x.size
        mean = x.mean()
        se = x.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(1 - delta/2, df=n-1)
        return mean, mean - tcrit*se, mean + tcrit*se

    def mean_bootstrap(self, scores, delta, B=10000, seed=None):
        """Bootstrap CI."""
        rng = self._get_rng(seed)
        x = np.asarray(scores, dtype=float)
        n = x.size
        boots = np.empty(B)
        for b in range(B):
            boots[b] = rng.choice(x, size=n, replace=True).mean()
        low, high = np.percentile(boots, [100*delta/2, 100*(1-delta/2)])
        return x.mean(), low, high
