import numpy as np
from .singleton import SingletonMeta


class ConfidenceIntervals(metaclass=SingletonMeta):

    def __init__(self):
        self.rng = np.random.default_rng()

    def mean_t(self, scores, delta):
        """Student-t CI."""
        x = np.asarray(scores, dtype=float)
        n = x.size
        mean = x.mean()
        se = x.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(1 - delta/2, df=n-1)
        return mean, mean - tcrit*se, mean + tcrit*se

    def mean_bootstrap(self, scores, delta, B=10000):
        """Bootstrap CI."""
        x = np.asarray(scores, dtype=float)
        n = x.size
        boots = np.empty(B)
        for b in range(B):
            boots[b] = self.rng.choice(x, size=n, replace=True).mean()
        low, high = np.percentile(boots, [100*delta/2, 100*(1-delta/2)])
        return x.mean(), low, high


