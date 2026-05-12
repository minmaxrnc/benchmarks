# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import numpy as np
import random
import torch
import os


def seed_everything(seed: int = 42, deterministic: bool = False):
    # 1) Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # if you use multi-GPU

    # 3) Make CUDA/cuDNN deterministic (slower but reproducible)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enforce deterministic algorithms (PyTorch 1.12+)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # ok on older versions

        # Optional: if PyTorch warns about cuBLAS nondeterminism on your setup:
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

class Seeds:

    def __init__(self, max_seeds=None):
        self.seed = 0
        self.max_seeds = max_seeds

    def __iter__(self):
        return self

    def __next__(self):
        if self.seed >= self.max_seeds:
            raise StopIteration
        self.seed += 1
        seed = self.seed
        return seed

    def get_used_seeds(self) -> int:
        return self.seed


