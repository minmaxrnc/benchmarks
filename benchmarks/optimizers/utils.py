# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.


import torch
from collections import defaultdict

def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def estimate_bytes(optim: torch.optim.Optimizer):
    """
    Exact if states are initialized (after first step); otherwise approximate common cases.
    """
    # Try exact first
    exact = 0
    for st in optim.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                exact += _nbytes(v)
    if exact > 0:
        return exact

    # Approximation before first step
    # Most optimizers keep states in FP32 regardless of model dtype.
    approx = 0
    for group in optim.param_groups:
        for p in group["params"]:
            if not isinstance(p, torch.Tensor):
                continue
            p_bytes_fp32 = p.numel() * 4

            # Adam/AdamW/RAdam: m,v -> ~2x params (in FP32)
            if isinstance(optim, (torch.optim.Adam, torch.optim.AdamW, torch.optim.RAdam)):
                approx += 2 * p_bytes_fp32
            # SGD with momentum: momentum buffer -> ~1x params (in FP32)
            elif isinstance(optim, torch.optim.SGD) and getattr(optim, "momentum", 0) != 0:
                approx += p_bytes_fp32
            # Adagrad: also ~2x
            elif isinstance(optim, torch.optim.Adagrad):
                approx += 2 * p_bytes_fp32
            # Otherwise, assume no state
            else:
                approx += 0


    return approx


