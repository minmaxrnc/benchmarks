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

def _unique_tensors(tensors):
    # Avoid double-counting shared / tied weights
    seen, uniq = set(), []
    for t in tensors:
        if not torch.is_tensor(t):
            continue
        ptr = t.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            uniq.append(t)
    return uniq


def estimate_model_bytes(model: torch.nn.Module):
    params = _unique_tensors([p.detach() for p in model.parameters()])
    buffers = _unique_tensors(list(model.buffers()))

    param_bytes  = sum(_nbytes(p) for p in params)
    buffer_bytes = sum(_nbytes(b) for b in buffers)
    # Gradients exist only for params that require grad; they match param dtype by default
    grad_bytes   = sum(_nbytes(p) for p in model.parameters() if p.requires_grad)

    return {
        "parameters": param_bytes,
        "buffers": buffer_bytes,
        "gradients_if_training": grad_bytes,
        "model_total_if_training": param_bytes + buffer_bytes + grad_bytes,
        "model_total_inference": param_bytes + buffer_bytes,
    }


