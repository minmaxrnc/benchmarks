# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch

def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def estimate_sample_bytes(dataset):
    input_bytes  = _nbytes(dataset.data[0]['inputs'])
    output_bytes = _nbytes(dataset.data[0]['outputs'])
    mask_bytes   = _nbytes(dataset.data[0]['mask'])
    return input_bytes + output_bytes + mask_bytes

def compute_sample_bytes(sample):
    input_bytes  = _nbytes(sample['inputs'])
    output_bytes = _nbytes(sample['outputs'])
    mask_bytes   = _nbytes(sample['mask'])
    return input_bytes + output_bytes + mask_bytes
