# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch

def optimal_io_dtype(vocab_size):
        io_dtype = None
        for dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
            if vocab_size <= torch.iinfo(dtype).max:
                io_dtype = dtype
                break
        if io_dtype is None:
            raise Exception(
                f"Vocabulary is too large to be represted as a uint: vocab_size={vocab_size}"
            )
        return io_dtype

