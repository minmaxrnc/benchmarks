# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch


# Set device: cuda or cpu (preference in this order)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

pin_memory = torch.cuda.is_available()

