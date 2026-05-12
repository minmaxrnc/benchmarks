# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from typing import List, Optional, Tuple
from torch import optim
from .optimizer import Optimizer


class AdamW(Optimizer, optim.AdamW):

    def __init__(self, name, decay, no_decay, lr, weight_decay, **kwargs):
        params = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        super().__init__(name, params=params, lr=lr, _weight_decay=weight_decay, **kwargs)


    @staticmethod
    def get_properties():
        return {
            'requires_decay_groups': True
        }


