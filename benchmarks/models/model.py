# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Model(nn.Module):
    """Abstract class for models, defining their interface."""

    def __init__(self, name, *args, **kwargs):
        if type(self) == Model:
            raise Exception('Attempted to instantiate abstract class')
        self.name = name
        super().__init__(*args, **kwargs)


    def create_weight_decay_optim_groups(self):
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                if p.ndim >= 2 and "norm" not in name.lower():
                    decay.append(p)        # weight matrices, conv kernels, etc.
                else:
                    no_decay.append(p)     # biases, LayerNorm/BatchNorm gamma/beta
        return decay, no_decay


    def forward(self, x: torch.Tensor, state=None, unroll_steps: int = -1, return_state: bool = False) -> torch.Tensor:
        raise Exception('It must be implemented')

    @staticmethod
    def get_required_kwargs() -> list:
        return []

    def supports_unroll_steps(self):
        return False


