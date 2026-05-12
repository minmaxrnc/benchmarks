# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import torch
from ..utils.device import device
from .metric import Metric
import warnings


class TokenAccuracy(Metric):

    def __init__(self, name):
        super().__init__(name)
        self.correct = torch.tensor(0.0, dtype=torch.long, device=device)
        self.total   = torch.tensor(0,   dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """
        logits: [B, T, C]
        labels: [B, T]
        mask:   [B, T] (bool or {0,1})
        """

        pred    = logits.argmax(dim=-1)  # [B, T]
        correct = (pred == labels) & mask

        self.correct += correct.sum().to(device=self.correct.device, dtype=self.correct.dtype)
        self.total   += mask.sum().to(device=self.total.device,      dtype=self.total.dtype)
        return self


    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        if self.total.item() == 0:
            return 0.0
        acc = self.correct.to(dtype=torch.double) / self.total.to(dtype=torch.double)
        return acc.item()


    @torch.no_grad()
    def reset(self):
        self.correct.zero_()
        self.total.zero_()
        return self

