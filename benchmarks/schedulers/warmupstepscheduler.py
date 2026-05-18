# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from .scheduler import Scheduler


class WarmupStepLR(Scheduler):
    """StepLR with linear warmup over a fraction of total training epochs.

    During the warmup phase the learning rate rises linearly from
    base_lr / warmup_steps to base_lr.  After warmup, standard StepLR
    decay is applied (counting epochs from the end of warmup).

    Args:
        epochs          – total number of training epochs (injected by trainer)
        step_size       – decay period in epochs (post-warmup)
        gamma           – multiplicative decay factor
        warmup_fraction – fraction of total epochs used for warmup (default 0.1)
    """

    def __init__(
        self,
        name:             str,
        optimizer,
        epochs:           int,
        step_size:        int,
        gamma:            float = 0.1,
        warmup_fraction:  float = 0.1,
    ):
        self._warmup_steps = max(1, round(warmup_fraction * epochs))
        self._step_size    = step_size
        self._gamma        = gamma
        super().__init__(name, optimizer=optimizer)

    def get_lr(self) -> list[float]:
        e = self.last_epoch
        if e < self._warmup_steps:
            scale = (e + 1) / self._warmup_steps
        else:
            scale = self._gamma ** ((e - self._warmup_steps) // self._step_size)
        return [base_lr * scale for base_lr in self.base_lrs]

    @staticmethod
    def get_required_kwargs() -> list[str]:
        return ['epochs']

    @staticmethod
    def get_properties() -> dict:
        return {'step': 'epoch'}
