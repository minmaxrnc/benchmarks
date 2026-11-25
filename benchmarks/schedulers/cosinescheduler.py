# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from torch.optim import lr_scheduler
from .scheduler import Scheduler


class CosineAnnealingLR(Scheduler, lr_scheduler.CosineAnnealingLR):
    """Cosine decay of the learning rate from base_lr down to eta_min over
    `epochs` epochs (T_max = epochs, injected by the trainer).

    Args:
        epochs  – total number of training epochs (injected by trainer), used
                  as T_max
        eta_min – minimum learning rate reached at the end of decay (default 0.0)
    """

    def __init__(
        self,
        name:    str,
        optimizer,
        epochs:  int,
        eta_min: float = 0.0,
    ):
        super().__init__(name, optimizer=optimizer, T_max=epochs, eta_min=eta_min)


    @staticmethod
    def get_required_kwargs() -> list[str]:
        return ['epochs']


    @staticmethod
    def get_properties() -> dict:
        return {
            'step': 'epoch'
        }
