# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from ..optimizers.optimizer import Optimizer
from .scheduler import Scheduler


class NoneScheduler(Scheduler):
    """A scheduler that leaves the lr unchanged."""

    def __init__(self, name, optimizer: Optimizer) -> None:
        super().__init__(name, optimizer=optimizer)


    def get_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


    def step(self):
        if not self._is_initial:
            raise Exception('This should never be called except during init')



    @staticmethod
    def get_properties():
        return {
            'step': 'never'
        }


