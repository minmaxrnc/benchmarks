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

class StepLR(Scheduler, lr_scheduler.StepLR):
    """Wrapper of StepLR scheduler."""

    def __init__(
            self,
            name,
            *args,
            **kwargs
            ):
        super().__init__(name, *args, **kwargs)


    @staticmethod
    def get_properties():
        return {
            'step': 'epoch'
        }

