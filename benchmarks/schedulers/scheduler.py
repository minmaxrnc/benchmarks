# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import LRScheduler
from ..utils.properties import Properties


class Scheduler(ABC, Properties, LRScheduler):
    """Abstract class for learning-rate schedulers."""

    def __init__(
        self,
        name,
        *args,
        **kwargs
    ):
        self.name = name
        super().__init__(*args, **kwargs)


    @staticmethod
    def get_required_kwargs():
        return []


