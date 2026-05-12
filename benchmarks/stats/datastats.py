# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import sys
from ..utils import meta
import warnings
from typing import List, Union, Dict


def instantiate(dataset_size, stats_kinds):
    return DataStats(dataset_size, stats_kinds)


class DataStats:

    def __init__(self, dataset_size,  stats_kinds):
        self.dataset_size = dataset_size
        self.partials = {}
        for stats_kind in stats_kinds:
            self.partials[stats_kind] = {
                'total': 0,
                'count': 0,
                'max': 0,
                'min': float('inf')
            }

    def update(self, stats: Dict[str, Union[Union[float, int], List[Union[float, int]]]]):
        for stats_kind in self.partials:
            stats_v = stats[stats_kind]
            if type(stats_v) != list:
                stats_v = [stats_v]
            self.partials[stats_kind]['total'] += sum(stats_v)
            self.partials[stats_kind]['count'] += len(stats_v)
            self.partials[stats_kind]['max']    = max(self.partials[stats_kind]['max'], max(stats_v))
            self.partials[stats_kind]['min']    = min(self.partials[stats_kind]['min'], min(stats_v))

    def compute(self) -> dict:
        result = {}
        for stats_kind in self.partials:
            partials = self.partials[stats_kind]
            result[stats_kind] = {
                'avg': partials['total'] / partials['count'] if partials['count'] > 0 else 0,
                'max': partials['max'],
                'min': int(partials['min']) if partials['count'] > 0 else 0
            }
            result['size'] = self.dataset_size
        return result

