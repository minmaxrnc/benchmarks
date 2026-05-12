# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta


class Evaluations(Factory, metaclass=SingletonMeta):
    @staticmethod
    def _str_as_repr():
        return True

evaluations = Evaluations(__name__)

