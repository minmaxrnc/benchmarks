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

from .tokenaccuracy import TokenAccuracy
from .chunkedtokenaccuracy import ChunkedTokenAccuracy

class Metrics(Factory, metaclass=SingletonMeta):
    pass

metrics = Metrics(__name__,classes=[
    TokenAccuracy,
    ChunkedTokenAccuracy
])

