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

class Experiments(Factory, metaclass=SingletonMeta):

    @staticmethod
    def _str_as_repr():
        return True

    def get_models(self, experiment):
        models_trainers = self.get_meta(experiment)['args']['models_trainers']
        return [mt['model'] for mt in models_trainers]

    def get_models_trainers(self, experiment):
        return self.get_meta(experiment)['args']['models_trainers']

experiments = Experiments(__name__)

