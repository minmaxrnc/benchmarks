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
from .utils import estimate_model_bytes


class Models(Factory, metaclass=SingletonMeta):

    def estimate_bytes(self, model):
        return estimate_model_bytes(model)


models = Models(__name__, classes=[])


def register_model(cls):
    """Register a model class so it can be used in experiments.

    Call this from your model package's __init__.py:

        from benchmarks import register_model
        from .mymodel import MyModel_LM
        register_model(MyModel_LM)

    The class name (cls.__name__) must match the 'class' field in meta/models.yaml.
    """
    models.add_class(cls)
