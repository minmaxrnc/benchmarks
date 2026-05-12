# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import sys
from copy import deepcopy
from math import log, ceil

from . import meta
from ..utils.misc import str_class, repr_class

from ..utils.properties import Properties

from .templatestrings import match_template


class Factory:

    _str_as_repr = False

    def __init__(self, module_name, classes=[]):

        self._name    = module_name.split('.')[-1]
        self._meta    = meta.load(self._name)
        self._classes = {_class.__name__: _class for _class in classes}


    def instantiate(self, instance_name, *runtime_args, **runtime_kwargs):
        Class = self._get_class(instance_name)
        loaded_kwargs = self._load_kwargs(instance_name)
        return Class(instance_name, *runtime_args, **loaded_kwargs, **runtime_kwargs)


    def get_meta(self, instance):
        if type(instance) == str:
            instance_name = instance
        else:
            instance_name = instance.name
        return deepcopy(self._meta[instance_name])


    def get_property(self, instance, property_name):
        if type(instance) == str:
            instance_name = instance
        else:
            instance_name = instance.name
        Class = self._get_class(instance_name)
        if property_name in self._meta[instance_name]:
            return deepcopy(self._meta[instance_name][property_name])
        elif issubclass(Class, Properties) and property_name in Class.get_properties():
            return deepcopy(Class.get_properties()[property_name])
        else:
            raise Exception(f"{instance_name} does not have property {property_name}")


    def get_required_kwargs(self, instance):
        if type(instance) == str:
            instance_name = instance
        else:
            instance_name = instance.name
        Class = self._get_class(instance_name)
        return Class.get_required_kwargs()


    def add_class(self, _class):
        self._classes[_class.__name__]  = _class

    def add_classes(self, *classes):
        self._classes |= {_class.__name__: _class for _class in classes}


    def repr(self, instance):
        if type(instance) == str:
            instance_name = instance
        else:
            instance_name = instance.name
        return self._repr(instance_name)

    def _normalise_name(self, instance_name):
        return instance_name.replace(')','').replace('(','')

    def _repr(self, instance_name):
        return self._normalise_name(instance_name)

    @staticmethod
    def _str_as_repr():
        return False

    def str(self, instance):
        if self._str_as_repr():
            return self.repr(instance)

        if type(instance) == str:
            instance_name = instance
        else:
            instance_name = instance.name
        return self._str(instance_name)

    def _str(self, instance_name):
        res = str_class(self._meta[instance_name]['class'], self._meta[instance_name]['args'])
        return res.replace(instance_name, instance_name.replace(')','').replace('(',''))


    def _get_class(self, instance):
        if type(instance) == str:
            instance_name = instance
        else:
            instance_name = instance.name
        class_name = self._meta[instance_name]['class']
        return self._classes[class_name]


    def _load_kwargs(self, instance_name):
        return self._meta[instance_name]['args']

