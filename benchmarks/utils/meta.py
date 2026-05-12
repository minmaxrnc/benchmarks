# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from ..definitions import PROJECT_ROOT
from ..definitions import META_IMPORT_LITERAL
from ..definitions import META_MAIN_FILENAME
from ..definitions import META_ALL_LITERAL
from ..definitions import META_AS_LITERAL
from ..definitions import META_TYPE_SEPARATOR
from ..definitions import META_DATE_FORMAT as DATE_FORMAT

from .singleton import SingletonMeta

from .templatestrings import match_template

class Meta(dict):

    def __init__(self, meta):
        self._meta = meta
        self.__cache_compiled_meta = {}


    def __getitem__(self, instance_name):

        def __evaluate(s, parentheses=True):
            i = 0
            result = ''
            while i < len(s):
                if s[i] == '(':
                    j = s.index(')', i)
                    partial = ''
                    if s.find('+',i+1,j) >= 0:
                        k = s.find('+',i+1,j)
                        partial = str(int(s[i+1:k]) + int(s[k+1:j]))
                    elif s.find('-',i+1,j) >= 0:
                        k = s.find('-',i,j)
                        partial = str(int(s[i+1:k]) - int(s[k+1:j]))
                    elif s.find('*',i+1,j) >= 0:
                        k = s.find('*',i+1,j)
                        partial = str(int(s[i+1:k]) * int(s[k+1:j]))
                    else:
                        partial = s[i+1:j]

                    if parentheses:
                        result += '(' + partial + ')'
                    else:
                        result += partial
                    i = j+1
                else:
                    result += s[i]
                    i += 1
            return result

        def __replace(_entry, params):
            if type(_entry) == dict:
                new_entry = {}
                for _key in _entry:
                    new_entry[_key] = __replace(_entry[_key], params)
                return new_entry
            elif type(_entry) == list:
                new_entry = []
                for _x in _entry:
                    new_entry.append(__replace(_x, params))
                return new_entry
            elif type(_entry) == str:
                if (_entry[0] == '{' and _entry[-1] == '}'):
                    if _entry[1:-1] in params:
                        return params[_entry[1:-1]]
                    else:
                        raise Exception(f"Failed to replace {_entry[1:-1]}")
                else:
                    return __evaluate(_entry.format(**params))
            else:
                return _entry


        compiled_instance_name = __evaluate(instance_name)
        if compiled_instance_name not in self.__cache_compiled_meta:
            matched_keys = []
            for instance_key in self._meta:
                params = match_template(instance_key, compiled_instance_name)
                if params is not None:
                    matched_keys.append(instance_key)
                    compiled_meta = __replace(self._meta[instance_key], params)

            if len(matched_keys) == 1:
                self.__cache_compiled_meta[compiled_instance_name] = compiled_meta
            elif len(matched_keys) == 0:
                raise Exception(f"Not found: {compiled_instance_name} in available keys {list(self._meta.keys())}")
            else:
                raise Exception(f"Multiple matches for {compiled_instance_name}: {matched_keys}")

        return self.__cache_compiled_meta[compiled_instance_name]

    def __repr__(self):
        return repr(self._meta)

    def __len__(self):
        return len(self._meta)

    def copy(self):
        return self._meta.copy()

    def has_key(self, k):
        return k in self._meta

    def keys(self):
        return self._meta.keys()

    def values(self):
        return self._meta.values()

    def items(self):
        return self._meta.items()

    def __iter__(self):
        return iter(self._meta)

    def __unicode__(self):
        return unicode(repr(self._meta))


    def __delitem__(self, key):
        raise Exception('This method cannot be called')

    def clear(self):
        raise Exception('This method cannot be called')

    def update(self, *args, **kwargs):
        raise Exception('This method cannot be called')

    def pop(self, *args):
        raise Exception('This method cannot be called')

    def __cmp__(self, dict_):
        raise Exception('This method cannot be called')

    def __contains__(self, item):
        raise Exception('This method cannot be called')

    def __setitem__(self, key, item):
        raise Exception('This method cannot be called')


class _Meta(metaclass=SingletonMeta):

    __cache = {}

    def __init__(self):
        self.cache = {}
        self.cache_onlyenabled = {}


    def load(self, scope=None, only_enabled=False):
        if scope is None:
            scope = '_none'
        if only_enabled:
            if scope not in self.cache_onlyenabled:
                self.cache_onlyenabled[scope] = self._load(scope, only_enabled)
            return Meta(deepcopy(self.cache_onlyenabled[scope]))
        else:
            if scope not in self.cache:
                self.cache[scope] = self._load(scope, only_enabled)
            return Meta(deepcopy(self.cache[scope]))


    @staticmethod
    def _load(scope=None, only_enabled=False):
        """
        Default scope loads the entire meta
        """

        def _isterminal(meta_entry):
            return not isinstance(meta_entry, dict)

        def _istoload(meta_entry):
            if not only_enabled:
                return True
            if 'enabled' not in meta_entry:
                return True
            return meta_entry['enabled']

        def _import_some(query, data):
            result = {}
            for key in query:
                if META_AS_LITERAL in query[key]:
                    new_key = query[key][META_AS_LITERAL]
                    result[new_key] = data[key]
                else:
                    result |= _import_some(query[key], data[key])
            return result

        def _import(base_path, path, meta_entry):
            result = {}
            for subpath in meta_entry:
                if subpath[0:2] == '..':
                    # relative import (2)
                    full_path = '.'.join(base_path.split('.')[:-1]) + subpath[1:]
                elif subpath[0] == '.':
                    # relative import (1)
                    full_path = base_path + subpath
                else:
                    # absolute import
                    full_path = subpath
                if full_path not in _Meta.__cache:
                    with open(full_path + '.yaml') as f:
                        _Meta.__cache[full_path] = yaml.safe_load(f) or {}
                imported_entry = _Meta.__cache[full_path]

                if meta_entry[subpath] == META_ALL_LITERAL:
                    # import all fields
                    new_key = subpath.split('.')[-1] # last element of path
                    result[new_key] = imported_entry
                else:
                    # import only some fields
                    result |= _import_some(meta_entry[subpath], imported_entry)
            return result

        def _instantiate_params(params, template_entry):
            if type(template_entry) != dict and type(template_entry) != list:
                if type(template_entry) == str:
                    return template_entry.format(**params)
                else:
                    return template_entry


            if type(template_entry) == dict:
                result = {}
                for key in template_entry:
                    result[key] = _instantiate_params(params, template_entry[key])
                return result

            if type(template_entry) == list:
                result = []
                for x in template_entry:
                    result.append(_instantiate_params(params, x))
                return result


        def _compile_from_template(base_path, path, entry_key):

            if path not in _Meta.__cache:
                with open(path + '.t.yaml') as f:
                    _Meta.__cache[path] = yaml.safe_load(f) or {}
            template_entry = _Meta.__cache[path]

            result = None
            for template in template_entry:
                params = match_template(template, entry_key)
                if params is not None:
                    if result is not None:
                        raise Exception(f"Duplicate template for '{entry_key}': '{result}' and '{template}'")
                    result = _instantiate_params(params, template_entry[template])

            if result is None:
                raise Exception(f"Not found template for '{entry_key}'")

            return result


        def _compile(base_path, path, meta_entry):
            result = {}
            for key in meta_entry:
                if key == META_IMPORT_LITERAL:
                    result |= _import(base_path, path, meta_entry[key])
                else:
                    untyped_key, *entry_type = key.rsplit(META_TYPE_SEPARATOR, 1)
                    entry_type = entry_type[0] if len(entry_type) > 0 else None
                    if _isterminal(meta_entry[key]):
                        # if entry_type == META_DATE_TYPE:
                        #     result[untyped_key] = datetime.strptime(
                        #             meta_entry[key],
                        #             DATE_FORMAT
                        #             )
                        # else:
                        result[untyped_key] = meta_entry[key]
                    elif _istoload(meta_entry[key]):
                        subpath = path + '.' + untyped_key
                        if 'from_template' in meta_entry[key] and meta_entry[key]['from_template']:
                            result[key] = _compile(
                                base_path,
                                subpath,
                                _compile_from_template(
                                    base_path,
                                    path,
                                    key
                                )
                            )
                        else:
                            result[key] = _compile(base_path, subpath, meta_entry[key])
            return result

        p = Path(META_MAIN_FILENAME)
        meta_main_filename_no_extension = os.path.join(p.parent, p.stem)
        if scope is not None:
            meta_relevant_filename = meta_main_filename_no_extension + '.' + scope
        else:
            meta_relevant_filename = meta_main_filename_no_extension
        path = os.path.join(PROJECT_ROOT, meta_relevant_filename)

        # call compile with META_IMPORT_LITERAL to trigger initial loading
        with open(path + '.yaml') as f:
            meta_data = yaml.safe_load(f) or {}
            meta = _compile(path, path, meta_data)

        return meta


_meta = _Meta()

# Public attribute
load = _meta.load

