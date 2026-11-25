# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""torch.compile with the Inductor settings that a model asks for.

A model can return Inductor config overrides from inductor_config()
(benchmarks.models.model.Model); they are applied, process-wide, before the
model is compiled. The MinMax RNC wrapper uses this for the minmaxrnc 0.4
known issue: with neuron arrays of state dimension d > 1, Inductor fails in
fp32 inference (e.g. validation and evaluation under torch.no_grad()) unless
`torch._inductor.config.triton.coalesce_tiling_analysis` is False.
"""

import torch


def apply_inductor_config(overrides):
    """Set torch._inductor.config entries given as {'a.b': value}. Entries this
    torch version does not have are skipped. Returns the entries set."""
    if not overrides:
        return {}
    import torch._inductor.config as inductor_config
    applied = {}
    for key, value in overrides.items():
        *path, attr = key.split('.')
        obj = inductor_config
        try:
            for part in path:
                obj = getattr(obj, part)
            if not hasattr(obj, attr):
                raise AttributeError(attr)
        except AttributeError:
            print(f"Inductor config '{key}' not available in torch {torch.__version__}: skipped")
            continue
        setattr(obj, attr, value)
        applied[key] = value
    if applied:
        print(f"Inductor config set for this model: {applied}")
    return applied


def compile_model(model):
    """torch.compile(model), after applying model.inductor_config() if any."""
    get_overrides = getattr(model, 'inductor_config', None)
    if callable(get_overrides):
        apply_inductor_config(get_overrides())
    return torch.compile(model)
