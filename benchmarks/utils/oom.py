# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Recognising out-of-memory errors.

Kernels compiled by torch.compile (Triton) do not always report running out of
memory as torch.cuda.OutOfMemoryError: CUDA errors raised inside a Triton kernel
launch surface as plain RuntimeErrors ("Triton Error [CUDA]: out of memory"),
and Triton's own resource limits (shared memory, registers) raise
OutOfResources. The retry loops of the trainer and the evaluator catch
OOM_EXCEPTION_TYPES and re-raise anything for which is_oom_error is False.
"""

import torch

try:
    from triton.runtime.errors import OutOfResources as _TritonOutOfResources
except ImportError:
    _TritonOutOfResources = None


OOM_EXCEPTION_TYPES = (RuntimeError,) + (
    (_TritonOutOfResources,) if _TritonOutOfResources is not None else ()
)


def is_oom_error(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if _TritonOutOfResources is not None and isinstance(e, _TritonOutOfResources):
        return True
    message = str(e).lower()
    return 'out of memory' in message or 'out of resource' in message
