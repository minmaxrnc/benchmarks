# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.


from .emastopper import EmaStopper


def instantiate(name):
    if name != 'emastopper':
        raise ValueError("The only stopper is `emastopper'")
    return EmaStopper()

