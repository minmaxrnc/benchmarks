# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.


def format_accuracy(accuracy, p=4):

    f = '{:.' + str(p) + 'f}'
    s = '1.' + '0' * p

    if f.format(accuracy) == s and accuracy < 1.0:
        return s + '~'
    else:
        return f.format(accuracy)

def format_loss(loss, p=4):

    f = '{:.' + str(p) + 'f}'
    s = '0.' + '0' * p

    if f.format(loss) == s and loss > 0.0:
        return s + '~'
    else:
        return f.format(loss)

