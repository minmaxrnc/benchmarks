# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""GroupStateTracking (the Word(G) task): labels against a brute-force
computation with explicit permutations, and the cosine scheduler used by the
groups trainers. Items are generated in memory (no saved datasets)."""
import itertools

import pytest
import torch

from benchmarks.datasets.groupstatetracking import GroupStateTracking
from benchmarks.schedulers.cosinescheduler import CosineAnnealingLR

pytestmark = pytest.mark.group('groups')


def _elements(n):
    """S_n in the documented order: the identity first, then the other
    permutations of range(n) in lexicographic order."""
    identity = tuple(range(n))
    return [identity] + [p for p in itertools.permutations(range(n)) if p != identity]


def _compose(p, q):
    """p o q: apply q, then p."""
    return tuple(p[q[i]] for i in range(len(q)))


def _dataset(group, length=12, size=5, seed=1):
    return GroupStateTracking('word', size=size, length=[length, length], seed=seed,
                              kind='train', group=group)


@pytest.mark.parametrize('group, n', [('S3', 3), ('S4', 4)])
@pytest.mark.parametrize('idx', range(5))
def test_labels_are_the_running_product(group, n, idx):
    ds = _dataset(group)
    item = ds[idx]
    x, y = item['inputs'].tolist(), item['outputs'].tolist()
    elements = _elements(n)
    assert len(x) == len(y) == 12
    gamma = elements[0]
    for t, (x_t, y_t) in enumerate(zip(x, y)):
        sigma = elements[x_t]
        # recurrence: gamma_t = gamma_{t-1} o sigma_t
        gamma = _compose(gamma, sigma)
        assert elements[y_t] == gamma
        # closed form, from scratch: gamma_t = sigma_1 o ... o sigma_t
        product = elements[0]
        for s in x[:t + 1]:
            product = _compose(product, elements[s])
        assert product == gamma


def test_order_matters_on_a_non_abelian_group():
    # S_3 is not abelian, so the reversed product sigma_t o ... o sigma_1
    # differs from the label for some sequence.
    elements = _elements(3)
    ds = _dataset('S3', length=16, size=20)
    differs = False
    for idx in range(len(ds)):
        x, y = ds[idx]['inputs'].tolist(), ds[idx]['outputs'].tolist()
        rev = elements[0]
        for s in x:
            rev = _compose(elements[s], rev)
        differs |= elements[y[-1]] != rev
    assert differs


def test_every_step_is_labelled_and_tokens_are_elements():
    ds = _dataset('S4', length=20)
    item = ds[0]
    assert bool(item['mask'].all())
    assert int(item['inputs'].max()) < 24 and int(item['outputs'].max()) < 24


def test_vocabulary_size_is_group_order_plus_two():
    assert GroupStateTracking.get_iosize(group='S3') == 8
    assert GroupStateTracking.get_iosize(group='S4') == 26
    assert _dataset('S3').vocab_size == 8


def test_items_are_deterministic_per_seed():
    a, b, c = _dataset('S3', seed=1), _dataset('S3', seed=1), _dataset('S3', seed=2)
    assert torch.equal(a[0]['inputs'], b[0]['inputs'])
    assert not torch.equal(a[0]['inputs'], c[0]['inputs'])


class TestCosineScheduler:

    def _scheduler(self, epochs=10, lr=1e-3):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([param], lr=lr)
        return optimizer, CosineAnnealingLR('cosine', optimizer, epochs=epochs, eta_min=0.0)

    def test_anneals_to_zero_over_the_epochs(self):
        optimizer, scheduler = self._scheduler()
        lrs = [optimizer.param_groups[0]['lr']]
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        assert lrs[0] == pytest.approx(1e-3)
        assert lrs[5] == pytest.approx(5e-4)
        assert lrs[10] == pytest.approx(0.0, abs=1e-12)
        assert all(a >= b for a, b in zip(lrs, lrs[1:]))

    def test_steps_per_epoch_and_needs_epochs(self):
        assert CosineAnnealingLR.get_properties() == {'step': 'epoch'}
        assert CosineAnnealingLR.get_required_kwargs() == ['epochs']
