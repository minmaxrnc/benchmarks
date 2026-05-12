# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for optimizer construction and parameter groups."""
import pytest
import torch
import torch.nn as nn
from benchmarks.optimizers.adam import Adam
from benchmarks.optimizers.adamw import AdamW

pytestmark = pytest.mark.group('optimizers')


@pytest.fixture
def model():
    return nn.Linear(8, 4)


@pytest.fixture
def decay_groups(model):
    decay    = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return decay, no_decay


class TestAdam:

    def test_instantiation(self, model):
        opt = Adam('adam', params=model.parameters(), lr=1e-3, weight_decay=1e-4)
        assert opt.name == 'adam'

    def test_has_param_groups(self, model):
        opt = Adam('adam', params=model.parameters(), lr=1e-3, weight_decay=1e-4)
        assert len(opt.param_groups) == 1

    def test_lr_stored(self, model):
        opt = Adam('adam', params=model.parameters(), lr=1e-3, weight_decay=1e-4)
        assert opt.param_groups[0]['lr'] == pytest.approx(1e-3)

    def test_step_does_not_raise(self, model):
        opt = Adam('adam', params=model.parameters(), lr=1e-3, weight_decay=0.0)
        x    = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_requires_decay_groups_property_is_false(self):
        assert Adam.get_properties()['requires_decay_groups'] is False


class TestAdamW:

    def test_instantiation(self, decay_groups):
        decay, no_decay = decay_groups
        opt = AdamW('adamW', decay=decay, no_decay=no_decay, lr=1e-3, weight_decay=0.1)
        assert opt.name == 'adamW'

    def test_two_param_groups(self, decay_groups):
        decay, no_decay = decay_groups
        opt = AdamW('adamW', decay=decay, no_decay=no_decay, lr=1e-3, weight_decay=0.1)
        assert len(opt.param_groups) == 2

    def test_decay_group_has_weight_decay(self, decay_groups):
        decay, no_decay = decay_groups
        opt = AdamW('adamW', decay=decay, no_decay=no_decay, lr=1e-3, weight_decay=0.1)
        assert opt.param_groups[0]['weight_decay'] == pytest.approx(0.1)

    def test_no_decay_group_has_zero_weight_decay(self, decay_groups):
        decay, no_decay = decay_groups
        opt = AdamW('adamW', decay=decay, no_decay=no_decay, lr=1e-3, weight_decay=0.1)
        assert opt.param_groups[1]['weight_decay'] == pytest.approx(0.0)

    def test_requires_decay_groups_property_is_true(self):
        assert AdamW.get_properties()['requires_decay_groups'] is True
