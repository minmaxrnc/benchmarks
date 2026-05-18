# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for LR-scheduler construction and step logic."""
import pytest
import torch.nn as nn
from benchmarks.optimizers.adam import Adam
from benchmarks.schedulers.nonescheduler import NoneScheduler
from benchmarks.schedulers.stepscheduler import StepLR
from benchmarks.schedulers.warmupstepscheduler import WarmupStepLR

pytestmark = pytest.mark.group('schedulers')


@pytest.fixture
def optimizer():
    model = nn.Linear(4, 2)
    return Adam('adam', params=model.parameters(), lr=0.1, weight_decay=0.0)


class TestNoneScheduler:

    def test_instantiation(self, optimizer):
        sched = NoneScheduler('none', optimizer=optimizer)
        assert sched.name == 'none'

    def test_lr_unchanged_after_construction(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        NoneScheduler('none', optimizer=optimizer)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr)

    def test_get_lr_returns_list(self, optimizer):
        sched = NoneScheduler('none', optimizer=optimizer)
        lrs = sched.get_lr()
        assert isinstance(lrs, list)
        assert len(lrs) == len(optimizer.param_groups)

    def test_step_property_is_never(self):
        assert NoneScheduler.get_properties()['step'] == 'never'


class TestStepLR:

    def test_instantiation(self, optimizer):
        sched = StepLR('step', optimizer=optimizer, step_size=5, gamma=0.5)
        assert sched.name == 'step'

    def test_lr_unchanged_before_step_size(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = StepLR('step', optimizer=optimizer, step_size=5, gamma=0.5)
        sched.step()        # epoch 1 — no decay yet
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr)

    def test_lr_decays_at_step_size(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = StepLR('step', optimizer=optimizer, step_size=2, gamma=0.5)
        sched.step()        # epoch 1
        sched.step()        # epoch 2 — decay kicks in
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr == pytest.approx(initial_lr * 0.5)

    def test_lr_decays_again_at_second_multiple(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = StepLR('step', optimizer=optimizer, step_size=2, gamma=0.5)
        for _ in range(4):
            sched.step()    # epochs 1-4; decay at epoch 2 and 4
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr == pytest.approx(initial_lr * 0.5 ** 2)

    def test_step_property_is_epoch(self):
        assert StepLR.get_properties()['step'] == 'epoch'


class TestWarmupStepLR:

    # epochs=20, warmup_fraction=0.2 → warmup_steps=4
    # At last_epoch=0 (init): scale = 1/4, LR = 0.025
    # At last_epoch=3 (after 3 steps): scale = 4/4 = 1.0, LR = 0.1
    # Then StepLR with step_size=2, gamma=0.5 counting from end of warmup.

    def test_instantiation(self, optimizer):
        sched = WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                             step_size=2, gamma=0.5, warmup_fraction=0.2)
        assert sched.name == 'step_warmup'

    def test_lr_starts_below_base(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                     step_size=2, gamma=0.5, warmup_fraction=0.2)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * 0.25)

    def test_lr_ramps_linearly_during_warmup(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                             step_size=2, gamma=0.5, warmup_fraction=0.2)
        sched.step()   # last_epoch=1 → scale=2/4
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * 0.5)
        sched.step()   # last_epoch=2 → scale=3/4
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * 0.75)

    def test_lr_reaches_full_at_end_of_warmup(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                             step_size=2, gamma=0.5, warmup_fraction=0.2)
        for _ in range(3):   # last_epoch reaches 3 = warmup_steps - 1
            sched.step()
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr)

    def test_no_decay_immediately_after_warmup(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                             step_size=2, gamma=0.5, warmup_fraction=0.2)
        for _ in range(4):   # last_epoch=4: post_warmup=0, scale=0.5^0=1
            sched.step()
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr)

    def test_decay_after_step_size_post_warmup(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                             step_size=2, gamma=0.5, warmup_fraction=0.2)
        for _ in range(6):   # last_epoch=6: post_warmup=2, scale=0.5^1
            sched.step()
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * 0.5)

    def test_second_decay(self, optimizer):
        initial_lr = optimizer.param_groups[0]['lr']
        sched = WarmupStepLR('step_warmup', optimizer=optimizer, epochs=20,
                             step_size=2, gamma=0.5, warmup_fraction=0.2)
        for _ in range(8):   # last_epoch=8: post_warmup=4, scale=0.5^2
            sched.step()
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * 0.25)

    def test_step_property_is_epoch(self):
        assert WarmupStepLR.get_properties()['step'] == 'epoch'

    def test_required_kwargs_includes_epochs(self):
        assert 'epochs' in WarmupStepLR.get_required_kwargs()
