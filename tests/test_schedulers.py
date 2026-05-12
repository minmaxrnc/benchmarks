"""Tests for LR-scheduler construction and step logic."""
import pytest
import torch.nn as nn
from benchmarks.optimizers.adam import Adam
from benchmarks.schedulers.nonescheduler import NoneScheduler
from benchmarks.schedulers.stepscheduler import StepLR

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
