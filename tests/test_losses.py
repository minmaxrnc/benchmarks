# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for loss functions."""
import pytest
import torch
from benchmarks.losses.crossentropyloss import CrossEntropyLoss

pytestmark = pytest.mark.group('losses')

B, T, C = 4, 8, 10


@pytest.fixture
def loss():
    return CrossEntropyLoss('CrossEntropyLoss')


@pytest.fixture
def full_mask():
    return torch.ones(B, T, dtype=torch.bool)


class TestCrossEntropyLossOutput:

    def test_returns_scalar(self, loss, full_mask):
        logits  = torch.randn(B, T, C)
        targets = torch.randint(0, C, (B, T))
        result  = loss(logits, targets, full_mask)
        assert result.shape == torch.Size([])

    def test_loss_is_positive(self, loss, full_mask):
        logits  = torch.randn(B, T, C)
        targets = torch.randint(0, C, (B, T))
        assert loss(logits, targets, full_mask).item() > 0

    def test_near_zero_on_confident_correct_predictions(self, loss, full_mask):
        targets = torch.zeros(B, T, dtype=torch.long)
        logits  = torch.full((B, T, C), -1e6)
        logits[:, :, 0] = 1e6
        assert loss(logits, targets, full_mask).item() < 0.01

    def test_high_loss_on_confident_wrong_predictions(self, loss, full_mask):
        targets = torch.zeros(B, T, dtype=torch.long)   # correct class is 0
        logits  = torch.full((B, T, C), -1e6)
        logits[:, :, 1] = 1e6                            # model predicts class 1
        result = loss(logits, targets, full_mask)
        assert result.item() > 5.0


class TestCrossEntropyLossMasking:

    def test_all_masked_out_does_not_raise(self, loss):
        logits  = torch.randn(B, T, C)
        targets = torch.randint(0, C, (B, T))
        mask    = torch.zeros(B, T, dtype=torch.bool)
        # With all positions masked the denominator is 0; should not raise
        result = loss(logits, targets, mask)
        assert result.shape == torch.Size([])

    def test_masking_changes_loss_value(self, loss):
        targets = torch.zeros(B, T, dtype=torch.long)
        logits  = torch.randn(B, T, C)
        full    = torch.ones(B, T, dtype=torch.bool)
        half    = full.clone()
        half[:, T // 2:] = False
        # Losses differ because different tokens are counted
        loss_full = loss(logits, targets, full).item()
        loss_half = loss(logits, targets, half).item()
        assert loss_full != pytest.approx(loss_half)


class TestCrossEntropyLossClone:

    def test_clone_default_keeps_reduction(self, loss):
        cloned = loss.clone()
        assert cloned.reduction == loss.reduction

    def test_clone_with_new_reduction(self, loss):
        cloned = loss.clone(reduction='none')
        assert cloned.reduction == 'none'
        assert loss.reduction != 'none'

    def test_clone_is_independent_instance(self, loss):
        cloned = loss.clone()
        assert cloned is not loss
