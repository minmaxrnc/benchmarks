"""Tests for accuracy metrics."""
import pytest
import torch
from benchmarks.metrics.tokenaccuracy import TokenAccuracy

pytestmark = pytest.mark.group('metrics')

B, T, C = 4, 8, 10


@pytest.fixture
def metric():
    m = TokenAccuracy('TokenAccuracy')
    m.reset()
    return m


def _perfect_logits(targets):
    """Return logits that confidently predict every target class."""
    B, T = targets.shape
    logits = torch.full((B, T, C), -1e6)
    logits.scatter_(-1, targets.unsqueeze(-1), 1e6)
    return logits


class TestTokenAccuracyBasic:

    def test_perfect_accuracy(self, metric):
        targets = torch.randint(0, C, (B, T))
        mask    = torch.ones(B, T, dtype=torch.bool)
        metric.update(_perfect_logits(targets), targets, mask)
        assert metric.compute() == pytest.approx(1.0)

    def test_zero_accuracy(self, metric):
        targets = torch.zeros(B, T, dtype=torch.long)
        logits  = torch.full((B, T, C), -1e6)
        logits[:, :, 1] = 1e6          # always predicts class 1, targets are 0
        mask = torch.ones(B, T, dtype=torch.bool)
        metric.update(logits, targets, mask)
        assert metric.compute() == pytest.approx(0.0)

    def test_result_is_in_unit_interval(self, metric):
        logits  = torch.randn(B, T, C)
        targets = torch.randint(0, C, (B, T))
        mask    = torch.ones(B, T, dtype=torch.bool)
        metric.update(logits, targets, mask)
        acc = metric.compute()
        assert 0.0 <= acc <= 1.0

    def test_empty_mask_returns_zero(self, metric):
        logits  = torch.randn(B, T, C)
        targets = torch.randint(0, C, (B, T))
        mask    = torch.zeros(B, T, dtype=torch.bool)
        metric.update(logits, targets, mask)
        assert metric.compute() == 0.0


class TestTokenAccuracyMask:

    def test_only_masked_positions_counted(self, metric):
        targets = torch.zeros(B, T, dtype=torch.long)
        logits  = _perfect_logits(targets)
        # Mask only the first column; model predicts it correctly
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 0] = True
        metric.update(logits, targets, mask)
        assert metric.compute() == pytest.approx(1.0)

    def test_correct_outside_mask_not_counted(self, metric):
        # All positions correct, but mask covers only wrong-prediction positions
        targets = torch.zeros(B, T, dtype=torch.long)
        logits  = torch.full((B, T, C), -1e6)
        logits[:, :, 0] = 1e6        # correct for all
        logits[:, 0, :] = -1e6       # first col: wrong
        logits[:, 0, 1] = 1e6
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 0] = True            # only wrong column is masked in
        metric.update(logits, targets, mask)
        assert metric.compute() == pytest.approx(0.0)


class TestTokenAccuracyAccumulation:

    def test_accumulates_over_multiple_batches(self, metric):
        targets = torch.zeros(B, T, dtype=torch.long)
        logits  = _perfect_logits(targets)
        mask    = torch.ones(B, T, dtype=torch.bool)
        metric.update(logits, targets, mask)
        metric.update(logits, targets, mask)
        assert metric.compute() == pytest.approx(1.0)

    def test_reset_clears_state(self, metric):
        targets = torch.randint(0, C, (B, T))
        logits  = torch.randn(B, T, C)
        mask    = torch.ones(B, T, dtype=torch.bool)
        metric.update(logits, targets, mask)
        metric.reset()
        assert metric.compute() == 0.0
        assert metric.correct.item() == 0
        assert metric.total.item() == 0
