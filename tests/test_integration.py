# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Synthetic end-to-end integration tests — no saved datasets required."""
import pytest
import torch
from vanilla_rnn.model import VanillaRNN
from benchmarks.losses.crossentropyloss import CrossEntropyLoss
from benchmarks.metrics.tokenaccuracy import TokenAccuracy
from benchmarks.optimizers.adam import Adam

pytestmark = pytest.mark.group('integration')

VOCAB  = 15
BATCH  = 4
SEQ    = 12


@pytest.fixture
def components():
    torch.manual_seed(0)
    model   = VanillaRNN('test', iosize=VOCAB, d_model=32, n_layers=1)
    loss_fn = CrossEntropyLoss('CrossEntropyLoss')
    metric  = TokenAccuracy('TokenAccuracy')
    opt     = Adam('adam', params=model.parameters(), lr=1e-2, weight_decay=0.0)
    return model, loss_fn, metric, opt


class TestSingleTrainingStep:

    def test_forward_backward_runs(self, components):
        model, loss_fn, _, opt = components
        x    = torch.randint(0, VOCAB, (BATCH, SEQ))
        y    = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)

        model.train(True)
        torch.set_grad_enabled(True)
        opt.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y, mask)
        loss.backward()
        opt.step()

        assert loss.item() > 0

    def test_gradients_computed_after_backward(self, components):
        model, loss_fn, _, opt = components
        x    = torch.randint(0, VOCAB, (BATCH, SEQ))
        y    = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)

        model.train(True)
        torch.set_grad_enabled(True)
        opt.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits, y, mask)
        loss.backward()

        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_params_change_after_step(self, components):
        model, loss_fn, _, opt = components
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        x    = torch.randint(0, VOCAB, (BATCH, SEQ))
        y    = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)

        model.train(True)
        torch.set_grad_enabled(True)
        opt.zero_grad()
        model(x).sum().backward()   # simplified loss
        opt.step()

        changed = any(
            not torch.allclose(p, before[n])
            for n, p in model.named_parameters()
        )
        assert changed


class TestLossDecreasesOnOverfit:

    def test_loss_decreases_over_50_steps(self):
        torch.manual_seed(42)
        model   = VanillaRNN('test', iosize=VOCAB, d_model=64, n_layers=1)
        loss_fn = CrossEntropyLoss('CrossEntropyLoss')
        opt     = Adam('adam', params=model.parameters(), lr=1e-2, weight_decay=0.0)

        x    = torch.randint(0, VOCAB, (BATCH, SEQ))
        y    = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)

        model.train(True)
        torch.set_grad_enabled(True)

        first_loss = None
        for _ in range(50):
            opt.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y, mask)
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            opt.step()

        assert loss.item() < first_loss


class TestEvalLoop:

    def test_metric_in_range_after_eval_pass(self, components):
        model, loss_fn, metric, _ = components
        metric.reset()

        x    = torch.randint(0, VOCAB, (BATCH, SEQ))
        y    = torch.randint(0, VOCAB, (BATCH, SEQ))
        mask = torch.ones(BATCH, SEQ, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            logits = model.forward(x)
            loss   = loss_fn(logits, y, mask)
            metric.update(logits, y, mask)

        acc = metric.compute()
        assert 0.0 <= acc <= 1.0
        assert loss.item() > 0
