# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for the Trainer: chunk-level gradient checkpointing
(_forward_checkpointed), gradient accumulation, the OOM-retry policy, gradient
norms, the train_log CSV, and end-to-end runs of Trainer.fit on CPU with
in-memory datasets (best-model selection, seed-skip rule, run_env, epoch
checkpoints, OOM retry). No CUDA or saved datasets needed.
"""
import copy
import csv
import glob
import json
import os

import torch
import torch.nn as nn
import pytest

from benchmarks.trainers import trainer as trainer_module
from benchmarks.trainers.trainer import Trainer
from benchmarks.models.model import Model
from benchmarks.losses.crossentropyloss import CrossEntropyLoss
from benchmarks.metrics.tokenaccuracy import TokenAccuracy
from benchmarks.utils import config
from minmax_rnc.model import MinMaxRNC_LM
from vanilla_rnn.model import VanillaRNN

pytestmark = pytest.mark.group('trainer')


class _ToyRecurrentModel(Model):
    """Minimal recurrent model (state at t depends on state at t-1), so that
    chunking exercises the state carried across chunk boundaries."""

    def __init__(self, vocab_size=5, d_model=4):
        super().__init__('toy_recurrent')
        self.emb = nn.Embedding(vocab_size, d_model)
        self.w   = nn.Linear(d_model, d_model)

    def forward(self, x, state=None, unroll_steps=-1, return_state=False):
        h = self.emb(x)  # (B, T, D)
        if state is None:
            state = torch.zeros(x.size(0), h.size(-1))
        outputs = []
        for t in range(h.size(1)):
            state = torch.tanh(self.w(h[:, t]) + state)
            outputs.append(state)
        y = torch.stack(outputs, dim=1)  # (B, T, D)
        if return_state:
            return y, state
        return y

    def supports_unroll_steps(self):
        return True


class _NonRecurrentModel(Model):
    """A model without unroll_steps support: called directly, no chunking."""

    def __init__(self):
        super().__init__('non_recurrent')
        self.calls = []

    def forward(self, x, state=None, unroll_steps=-1, return_state=False):
        self.calls.append(x)
        if return_state:
            return x, None
        return x


class _UnrollStepsCapturingModel(Model):
    """Records the unroll_steps value it was called with."""

    def __init__(self):
        super().__init__('capturing')
        self.seen_unroll_steps = []

    def forward(self, x, state=None, unroll_steps=-1, return_state=False):
        self.seen_unroll_steps.append(unroll_steps)
        if return_state:
            return x, state
        return x

    def supports_unroll_steps(self):
        return True


class _BatchSizeCapturingModel(Model):
    """Records the batch size (dim 0) it was called with."""

    def __init__(self):
        super().__init__('batch_capturing')
        self.seen_batch_sizes = []

    def forward(self, x, state=None, unroll_steps=-1, return_state=False):
        self.seen_batch_sizes.append(x.size(0))
        return x

    def supports_unroll_steps(self):
        return True


class _SumLoss:
    def __call__(self, logits, y, mask):
        return logits.float().sum()


class _CountMetric:
    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def update(self, logits, y, mask):
        self.n += logits.size(0)

    def compute(self):
        return self.n


def _tiny_minmax(vocab=6):
    return MinMaxRNC_LM('m', iosize=vocab, n_layers=2, d_model=16, d_state=4,
                        ffn_dropout=0.0, head_dropout=0.0)


# ---------------------------------------------------------------------------
# _forward_checkpointed
# ---------------------------------------------------------------------------

class TestForwardCheckpointed:

    def test_non_unroll_model_gets_a_single_plain_call(self):
        model = _NonRecurrentModel()
        x = torch.zeros(2, 3)

        out = Trainer._forward_checkpointed(None, model, x, unroll_steps=1)

        assert len(model.calls) == 1
        assert out is x

    def test_non_positive_unroll_steps_falls_back_to_one_call(self):
        model = _UnrollStepsCapturingModel()
        x = torch.zeros(2, 3)

        Trainer._forward_checkpointed(None, model, x, unroll_steps=0)

        assert model.seen_unroll_steps == [0]

    def test_sequence_not_longer_than_unroll_steps_is_one_plain_call(self):
        model = _UnrollStepsCapturingModel()
        x = torch.zeros(2, 10)

        Trainer._forward_checkpointed(None, model, x, unroll_steps=10)

        assert model.seen_unroll_steps == [10]

    def test_chunking_calls_model_once_per_chunk(self):
        model = _UnrollStepsCapturingModel()
        x = torch.zeros(2, 10)

        Trainer._forward_checkpointed(None, model, x, unroll_steps=3)

        # 10 tokens in chunks of 3 -> 4 calls (3, 3, 3, 1)
        assert model.seen_unroll_steps == [3, 3, 3, 1]

    def test_return_state_returns_the_final_state(self):
        torch.manual_seed(0)
        model = _ToyRecurrentModel()
        x = torch.randint(0, 5, (2, 10))

        _, state_ref = model(x, return_state=True)
        _, state = Trainer._forward_checkpointed(None, model, x, unroll_steps=3, return_state=True)

        assert torch.allclose(state_ref, state, atol=1e-6)

    def test_chunked_forward_matches_unchunked_forward(self):
        torch.manual_seed(0)
        model = _ToyRecurrentModel()
        x = torch.randint(0, 5, (2, 10))

        y_unchunked = model(x)
        y_chunked   = Trainer._forward_checkpointed(None, model, x, unroll_steps=3)

        assert torch.allclose(y_unchunked, y_chunked, atol=1e-6)

    def test_chunked_backward_matches_unchunked_backward(self):
        torch.manual_seed(0)
        model_unchunked = _ToyRecurrentModel()
        model_chunked   = copy.deepcopy(model_unchunked)

        x = torch.randint(0, 5, (2, 10))

        model_unchunked(x).sum().backward()
        grads_unchunked = [p.grad.clone() for p in model_unchunked.parameters()]

        Trainer._forward_checkpointed(None, model_chunked, x, unroll_steps=3).sum().backward()
        grads_chunked = [p.grad.clone() for p in model_chunked.parameters()]

        for g_unchunked, g_chunked in zip(grads_unchunked, grads_chunked):
            assert torch.allclose(g_unchunked, g_chunked, atol=1e-5)

    def test_minmax_chunked_gradients_match_unchunked(self):
        torch.manual_seed(0)
        model_unchunked = _tiny_minmax().train()
        model_chunked   = copy.deepcopy(model_unchunked)
        x = torch.randint(0, 6, (3, 20))
        y = torch.randint(0, 6, (3, 20))
        mask = torch.ones(3, 20, dtype=torch.bool)
        loss_fn = CrossEntropyLoss('CrossEntropyLoss')

        loss_fn(Trainer._forward_checkpointed(None, model_unchunked, x, 20), y, mask).backward()
        loss_fn(Trainer._forward_checkpointed(None, model_chunked, x, 7), y, mask).backward()

        for (name, p_u), p_c in zip(model_unchunked.named_parameters(), model_chunked.parameters()):
            if p_u.grad is None:
                assert p_c.grad is None, name
                continue
            assert torch.allclose(p_u.grad, p_c.grad, atol=1e-5, rtol=1e-4), name


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

class TestGradAccumMicroBatches:

    def _trainer(self):
        # _grad_accum_micro_batches does not read self.
        return object.__new__(Trainer)

    def test_default_of_one_returns_a_single_micro_batch(self):
        trainer = self._trainer()
        x, y, mask = torch.zeros(8, 3), torch.zeros(8, 3), torch.ones(8, 3, dtype=torch.bool)

        micro_batches = trainer._grad_accum_micro_batches(x, y, mask, grad_accum_steps=1)

        assert len(micro_batches) == 1
        assert micro_batches[0][0].size(0) == 8

    def test_evenly_divisible_batch_splits_into_equal_micro_batches(self):
        trainer = self._trainer()
        x, y, mask = torch.zeros(8, 3), torch.zeros(8, 3), torch.ones(8, 3, dtype=torch.bool)

        micro_batches = trainer._grad_accum_micro_batches(x, y, mask, grad_accum_steps=4)

        assert len(micro_batches) == 4
        assert all(x_mb.size(0) == 2 for x_mb, _, _ in micro_batches)

    def test_uneven_batch_uses_a_smaller_final_micro_batch(self):
        trainer = self._trainer()
        x, y, mask = torch.zeros(8, 3), torch.zeros(8, 3), torch.ones(8, 3, dtype=torch.bool)

        micro_batches = trainer._grad_accum_micro_batches(x, y, mask, grad_accum_steps=3)

        assert [x_mb.size(0) for x_mb, _, _ in micro_batches] == [3, 3, 2]

    def test_micro_batches_reconstruct_the_original_batch(self):
        trainer = self._trainer()
        x = torch.arange(8 * 3).reshape(8, 3)

        micro_batches = trainer._grad_accum_micro_batches(x, x.clone(), x.clone(), grad_accum_steps=3)

        assert torch.equal(torch.cat([x_mb for x_mb, _, _ in micro_batches], dim=0), x)


class TestRunTrainEpochGradAccum:
    """One optimizer step on a full batch equals one step with the batch split
    into equal micro-batches (mean loss, all tokens unmasked)."""

    def _trainer(self):
        trainer = object.__new__(Trainer)
        trainer.tracker_enabled = False
        trainer.grad_clip = None
        return trainer

    def _one_epoch(self, grad_accum_steps, unroll_steps):
        torch.manual_seed(0)
        model = _tiny_minmax()
        torch.manual_seed(1)
        x = torch.randint(0, 6, (4, 12))
        loader = [{'inputs': x, 'outputs': x.clone(), 'mask': torch.ones(4, 12, dtype=torch.bool)}]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = self._trainer()
        scheduler = trainer._init_scheduler('none', optimizer, 1, len(loader))
        loss, _, step, stats = trainer._run_train_epoch(
            model=model, loader=loader, metric=TokenAccuracy('TokenAccuracy'),
            loss_fn=CrossEntropyLoss('CrossEntropyLoss'), optimizer=optimizer,
            scheduler=scheduler, tracker=None, step=0, unroll_steps=unroll_steps,
            grad_accum_steps=grad_accum_steps,
        )
        return model, loss, step, stats

    def test_micro_batching_gives_the_same_update(self):
        ref, loss_ref, _, _ = self._one_epoch(grad_accum_steps=1, unroll_steps=12)
        acc, loss_acc, _, _ = self._one_epoch(grad_accum_steps=2, unroll_steps=12)
        assert loss_ref == pytest.approx(loss_acc, rel=1e-5)
        for p_ref, p_acc in zip(ref.parameters(), acc.parameters()):
            assert torch.allclose(p_ref, p_acc, atol=1e-6)

    def test_chunked_training_gives_the_same_update(self):
        ref, _, _, _ = self._one_epoch(grad_accum_steps=1, unroll_steps=12)
        chunked, _, _, _ = self._one_epoch(grad_accum_steps=1, unroll_steps=5)
        for p_ref, p_c in zip(ref.parameters(), chunked.parameters()):
            assert torch.allclose(p_ref, p_c, atol=1e-6)

    def test_epoch_stats(self):
        _, _, step, stats = self._one_epoch(grad_accum_steps=2, unroll_steps=5)
        assert step == 1
        assert stats['tokens'] == 4 * 12
        assert stats['throughput'] > 0
        assert stats['grad_norm_mean'] > 0
        assert stats['state_norm_mean'] > 0
        assert set(Trainer.STATS_COLUMNS) <= set(stats)


class TestRunValidateGradAccum:
    """_run_validate splits each batch into the same micro-batches as training,
    so that growing grad_accum_steps on OOM also bounds its memory."""

    def _trainer(self):
        return object.__new__(Trainer)

    def _loader(self, batch_size=8, seq_len=3):
        x = torch.zeros(batch_size, seq_len)
        return [{'inputs': x, 'outputs': x.clone(), 'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)}]

    def test_splits_batch_into_grad_accum_micro_batches(self):
        model = _BatchSizeCapturingModel()
        self._trainer()._run_validate(
            model=model, loader=self._loader(), metric=_CountMetric(), loss_fn=_SumLoss(),
            desc='test', unroll_steps=-1, grad_accum_steps=4,
        )
        assert model.seen_batch_sizes == [2, 2, 2, 2]

    def test_default_grad_accum_steps_is_a_single_micro_batch(self):
        model = _BatchSizeCapturingModel()
        self._trainer()._run_validate(
            model=model, loader=self._loader(), metric=_CountMetric(), loss_fn=_SumLoss(),
            desc='test', unroll_steps=-1,
        )
        assert model.seen_batch_sizes == [8]

    def test_metric_and_loss_still_see_every_sample(self):
        model = _BatchSizeCapturingModel()
        val_loss, val_acc = self._trainer()._run_validate(
            model=model, loader=self._loader(), metric=_CountMetric(), loss_fn=_SumLoss(),
            desc='test', unroll_steps=-1, grad_accum_steps=4,
        )
        assert val_acc == 8
        assert val_loss == 0.0


# ---------------------------------------------------------------------------
# OOM retry policy
# ---------------------------------------------------------------------------

class _FakeLenDataset:
    def __init__(self, max_len):
        self.max_len = max_len


class TestOomRetry:

    def test_prefers_growing_grad_accum_steps_when_micro_batch_still_shrinkable(self):
        assert Trainer._oom_retry(131072, 1, 64) == (131072, 2)

    def test_keeps_growing_grad_accum_steps_across_repeated_retries(self):
        unroll_steps, grad_accum_steps = 131072, 1
        for _ in range(4):
            unroll_steps, grad_accum_steps = Trainer._oom_retry(unroll_steps, grad_accum_steps, 64)
        assert (unroll_steps, grad_accum_steps) == (131072, 16)

    def test_falls_back_to_shrinking_unroll_steps_once_micro_batch_is_1(self):
        assert Trainer._oom_retry(131072, 64, 64) == (65536, 64)

    def test_raises_once_both_levers_are_exhausted(self):
        with pytest.raises(ValueError):
            Trainer._oom_retry(1, 64, 64)

    def test_uneven_division_still_detects_micro_batch_of_one(self):
        # ceil(64/63) = 2 > 1
        assert Trainer._oom_retry(131072, 63, 64) == (131072, 126)

    def test_recognises_oom_errors(self):
        assert Trainer._is_oom_error(RuntimeError('Triton Error [CUDA]: out of memory'))
        assert Trainer._is_oom_error(torch.cuda.OutOfMemoryError('CUDA out of memory'))
        assert not Trainer._is_oom_error(RuntimeError('shape mismatch'))


class TestInitialUnrollSteps:

    def _trainer_with(self, max_unroll_steps):
        trainer = object.__new__(Trainer)
        trainer.max_unroll_steps = max_unroll_steps
        return trainer

    def test_default_none_uses_the_longest_of_train_and_val(self):
        trainer = self._trainer_with(None)
        assert trainer._initial_unroll_steps(_FakeLenDataset(100), _FakeLenDataset(300)) == 300

    def test_explicit_value_overrides_dataset_lengths(self):
        trainer = self._trainer_with(4096)
        assert trainer._initial_unroll_steps(_FakeLenDataset(131072), _FakeLenDataset(131072)) == 4096


# ---------------------------------------------------------------------------
# Gradient norms and the CSV log
# ---------------------------------------------------------------------------

class TestMeasureAndClipGradNorm:

    def _model_with_grad(self, grad_values):
        model = nn.Linear(len(grad_values), 1, bias=False)
        model.weight.grad = torch.tensor([grad_values])
        return model

    def test_no_grad_clip_reports_norm_and_leaves_gradients_untouched(self):
        model = self._model_with_grad([3.0, 4.0, 0.0])  # norm 5
        assert Trainer._measure_and_clip_grad_norm(model, grad_clip=None) == pytest.approx(5.0)
        assert torch.allclose(model.weight.grad, torch.tensor([[3.0, 4.0, 0.0]]))

    def test_grad_clip_scales_gradients_but_still_reports_pre_clip_norm(self):
        model = self._model_with_grad([3.0, 4.0, 0.0])
        assert Trainer._measure_and_clip_grad_norm(model, grad_clip=2.5) == pytest.approx(5.0)
        assert torch.allclose(model.weight.grad, torch.tensor([[1.5, 2.0, 0.0]]))

    def test_grad_clip_larger_than_norm_does_not_scale(self):
        model = self._model_with_grad([3.0, 4.0, 0.0])
        assert Trainer._measure_and_clip_grad_norm(model, grad_clip=10.0) == pytest.approx(5.0)
        assert torch.allclose(model.weight.grad, torch.tensor([[3.0, 4.0, 0.0]]))


class TestGradNormEpochStats:

    def test_aggregates_mean_min_max(self):
        assert Trainer._grad_norm_epoch_stats([1.0, 3.0, 2.0]) == {'mean': 2.0, 'min': 1.0, 'max': 3.0}

    def test_empty_list_reports_nan_rather_than_crashing(self):
        assert all(v != v for v in Trainer._grad_norm_epoch_stats([]).values())


class TestCsvWriteHistory:
    """The original columns keep their positions; the new ones are appended."""

    def _trainer(self, tmp_path):
        trainer = object.__new__(Trainer)
        trainer.csv_log_path = str(tmp_path)
        return trainer

    def _read(self, trainer):
        path = trainer._path('train_log', '.csv', 0.001, 8, 'adam', 'none', 1)
        with open(path) as f:
            return list(csv.reader(f))

    def test_header_and_row(self, tmp_path):
        trainer = self._trainer(tmp_path)
        stats = {c: float(i) for i, c in enumerate(Trainer.STATS_COLUMNS)}
        history = [{
            'epoch': 1, 'lr': [0.001],
            'train_loss': 0.5, 'train_acc': 0.9, 'val_loss': 0.6, 'val_acc': 0.8,
            'runtime': 10.0, 'timelimit': '', 'earlystop': '', **stats,
        }]
        trainer._csv_write_history(history, lr=0.001, batch_size=8, optimizer_name='adam',
                                   scheduler_name='none', seed=1, n_lr=1)
        rows = self._read(trainer)
        assert rows[0] == [
            'epoch', 'lr_0', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'runtime', 'timelimit', 'earlystop',
        ] + Trainer.STATS_COLUMNS
        assert rows[1][:9] == ['1', '0.001', '0.5', '0.9', '0.6', '0.8', '10.0', '', '']
        assert rows[1][9:] == [str(float(i)) for i in range(len(Trainer.STATS_COLUMNS))]

    def test_missing_stats_are_left_empty(self, tmp_path):
        trainer = self._trainer(tmp_path)
        history = [{
            'epoch': 1, 'lr': [0.001],
            'train_loss': 0.5, 'train_acc': 0.9, 'val_loss': 0.6, 'val_acc': 0.8,
            'runtime': 10.0, 'timelimit': '', 'earlystop': '',
        }]
        trainer._csv_write_history(history, lr=0.001, batch_size=8, optimizer_name='adam',
                                   scheduler_name='none', seed=1, n_lr=1)
        assert self._read(trainer)[1][9:] == [''] * len(Trainer.STATS_COLUMNS)


class TestCopyStateDict:

    def test_copy_does_not_alias_live_parameters(self):
        model = nn.Linear(3, 2)
        copied = Trainer._copy_state_dict(model)
        aliased = model.state_dict()
        with torch.no_grad():
            model.weight.add_(1.0)
        assert torch.equal(aliased['weight'], model.weight)
        assert not torch.equal(copied['weight'], model.weight)


# ---------------------------------------------------------------------------
# End-to-end Trainer.fit on CPU with in-memory datasets
# ---------------------------------------------------------------------------

VOCAB = 6
SEQ   = 12


class _InMemoryDataset:
    def __init__(self, seed, n):
        g = torch.Generator().manual_seed(seed * 1000 + n)
        self.x = torch.randint(0, VOCAB, (n, SEQ), generator=g)
        self.max_len = SEQ

    def __len__(self):
        return self.x.size(0)

    def get_loader(self, batch_size):
        batches = []
        for i in range(0, len(self) - batch_size + 1, batch_size):
            x = self.x[i:i + batch_size]
            batches.append({'inputs': x, 'outputs': x.clone(),
                            'mask': torch.ones_like(x, dtype=torch.bool)})
        return batches


class _FakeDatasets:
    def instantiate(self, name, seed):
        return _InMemoryDataset(seed, 16 if name == 'train' else 8)

    def repr(self, dataset, seed):
        return f'in-memory(seed={seed})'

    def get_iosize(self, name):
        return VOCAB


class _OomOnLargeBatches(MinMaxRNC_LM):
    """Raises a CUDA-style OOM error on micro-batches larger than 2."""

    def forward(self, x, *args, **kwargs):
        if x.size(0) > 2 and self.training:
            raise RuntimeError('CUDA out of memory (simulated)')
        return super().forward(x, *args, **kwargs)


class _FakeModels:
    def __init__(self, cls=MinMaxRNC_LM):
        self.cls = cls

    def instantiate(self, name, iosize, **kwargs):
        if self.cls is VanillaRNN:
            return VanillaRNN(name, iosize=iosize, d_model=16)
        return self.cls(name, iosize=iosize, n_layers=1, d_model=16, d_state=4)

    def get_required_kwargs(self, name):
        return []

    def get_meta(self, name):
        return {'class': self.cls.__name__, 'args': {'n_layers': 1, 'd_model': 16, 'd_state': 4}}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(trainer_module, 'datasets', _FakeDatasets())
    monkeypatch.setattr(trainer_module, 'models', _FakeModels())
    return monkeypatch


def _make_trainer(tmp_path, **kwargs):
    args = dict(
        model='m', loss='CrossEntropyLoss', metric='TokenAccuracy',
        train_dataset='train', val_dataset='val', epochs=3,
        csv_log_path=str(tmp_path), best_model_path=str(tmp_path / 'best_model.pt'),
        optimizer='adam', scheduler='none', lr=0.05, train_batch_size=4,
    )
    args.update(kwargs)
    return Trainer('trainer_test', **args)


ONE_SEED = {'ci_min_n': 1, 'ci_max_n': 1, 'stopper__enabled': False}


def _scripted_validation(monkeypatch, val_scores):
    """Replace the validation scores (desc 'Validate') with val_scores, one
    (loss, acc) per epoch, keeping the real training."""
    real = Trainer._run_validate
    scores = iter(val_scores)

    def fake(self, model, loader, metric, loss_fn, desc, unroll_steps, grad_accum_steps=1):
        result = real(self, model, loader, metric, loss_fn, desc, unroll_steps, grad_accum_steps)
        if desc == 'Validate':
            return next(scores)
        return result
    monkeypatch.setattr(Trainer, '_run_validate', fake)


def _load(path):
    return torch.load(path, map_location='cpu')


def _assert_same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k].cpu(), b[k].cpu()), k


class TestFitEndToEnd:

    def test_fit_writes_outputs(self, tmp_path, fake_registry):
        with config.overrides(ONE_SEED):
            _make_trainer(tmp_path, epochs=2).fit()
        assert (tmp_path / 'best_model.pt').exists()
        assert (tmp_path / 'train_log__summary.json').exists()
        with open(glob.glob(str(tmp_path / 'train_log__*__seed_1.csv'))[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert float(rows[0]['tokens']) == 16 * SEQ
        assert float(rows[0]['throughput']) > 0

    @pytest.mark.parametrize('best_model, expected_epoch', [('best_epoch', 1), ('last_epoch', 3)])
    def test_best_model_option(self, tmp_path, fake_registry, best_model, expected_epoch):
        # Epoch 1 has the best validation scores; training continues to epoch 3.
        _scripted_validation(fake_registry, [(0.1, 0.9), (0.5, 0.5), (0.6, 0.4)])
        with config.overrides(ONE_SEED):
            _make_trainer(tmp_path, best_model=best_model, save_checkpoint_epochs=[1, 3]).fit()

        saved = _load(tmp_path / 'best_model.pt')
        checkpoints = {
            e: _load(glob.glob(str(tmp_path / f'checkpoint__*__seed_1__epoch_{e}.pt'))[0])
            for e in (1, 3)
        }
        _assert_same_weights(saved, checkpoints[expected_epoch])
        other = 3 if expected_epoch == 1 else 1
        assert any(not torch.equal(saved[k].cpu(), checkpoints[other][k]) for k in saved)

        with open(tmp_path / 'train_log__summary.json') as f:
            summary = json.load(f)
        assert summary['best_model'] == best_model
        assert summary['values_for_best_model']['val_acc'] == 0.9

    def test_best_model_defaults_to_config(self, tmp_path, fake_registry):
        assert _make_trainer(tmp_path).best_model == 'best_epoch'
        with config.overrides({'best_model': 'last_epoch'}):
            assert _make_trainer(tmp_path).best_model == 'last_epoch'
            assert _make_trainer(tmp_path, best_model='best_epoch').best_model == 'best_epoch'

    def test_invalid_best_model_raises(self, tmp_path, fake_registry):
        with pytest.raises(ValueError):
            _make_trainer(tmp_path, best_model='last')

    def test_saved_checkpoint_loads_strictly_into_a_fresh_model(self, tmp_path, fake_registry):
        with config.overrides(ONE_SEED):
            _make_trainer(tmp_path, epochs=1, max_unroll_steps=5).fit()
        fresh = _FakeModels().instantiate('m', VOCAB)
        fresh.load_state_dict(_load(tmp_path / 'best_model.pt'), strict=True)

    def test_run_env_is_written_per_seed(self, tmp_path, fake_registry):
        with config.overrides(ONE_SEED):
            _make_trainer(tmp_path, epochs=1, max_unroll_steps=5, grad_accum_steps=2).fit()
        (path,) = glob.glob(str(tmp_path / 'run_env__*__seed_1.json'))
        with open(path) as f:
            env = json.load(f)
        assert env['seed'] == 1
        assert env['compile_model'] is False
        assert env['compile_source'] == 'meta.config.yaml'
        assert env['unroll_steps'] == 5
        assert env['grad_accum_steps'] == 2
        assert env['best_model'] == 'best_epoch'
        assert env['minmaxrnc'] is not None
        assert env['torch'] == torch.__version__
        assert env['model_args'] == {'n_layers': 1, 'd_model': 16, 'd_state': 4}

    def test_compile_model_trainer_arg_overrides_config(self, tmp_path, fake_registry):
        assert _make_trainer(tmp_path)._compile_enabled() is False
        assert _make_trainer(tmp_path, compile_model=True)._compile_enabled() is True
        with config.overrides({'compile_model': True}):
            assert _make_trainer(tmp_path)._compile_enabled() is True
            assert _make_trainer(tmp_path, compile_model=False)._compile_enabled() is False

    def test_fit_with_a_model_without_unroll_support(self, tmp_path, monkeypatch):
        monkeypatch.setattr(trainer_module, 'datasets', _FakeDatasets())
        monkeypatch.setattr(trainer_module, 'models', _FakeModels(VanillaRNN))
        with config.overrides(ONE_SEED):
            _make_trainer(tmp_path, epochs=1, grad_accum_steps=2).fit()
        fresh = VanillaRNN('m', iosize=VOCAB, d_model=16)
        fresh.load_state_dict(_load(tmp_path / 'best_model.pt'), strict=True)

    def test_oom_retry_grows_grad_accum_steps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(trainer_module, 'datasets', _FakeDatasets())
        monkeypatch.setattr(trainer_module, 'models', _FakeModels(_OomOnLargeBatches))
        with config.overrides(ONE_SEED):
            _make_trainer(tmp_path, epochs=1).fit()
        (path,) = glob.glob(str(tmp_path / 'run_env__*__seed_1.json'))
        with open(path) as f:
            env = json.load(f)
        assert env['grad_accum_steps'] == 2      # train_batch_size 4 -> micro-batches of 2
        assert env['unroll_steps'] == SEQ        # untouched


class TestSeedSkipRule:

    def _trainer(self, threshold):
        trainer = object.__new__(Trainer)
        trainer.val_acc_giveup_threshold = threshold
        return trainer

    def test_disabled_when_none(self):
        assert self._trainer(None)._give_up(0.0) is False

    def test_none_ci_bound_never_gives_up(self):
        assert self._trainer(0.5)._give_up(None) is False

    def test_gives_up_below_threshold(self):
        assert self._trainer(0.5)._give_up(0.4) is True
        assert self._trainer(0.5)._give_up(0.5) is False

    @pytest.mark.parametrize('threshold, expected_seeds', [(0.5, 2), (None, 4)])
    def test_fit_seed_count(self, tmp_path, fake_registry, threshold, expected_seeds):
        # One epoch per seed. Each seed is a bit more accurate (and has a
        # higher loss) than the previous one, so the val_loss CI rule never
        # stops early; all accuracies are far below 0.5, so the seed-skip rule
        # stops at ci_min_n = 2 seeds instead of ci_max_n = 4.
        _scripted_validation(fake_registry, [(1.0 + i, 0.1 + 0.01 * i) for i in range(10)])
        protocol = {'ci_min_n': 2, 'ci_max_n': 4, 'stopper__enabled': False,
                    'val_acc_giveup_threshold': threshold}
        with config.overrides(protocol):
            _make_trainer(tmp_path, epochs=1).fit()
        assert len(glob.glob(str(tmp_path / 'train_log__*__seed_*.csv'))) == expected_seeds
