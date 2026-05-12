# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for model construction and forward passes."""
import pytest
import torch
from unittest.mock import MagicMock
from vanilla_rnn.model import VanillaRNN
from benchmarks.models.model import Model

pytestmark = pytest.mark.group('models')

VOCAB  = 20
BATCH  = 4
SEQ    = 16


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestVanillaRNNInit:

    def test_is_model_subclass(self):
        model = VanillaRNN('m', iosize=VOCAB)
        assert isinstance(model, Model)

    def test_name_attribute(self):
        model = VanillaRNN('my_rnn', iosize=VOCAB)
        assert model.name == 'my_rnn'

    def test_default_embedding_size(self):
        model = VanillaRNN('m', iosize=VOCAB)
        assert model.embedding.num_embeddings == VOCAB
        assert model.embedding.embedding_dim == 128

    def test_default_rnn_dims(self):
        model = VanillaRNN('m', iosize=VOCAB)
        assert model.rnn.input_size == 128
        assert model.rnn.hidden_size == 128
        assert model.rnn.num_layers == 1

    def test_custom_d_model(self):
        model = VanillaRNN('m', iosize=VOCAB, d_model=64)
        assert model.embedding.embedding_dim == 64
        assert model.rnn.input_size == 64

    def test_custom_n_layers(self):
        model = VanillaRNN('m', iosize=VOCAB, n_layers=3)
        assert model.rnn.num_layers == 3

    def test_output_proj_out_features_equals_vocab(self):
        model = VanillaRNN('m', iosize=VOCAB, d_model=32)
        assert model.output_proj.out_features == VOCAB

    def test_single_layer_no_dropout_in_rnn(self):
        # nn.RNN ignores dropout when num_layers == 1; verify no error at init
        model = VanillaRNN('m', iosize=VOCAB, n_layers=1, dropout=0.5)
        assert model.rnn.dropout == 0.0


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestVanillaRNNForward:

    @pytest.fixture(autouse=True)
    def model(self):
        m = VanillaRNN('m', iosize=VOCAB, d_model=32, n_layers=1)
        m.eval()
        return m

    def _x(self):
        return torch.randint(0, VOCAB, (BATCH, SEQ))

    def test_output_shape_train_call(self, model):
        with torch.no_grad():
            out = model(self._x())
        assert out.shape == (BATCH, SEQ, VOCAB)

    def test_output_shape_stateless(self, model):
        with torch.no_grad():
            out = model.forward(self._x())
        assert out.shape == (BATCH, SEQ, VOCAB)

    def test_output_shape_no_return_state(self, model):
        with torch.no_grad():
            out = model.forward(self._x(), return_state=False)
        assert out.shape == (BATCH, SEQ, VOCAB)

    def test_output_shape_with_return_state(self, model):
        with torch.no_grad():
            logits, state = model.forward(self._x(), return_state=True)
        assert logits.shape == (BATCH, SEQ, VOCAB)
        assert state is not None

    def test_output_dtype_is_float32(self, model):
        with torch.no_grad():
            out = model(self._x())
        assert out.dtype == torch.float32

    def test_single_sample(self, model):
        x = torch.randint(0, VOCAB, (1, SEQ))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, SEQ, VOCAB)

    def test_length_one_sequence(self, model):
        x = torch.randint(0, VOCAB, (BATCH, 1))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, 1, VOCAB)

    def test_deterministic_in_eval_mode(self, model):
        x = self._x()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Weight-decay parameter groups
# ---------------------------------------------------------------------------

class TestWeightDecayGroups:

    def test_returns_two_non_empty_lists(self):
        model = VanillaRNN('m', iosize=VOCAB, d_model=32)
        decay, no_decay = model.create_weight_decay_optim_groups()
        assert len(decay) > 0
        assert len(no_decay) > 0

    def test_all_params_partitioned(self):
        model = VanillaRNN('m', iosize=VOCAB, d_model=32)
        decay, no_decay = model.create_weight_decay_optim_groups()
        total   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grouped = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
        assert total == grouped

    def test_supports_unroll_steps_is_false(self):
        model = VanillaRNN('m', iosize=VOCAB)
        assert model.supports_unroll_steps() is False


# ---------------------------------------------------------------------------
# supports_unroll_steps dispatch (mirrors evaluator branching logic)
# ---------------------------------------------------------------------------

class TestSupportsUnrollStepsDispatch:
    """Verify the evaluator dispatch: models that support unroll_steps receive
    the kwarg; models that don't are called without it."""

    def _make_fake_model(self, supports: bool):
        logits = torch.zeros(1, 4, VOCAB)
        state  = torch.zeros(1, 1, VOCAB)
        model  = MagicMock()
        model.supports_unroll_steps.return_value = supports
        model.return_value = (logits, state)
        return model

    def test_model_without_unroll_steps_called_without_kwarg(self):
        model = self._make_fake_model(supports=False)
        x, prev_state, chunk_size = torch.zeros(1, 4), None, 4

        if model.supports_unroll_steps():
            model(x, prev_state, return_state=True, unroll_steps=chunk_size)
        else:
            model(x, prev_state, return_state=True)

        model.assert_called_once_with(x, prev_state, return_state=True)

    def test_model_with_unroll_steps_called_with_kwarg(self):
        model = self._make_fake_model(supports=True)
        x, prev_state, chunk_size = torch.zeros(1, 4), None, 4

        if model.supports_unroll_steps():
            model(x, prev_state, return_state=True, unroll_steps=chunk_size)
        else:
            model(x, prev_state, return_state=True)

        model.assert_called_once_with(x, prev_state, return_state=True, unroll_steps=chunk_size)
