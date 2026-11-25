# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for the MinMax RNC wrapper (models/minmax_rnc) against minmaxrnc >= 0.4.1."""
import warnings

import pytest
import torch

from minmax_rnc.model import MinMaxRNC_LM
from benchmarks.models.models import models
from benchmarks.utils.meta import Meta

pytestmark = pytest.mark.group('models')

VOCAB = 20
BATCH = 3
SEQ   = 24


def _n_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def _model(**kwargs):
    args = dict(iosize=VOCAB, n_layers=2, d_model=90, d_state=40)
    args.update(kwargs)
    return MinMaxRNC_LM('m', **args)


def _x(vocab=VOCAB, batch=BATCH, seq=SEQ):
    return torch.randint(0, vocab, (batch, seq))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBuild:

    @pytest.mark.parametrize('output_gate', [True, False])
    @pytest.mark.parametrize('conv_type', ['basic', 'gated'])
    def test_builds_without_warnings(self, output_gate, conv_type):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            _model(output_gate=output_gate, conv_type=conv_type, s_r_init='asymmetric')

    def test_builds_from_meta_without_warnings(self):
        name = 'minmaxrnc__d_(90)__l_(2)__s_(40)__og_false__cv_gated'
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            model = models.instantiate(name, VOCAB)
        assert isinstance(model, MinMaxRNC_LM)
        assert _n_params(model) == 89_812

    def test_d_state_and_scalar_neurons_together_raise(self):
        with pytest.raises(ValueError):
            _model(d_state=40, scalar_neurons=40)

    def test_supports_unroll_steps(self):
        assert _model().supports_unroll_steps() is True


# ---------------------------------------------------------------------------
# Parameter counts (V=20, L=2, d_model 90, 40 scalar-state neurons). The
# totals include the 2 x 130 non-trainable initial-state entries.
# ---------------------------------------------------------------------------

class TestParameterCounts:

    @pytest.mark.parametrize('output_gate, conv_type, total, trainable', [
        (True,  'basic', 129_492, 129_232),
        (True,  'gated',  97_092,  96_832),
        (False, 'basic', 122_212, 121_952),
        (False, 'gated',  89_812,  89_552),
    ])
    def test_counts(self, output_gate, conv_type, total, trainable):
        model = _model(output_gate=output_gate, conv_type=conv_type)
        assert _n_params(model) == total
        assert _n_params(model, trainable_only=True) == trainable

    def test_s_r_init_does_not_change_count(self):
        assert _n_params(_model(s_r_init='asymmetric')) == _n_params(_model())

    def test_d_state_is_an_alias_of_scalar_neurons(self):
        torch.manual_seed(0)
        a = MinMaxRNC_LM('m', iosize=VOCAB, n_layers=2, d_model=90, d_state=40)
        torch.manual_seed(0)
        b = MinMaxRNC_LM('m', iosize=VOCAB, n_layers=2, d_model=90, scalar_neurons=40)
        sd_a, sd_b = a.state_dict(), b.state_dict()
        assert sd_a.keys() == sd_b.keys()
        assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


class TestVariantsDiffer:
    """output_gate and conv_type must reach the model (they were silently
    overridden before v0.1.3)."""

    def test_all_four_variants_have_distinct_state_dicts(self):
        signatures = set()
        for og in (True, False):
            for cv in ('basic', 'gated'):
                sd = _model(output_gate=og, conv_type=cv).state_dict()
                signatures.add(tuple(sorted((k, tuple(v.shape)) for k, v in sd.items())))
        assert len(signatures) == 4

    def test_output_gate_changes_outputs(self):
        x = _x()
        torch.manual_seed(0)
        a = _model(output_gate=True).eval()
        torch.manual_seed(0)
        b = _model(output_gate=False).eval()
        with torch.no_grad():
            assert not torch.allclose(a(x, unroll_steps=SEQ), b(x, unroll_steps=SEQ))


# ---------------------------------------------------------------------------
# Neuron arrays (0.4 API)
# ---------------------------------------------------------------------------

class TestNeuronArrays:

    def test_word_s3_model_parameter_count(self):
        # Word(S_3): one layer, 16 neurons of state dimension 3, no scalar-state
        # neurons, d_model 90, V = |S_3| + 2 = 8. The paper reports ~76k.
        model = MinMaxRNC_LM('m', iosize=8, n_layers=1, d_model=90, scalar_neurons=0,
                             neurons=[[3, 16]], s_r_init='asymmetric')
        assert _n_params(model, trainable_only=True) == 76_166
        assert _n_params(model) == 76_304

    def test_neurons_accept_lists_and_tuples(self):
        torch.manual_seed(0)
        a = MinMaxRNC_LM('m', iosize=8, n_layers=1, d_model=32, neurons=[[3, 4]])
        torch.manual_seed(0)
        b = MinMaxRNC_LM('m', iosize=8, n_layers=1, d_model=32, neurons=[(3, 4)])
        assert _n_params(a) == _n_params(b)
        assert a._backbone_cfg.neurons == b._backbone_cfg.neurons

    def test_scalar_and_array_neurons_combine(self):
        only_arrays = MinMaxRNC_LM('m', iosize=8, n_layers=1, d_model=32, neurons=[[3, 4]])
        both = MinMaxRNC_LM('m', iosize=8, n_layers=1, d_model=32, scalar_neurons=8,
                            neurons=[[3, 4]])
        assert _n_params(both) > _n_params(only_arrays)

    def test_neuron_array_forward_shapes(self):
        model = MinMaxRNC_LM('m', iosize=8, n_layers=2, d_model=32, neurons=[[3, 4]]).eval()
        with torch.no_grad():
            logits, state = model(_x(vocab=8), unroll_steps=SEQ, return_state=True)
        assert logits.shape == (BATCH, SEQ, 8)
        assert len(state) == 2

    def test_list_valued_args_from_meta_template(self):
        # Template parameters inside list-valued arguments are substituted
        # element-wise, keeping their types.
        meta = Meta({
            'mm__na_{N:int}x{S:int}': {
                'class': 'MinMaxRNC_LM',
                'args': {'n_layers': 1, 'd_model': 90, 'scalar_neurons': 0,
                         'neurons': [['{S}', '{N}']], 's_r_init': 'asymmetric'},
            }
        })
        args = meta['mm__na_(16)x(3)']['args']
        assert args['neurons'] == [[3, 16]]
        model = MinMaxRNC_LM('m', iosize=8, **args)
        assert _n_params(model, trainable_only=True) == 76_166


# ---------------------------------------------------------------------------
# Forward pass, state and unrolling
# ---------------------------------------------------------------------------

class TestForward:

    def test_logits_shape_and_dtype(self):
        model = _model().eval()
        with torch.no_grad():
            logits = model(_x(), unroll_steps=SEQ)
        assert logits.shape == (BATCH, SEQ, VOCAB)
        assert logits.dtype == torch.float32

    def test_state_structure(self):
        model = _model().eval()
        with torch.no_grad():
            _, state = model(_x(), unroll_steps=SEQ, return_state=True)
        assert isinstance(state, list) and len(state) == 2
        for layer_state in state:
            for t in layer_state.values():
                if torch.is_tensor(t):
                    assert t.shape[0] == BATCH

    @pytest.mark.parametrize('unroll_steps', [1, 5, 8, SEQ])
    def test_unroll_steps_invariance(self, unroll_steps):
        torch.manual_seed(0)
        model = _model().eval()
        x = _x()
        with torch.no_grad():
            ref = model(x, unroll_steps=SEQ)
            out = model(x, unroll_steps=unroll_steps)
        assert torch.allclose(ref, out, atol=1e-5)

    def test_chunked_with_state_equals_unchunked(self):
        torch.manual_seed(0)
        model = _model().eval()
        x = _x()
        with torch.no_grad():
            ref = model(x, unroll_steps=SEQ)
            state, chunks = None, []
            for x_chunk in x.split(7, dim=1):
                logits, state = model(x_chunk, state, unroll_steps=x_chunk.shape[1],
                                      return_state=True)
                chunks.append(logits)
        assert torch.allclose(ref, torch.cat(chunks, dim=1), atol=1e-5)

    def test_reset_reinitialises(self):
        model = _model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.reset()
        after = model.state_dict()
        assert before.keys() == after.keys()
        assert any(not torch.equal(before[k], after[k]) for k in before)


# ---------------------------------------------------------------------------
# Precision (model-level AMP through the wrapper)
# ---------------------------------------------------------------------------

class TestPrecision:

    def test_defaults_are_none(self):
        cfg = _model()._backbone_cfg
        assert cfg.amp_dtype is None
        assert cfg.recurrence_dtype is None

    def test_amp_and_recurrence_dtype_reach_backbone(self):
        cfg = _model(amp_dtype='bf16', recurrence_dtype='fp16')._backbone_cfg
        assert cfg.amp_dtype == 'bf16'
        assert cfg.recurrence_dtype == 'fp16'

    def test_rejects_unknown_dtypes(self):
        with pytest.raises(ValueError):
            _model(amp_dtype='fp64')
        with pytest.raises(ValueError):
            _model(recurrence_dtype='fp64')

    def test_bf16_amp_logits_are_float32_and_close_to_fp32(self):
        x = _x()
        torch.manual_seed(0)
        ref = _model().eval()
        torch.manual_seed(0)
        amp = _model(amp_dtype='bf16').eval()
        with torch.no_grad():
            out_ref = ref(x, unroll_steps=SEQ)
            out_amp = amp(x, unroll_steps=SEQ)
        assert out_amp.dtype == torch.float32
        assert torch.allclose(out_ref, out_amp, atol=5e-2, rtol=5e-2)

    def test_bf16_amp_fp16_recurrence_backward_runs(self):
        model = _model(amp_dtype='bf16', recurrence_dtype='fp16', n_layers=1, d_model=32, d_state=8)
        x, y = _x(), _x()
        logits = model(x, unroll_steps=SEQ)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
