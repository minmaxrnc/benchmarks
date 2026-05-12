# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for template-string matching and meta YAML loading."""
import pytest
from benchmarks.utils.templatestrings import match_template, compile_template
from benchmarks.utils import meta as meta_module

pytestmark = pytest.mark.group('meta')


class TestTemplateMatching:

    def test_match_two_int_params(self):
        params = match_template('rnn__d_{D:int}__l_{L:int}', 'rnn__d_(128)__l_(2)')
        assert params == {'D': 128, 'L': 2}

    def test_match_negative_int(self):
        params = match_template('x_{N:int}', 'x_(-1)')
        assert params is not None
        assert params['N'] == -1

    def test_match_single_int_param(self):
        params = match_template('sequences_train_{N:int}', 'sequences_train_(1)')
        assert params == {'N': 1}

    def test_no_match_different_prefix(self):
        params = match_template('rnn__d_{D:int}', 'transformer__d_(128)')
        assert params is None

    def test_no_match_int_without_parens(self):
        # Integer values in the concrete name must be parenthesised
        params = match_template('rnn__d_{D:int}', 'rnn__d_128')
        assert params is None

    def test_match_no_params_exact_string(self):
        params = match_template('CrossEntropyLoss', 'CrossEntropyLoss')
        assert params == {}

    def test_no_match_literal_mismatch(self):
        params = match_template('CrossEntropyLoss', 'MSELoss')
        assert params is None

    def test_matched_int_is_python_int(self):
        params = match_template('model__d_{D:int}', 'model__d_(64)')
        assert isinstance(params['D'], int)

    def test_partial_string_does_not_match(self):
        # Template requires full-string match (anchored ^ … $)
        params = match_template('rnn__d_{D:int}', 'rnn__d_(64)__extra')
        assert params is None

    def test_compile_template_returns_pattern_and_converters(self):
        regex, converters = compile_template('model_{N:int}')
        assert 'N' in converters
        assert regex.match('model_(7)') is not None
        assert regex.match('model_7') is None


class TestMetaLoading:

    def test_models_scope_loads(self):
        meta = meta_module.load('models')
        assert len(meta) > 0

    def test_models_scope_contains_vanilla_rnn_template(self):
        meta = meta_module.load('models')
        assert any('vanilla_rnn' in k for k in meta.keys())

    def test_losses_scope_contains_cross_entropy(self):
        meta = meta_module.load('losses')
        assert 'CrossEntropyLoss' in meta.keys()

    def test_metrics_scope_contains_token_accuracy(self):
        meta = meta_module.load('metrics')
        assert 'TokenAccuracy' in meta.keys()

    def test_optimizers_scope_contains_adam(self):
        meta = meta_module.load('optimizers')
        assert 'adam' in meta.keys()

    def test_schedulers_scope_contains_step(self):
        meta = meta_module.load('schedulers')
        assert 'step' in meta.keys()

    def test_vanilla_rnn_lookup_resolves_class(self):
        meta = meta_module.load('models')
        entry = meta['vanilla_rnn__d_(128)__l_(2)']
        assert entry['class'] == 'VanillaRNN'

    def test_vanilla_rnn_lookup_substitutes_d_model(self):
        meta = meta_module.load('models')
        entry = meta['vanilla_rnn__d_(64)__l_(1)']
        assert entry['args']['d_model'] == 64

    def test_vanilla_rnn_lookup_substitutes_n_layers(self):
        meta = meta_module.load('models')
        entry = meta['vanilla_rnn__d_(128)__l_(3)']
        assert entry['args']['n_layers'] == 3

    def test_cross_entropy_lookup(self):
        meta = meta_module.load('losses')
        entry = meta['CrossEntropyLoss']
        assert entry['class'] == 'CrossEntropyLoss'

    def test_missing_key_raises(self):
        meta = meta_module.load('models')
        with pytest.raises(Exception):
            _ = meta['does_not_exist']
