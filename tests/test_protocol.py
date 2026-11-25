# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Tests for config overrides (benchmarks.utils.config), the per-experiment
`protocol:` block (Experiment, Evaluation) and the seeded CI bootstrap."""
import pytest

from benchmarks.utils import config
from benchmarks.utils.ci import ConfidenceIntervals
from benchmarks.utils.meta import Meta
from benchmarks.stoppers import stoppers
from benchmarks.experiments import experiment as experiment_module
from benchmarks.experiments import utils as experiment_utils
from benchmarks.experiments.experiment import Experiment
from benchmarks.experiments.experiments import experiments
from benchmarks.evaluations import evaluation as evaluation_module
from benchmarks.evaluations.evaluation import Evaluation

pytestmark = pytest.mark.group('config')

MODEL   = 'minmaxrnc__d_(90)__l_(2)__s_(40)__og_true__cv_basic'
TRAINER = 'trainer_default'


# ---------------------------------------------------------------------------
# config.overrides
# ---------------------------------------------------------------------------

class TestConfigOverrides:

    def test_defaults_come_from_meta_config(self):
        assert config.get('stopper__patience') == 5
        assert config.get('ci_seed') is None
        assert config.get('val_acc_giveup_threshold') is None
        assert config.get('best_model') == 'best_epoch'
        assert config.get('compile_model') is False
        assert config.get('eval_max_batch_size') is None

    def test_override_applies_inside_and_is_restored_after(self):
        with config.overrides({'stopper__patience': 3}):
            assert config.get('stopper__patience') == 3
        assert config.get('stopper__patience') == 5

    def test_innermost_override_wins(self):
        with config.overrides({'ci_max_n': 5, 'stopper__patience': 3}):
            with config.overrides({'ci_max_n': 7}):
                assert config.get('ci_max_n') == 7
                assert config.get('stopper__patience') == 3
            assert config.get('ci_max_n') == 5
        assert config.get('ci_max_n') == 10

    def test_override_to_none_is_honoured(self):
        with config.overrides({'max_eval_runtime': None}):
            assert config.get('max_eval_runtime') is None

    def test_restored_after_exception(self):
        with pytest.raises(RuntimeError):
            with config.overrides({'ci_max_n': 2}):
                raise RuntimeError
        assert config.get('ci_max_n') == 10

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            with config.overrides({'stopper_patience': 3}):
                pass

    def test_empty_or_none_is_a_no_op(self):
        with config.overrides(None):
            assert config.get('ci_max_n') == 10
        with config.overrides({}):
            assert config.get('ci_max_n') == 10

    def test_returned_values_are_copies(self):
        with config.overrides({'ci_accuracy__cum': [[1.0, 0.01]]}):
            config.get('ci_accuracy__cum').append([0.5, 0.5])
            assert config.get('ci_accuracy__cum') == [[1.0, 0.01]]

    def test_stopper_reads_overridden_settings(self):
        with config.overrides({'stopper__enabled': False, 'stopper__patience': 3}):
            stopper = stoppers.instantiate('emastopper')
        assert stopper.enabled is False
        assert stopper.patience == 3


# ---------------------------------------------------------------------------
# Experiment protocol
# ---------------------------------------------------------------------------

class _RecordingTrainer:
    def __init__(self, record):
        self.record = record
        self.record['at_instantiate'] = self._snapshot()

    @staticmethod
    def _snapshot():
        return {k: config.get(k) for k in
                ('stopper__enabled', 'stopper__patience', 'ci_max_n', 'max_train_runtime')}

    def fit(self):
        self.record['at_fit'] = self._snapshot()


class _RecordingTrainers:
    def __init__(self, record):
        self.record = record

    def instantiate(self, name, **kwargs):
        self.record['epochs'] = kwargs['epochs']
        return _RecordingTrainer(self.record)

    def str(self, name):
        return name


def _experiment(protocol=None):
    return Experiment(
        'latching_(1)',
        models_trainers=[{'model': MODEL, 'trainer': TRAINER}],
        metric='TokenAccuracy', loss='CrossEntropyLoss',
        train_dataset='latching_train_(1)', val_dataset='latching_val_(1)',
        epochs=7, protocol=protocol,
    )


class TestExperimentProtocol:

    @pytest.fixture
    def record(self, tmp_path, monkeypatch):
        record = {}
        monkeypatch.setattr(experiment_utils, 'OUTPUT_DIR', str(tmp_path))
        monkeypatch.setattr(experiment_module, 'trainers', _RecordingTrainers(record))
        return record

    def test_protocol_applies_to_trainer_construction_and_fit(self, record):
        protocol = {'stopper__enabled': False, 'stopper__patience': 3,
                    'ci_max_n': 5, 'max_train_runtime': 43200}
        _experiment(protocol).run_model(MODEL, TRAINER)
        assert record['at_instantiate'] == protocol
        assert record['at_fit'] == protocol
        assert record['epochs'] == 7
        # restored afterwards
        assert config.get('ci_max_n') == 10

    def test_no_protocol_keeps_global_config(self, record):
        _experiment().run_model(MODEL, TRAINER)
        assert record['at_fit'] == {'stopper__enabled': True, 'stopper__patience': 5,
                                    'ci_max_n': 10, 'max_train_runtime': 14400}

    def test_unknown_protocol_key_fails_before_training(self, record):
        with pytest.raises(KeyError):
            _experiment({'patience': 3}).run_model(MODEL, TRAINER)
        assert 'at_fit' not in record

    def test_existing_experiments_have_no_protocol(self):
        assert experiments.get_protocol('latching_(1)') == {}

    def test_protocol_block_survives_meta_templating(self):
        meta = Meta({
            'exp_{N:int}': {
                'class': 'Experiment',
                'args': {'epochs': 30, 'protocol': {'stopper__patience': 3, 'ci_max_n': 5}},
            }
        })
        assert meta['exp_(1)']['args']['protocol'] == {'stopper__patience': 3, 'ci_max_n': 5}


# ---------------------------------------------------------------------------
# Evaluation protocol: global < experiment protocol < evaluation protocol
# ---------------------------------------------------------------------------

class _RecordingEvaluator:
    def __init__(self, record):
        self.record = record

    def evaluate(self):
        self.record['at_evaluate'] = {k: config.get(k) for k in
                                      ('ci_max_n', 'max_eval_runtime', 'eval_max_batch_size')}


class _RecordingEvaluators:
    def __init__(self, record):
        self.record = record

    def instantiate(self, name, **kwargs):
        self.record['kwargs'] = kwargs
        return _RecordingEvaluator(self.record)


class TestEvaluationProtocol:

    @pytest.fixture
    def record(self, tmp_path, monkeypatch):
        record = {}
        monkeypatch.setattr(evaluation_module, 'evaluators', _RecordingEvaluators(record))
        monkeypatch.setattr(evaluation_module, 'has_already_been_completed', lambda *a: False)
        monkeypatch.setattr(evaluation_module, 'trained_model_exists', lambda *a: True)
        monkeypatch.setattr(evaluation_module, 'load_model_state_dict', lambda *a: {})
        monkeypatch.setattr(evaluation_module, 'get_output_dir', lambda *a: str(tmp_path))
        monkeypatch.setattr(evaluation_module, 'write_completed', lambda *a: None)
        return record

    def _evaluation(self, monkeypatch, experiment_protocol, protocol=None):
        monkeypatch.setattr(experiments, 'get_protocol', lambda name: dict(experiment_protocol))
        return Evaluation('latching_(1)', experiment='latching_(1)',
                          metric='ChunkedTokenAccuracy', dataset='latching_evaluation_(1)',
                          models_trainers=[{'model': MODEL, 'trainer': TRAINER}],
                          protocol=protocol)

    def test_inherits_experiment_protocol(self, record, monkeypatch):
        ev = self._evaluation(monkeypatch, {'ci_max_n': 5, 'eval_max_batch_size': 256})
        ev.run_model(MODEL, TRAINER)
        assert record['at_evaluate'] == {'ci_max_n': 5, 'max_eval_runtime': 72000,
                                         'eval_max_batch_size': 256}

    def test_evaluation_protocol_takes_precedence(self, record, monkeypatch):
        ev = self._evaluation(monkeypatch, {'ci_max_n': 5, 'max_eval_runtime': 43200},
                              protocol={'max_eval_runtime': 3600})
        ev.run_model(MODEL, TRAINER)
        assert record['at_evaluate'] == {'ci_max_n': 5, 'max_eval_runtime': 3600,
                                         'eval_max_batch_size': None}
        assert config.get('max_eval_runtime') == 72000

    def test_trainer_compile_flag_is_passed_to_the_evaluator(self, record, monkeypatch):
        self._evaluation(monkeypatch, {}).run_model(MODEL, TRAINER)
        assert record['kwargs']['compile_model'] is None


# ---------------------------------------------------------------------------
# Seeded CI bootstrap
# ---------------------------------------------------------------------------

class TestCiSeed:

    SCORES = [0.91, 0.95, 0.97, 0.99]

    def test_seeded_bootstrap_is_deterministic(self):
        ci = ConfidenceIntervals()
        with config.overrides({'ci_seed': 123}):
            a = ci.mean_bootstrap(self.SCORES, 0.01, B=500)
            ci.mean_bootstrap([0.1, 0.2], 0.01, B=500)   # unrelated call in between
            b = ci.mean_bootstrap(self.SCORES, 0.01, B=500)
        assert a == b

    def test_explicit_seed_argument(self):
        ci = ConfidenceIntervals()
        assert ci.mean_bootstrap(self.SCORES, 0.01, B=500, seed=7) == \
               ci.mean_bootstrap(self.SCORES, 0.01, B=500, seed=7)

    def test_unseeded_by_default_uses_the_shared_rng(self):
        ci = ConfidenceIntervals()
        assert ci._get_rng(None) is ci.rng

    def test_bounds_contain_the_mean(self):
        ci = ConfidenceIntervals()
        mean, low, high = ci.mean_bootstrap(self.SCORES, 0.01, B=500, seed=1)
        assert low <= mean <= high
        assert mean == pytest.approx(sum(self.SCORES) / len(self.SCORES))
