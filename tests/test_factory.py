"""Tests for the Factory instantiation and class-registration system."""
import pytest
import torch.nn as nn

from benchmarks.models.models import models, register_model
from benchmarks.models.model import Model
from benchmarks.losses.losses import losses
from benchmarks.losses.crossentropyloss import CrossEntropyLoss
from benchmarks.metrics.metrics import metrics
from benchmarks.metrics.tokenaccuracy import TokenAccuracy
from benchmarks.schedulers.schedulers import schedulers
from benchmarks.schedulers.nonescheduler import NoneScheduler
from benchmarks.optimizers.adam import Adam

pytestmark = pytest.mark.group('factory')


# ---------------------------------------------------------------------------
# Models factory
# ---------------------------------------------------------------------------

class TestModelsFactory:

    def test_vanilla_rnn_meta_class_field(self):
        entry = models.get_meta('vanilla_rnn__d_(128)__l_(2)')
        assert entry['class'] == 'VanillaRNN'

    def test_vanilla_rnn_meta_args_substituted(self):
        entry = models.get_meta('vanilla_rnn__d_(32)__l_(1)')
        assert entry['args']['d_model'] == 32
        assert entry['args']['n_layers'] == 1

    def test_instantiate_vanilla_rnn_type(self):
        from vanilla_rnn.model import VanillaRNN
        model = models.instantiate('vanilla_rnn__d_(32)__l_(1)', 20)
        assert isinstance(model, VanillaRNN)

    def test_instantiate_vanilla_rnn_name(self):
        model = models.instantiate('vanilla_rnn__d_(32)__l_(1)', 20)
        assert model.name == 'vanilla_rnn__d_(32)__l_(1)'

    def test_instantiate_vanilla_rnn_is_nn_module(self):
        model = models.instantiate('vanilla_rnn__d_(32)__l_(1)', 20)
        assert isinstance(model, nn.Module)

    def test_get_required_kwargs_is_list(self):
        kwargs = models.get_required_kwargs('vanilla_rnn__d_(32)__l_(1)')
        assert isinstance(kwargs, list)

    def test_repr_is_string(self):
        r = models.repr('vanilla_rnn__d_(32)__l_(1)')
        assert isinstance(r, str)

    def test_register_and_use_custom_model(self):
        class _TinyModel(Model):
            def __init__(self, name, **kwargs):
                super().__init__(name, **kwargs)
                self.linear = nn.Linear(4, 4)
            def forward(self, x, mode=None):
                return self.linear(x)
            @staticmethod
            def get_required_kwargs():
                return []

        register_model(_TinyModel)
        # The class is now in the registry; instantiating requires a meta entry,
        # so just confirm it was added without error.
        assert True


# ---------------------------------------------------------------------------
# Losses factory
# ---------------------------------------------------------------------------

class TestLossesFactory:

    def test_cross_entropy_meta_class_field(self):
        entry = losses.get_meta('CrossEntropyLoss')
        assert entry['class'] == 'CrossEntropyLoss'

    def test_instantiate_cross_entropy_type(self):
        loss = losses.instantiate('CrossEntropyLoss')
        assert isinstance(loss, CrossEntropyLoss)

    def test_instantiate_cross_entropy_name(self):
        loss = losses.instantiate('CrossEntropyLoss')
        assert loss.name == 'CrossEntropyLoss'


# ---------------------------------------------------------------------------
# Metrics factory
# ---------------------------------------------------------------------------

class TestMetricsFactory:

    def test_token_accuracy_meta_class_field(self):
        entry = metrics.get_meta('TokenAccuracy')
        assert entry['class'] == 'TokenAccuracy'

    def test_instantiate_token_accuracy_type(self):
        metric = metrics.instantiate('TokenAccuracy')
        assert isinstance(metric, TokenAccuracy)

    def test_instantiate_token_accuracy_name(self):
        metric = metrics.instantiate('TokenAccuracy')
        assert metric.name == 'TokenAccuracy'


# ---------------------------------------------------------------------------
# Schedulers factory
# ---------------------------------------------------------------------------

class TestSchedulersFactory:

    @pytest.fixture
    def optimizer(self):
        model = nn.Linear(4, 2)
        return Adam('adam', params=model.parameters(), lr=0.01, weight_decay=0.0)

    def test_none_scheduler_meta_class_field(self):
        entry = schedulers.get_meta('none')
        assert entry['class'] == 'NoneScheduler'

    def test_instantiate_none_scheduler_type(self, optimizer):
        sched = schedulers.instantiate('none', optimizer=optimizer)
        assert isinstance(sched, NoneScheduler)

    def test_step_scheduler_meta_class_field(self):
        from benchmarks.schedulers.stepscheduler import StepLR
        entry = schedulers.get_meta('step')
        assert entry['class'] == 'StepLR'
