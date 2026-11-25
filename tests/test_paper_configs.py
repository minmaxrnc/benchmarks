# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Every experiment of the paper (configs/paper/*.yaml) resolves to the
intended models, trainers, datasets and protocol, and its models build.
No datasets on disk needed."""
import warnings
from pathlib import Path

import pytest
import yaml

from benchmarks.definitions import PAPER_CONFIGS_DIR
from benchmarks.experiments.experiments import experiments
from benchmarks.evaluations.evaluations import evaluations
from benchmarks.models.models import models
from benchmarks.trainers.trainers import trainers
from benchmarks.datasets.datasets import datasets
from benchmarks.optimizers.optimizers import optimizers
from benchmarks.schedulers.schedulers import schedulers
from benchmarks.runnables import reproduce
from benchmarks.utils import config

pytestmark = pytest.mark.group('paper')

S_L2 = 'minmax_s__l_(2)'

LONGRANGE_PROTOCOL = {
    'ci_max_n': 5,
    'stopper__patience': 3,
    'max_train_runtime': 43200,
    'max_train_runtime__per_params': 43200,
    'max_train_runtime__per_seed': 43200,
    'eval_max_batch_size': 256,
}


def _groups_pairs(layers, ds, trainer_names):
    return [(f'minmax_groups__l_({l})__d_({d})', t) for l in layers for d in ds for t in trainer_names]


# experiment -> (model/trainer pairs, train dataset, val dataset, epochs, protocol)
EXPECTED = {}
for n in range(1, 9):
    EXPECTED[f'main_latching_({n})'] = (
        [(S_L2, 'trainer_minmax')], f'latching_train_({n})', f'latching_val_({n})', 30, {})
    EXPECTED[f'throughput_latching_({n})'] = (
        [('minmax_s__l_(2)__amp_bf16', 'trainer_minmax_compiled_amp')],
        f'latching_train_({n})', f'latching_val_({n})', 30, {})
for n in range(1, 6):
    EXPECTED[f'main_sequences_({n})'] = (
        [(f'minmax_s__l_({n + 1})', 'trainer_minmax')], f'sequences_train_({n})', f'sequences_val_({n})', 30, {})
    EXPECTED[f'throughput_sequences_({n})'] = (
        [(f'minmax_s__l_({n + 1})__amp_bf16', 'trainer_minmax_compiled_amp')],
        f'sequences_train_({n})', f'sequences_val_({n})', 30, {})
for n in (0, 1):
    EXPECTED[f'main_inductionheads_({n})'] = (
        [(S_L2, 'trainer_minmax'), (S_L2, 'trainer_minmax_bs8'),
         ('minmax_m__l_(2)', 'trainer_minmax'), ('minmax_s__l_(3)', 'trainer_minmax')],
        f'inductionheads_train_({n})', f'inductionheads_val_({n})', 30, {})
    EXPECTED[f'throughput_inductionheads_({n})'] = (
        [('minmax_s__l_(2)__amp_bf16', 'trainer_minmax_compiled_amp'),
         ('minmax_m__l_(2)__amp_bf16', 'trainer_minmax_compiled_amp')],
        f'inductionheads_train_({n})', f'inductionheads_val_({n})', 30, {})
EXPECTED['groups_s3'] = (
    _groups_pairs((1, 2), (3, 4, 5), ('trainer_groups', 'trainer_groups_bs8')),
    'groupstatetracking_s3_train', 'groupstatetracking_s3_val', 30,
    {'stopper__patience': 3, 'val_acc_giveup_threshold': 0.5})
EXPECTED['groups_s4'] = (
    _groups_pairs((1,), (4, 5), ('trainer_groups',)),
    'groupstatetracking_s4_train__large', 'groupstatetracking_s4_val', 100,
    {'stopper__enabled': False, 'val_acc_giveup_threshold': 0.5})
EXPECTED['longrange_sequences_(1)'] = (
    [('minmax_s__l_(2)__amp_bf16__rec_fp16', 'trainer_minmax_longseq')],
    'sequences_grad_train_(1)', 'sequences_grad_val_(1)', 30, LONGRANGE_PROTOCOL)

EVALUATION_DATASET = {
    'main_latching': 'latching_evaluation_({})',
    'main_sequences': 'sequences_evaluation_({})',
    'main_inductionheads': 'inductionheads_evaluation_({})',
    'throughput_latching': 'latching_evaluation_({})',
    'throughput_sequences': 'sequences_evaluation_({})',
    'throughput_inductionheads': 'inductionheads_evaluation_({})',
    'groups_s3': 'groupstatetracking_s3_evaluation',
    'groups_s4': 'groupstatetracking_s4_evaluation',
    'longrange_sequences': 'sequences_grad_evaluation_({})',
}


def _paper_configs():
    configs = {}
    for path in sorted(Path(PAPER_CONFIGS_DIR).glob('*.yaml')):
        with open(path) as f:
            configs[path.stem] = yaml.safe_load(f)
    return configs


CONFIGS = _paper_configs()
ALL_EXPERIMENTS = [e for cfg in CONFIGS.values() for e in cfg['experiments']]
ALL_EVALUATIONS = [e for cfg in CONFIGS.values() for e in cfg['evaluations']]


def _pairs(experiment):
    return [(mt['model'], mt['trainer']) for mt in experiments.get_models_trainers(experiment)]


class TestPaperConfigFiles:

    def test_families(self):
        assert sorted(CONFIGS) == ['groups', 'longrange', 'main', 'throughput']
        assert reproduce.families() == ['groups', 'longrange', 'main', 'throughput']

    def test_every_paper_experiment_is_listed_once(self):
        assert sorted(ALL_EXPERIMENTS) == sorted(EXPECTED)
        assert len(ALL_EXPERIMENTS) == len(set(ALL_EXPERIMENTS))

    @pytest.mark.parametrize('family', sorted(CONFIGS))
    def test_each_experiment_has_its_evaluation(self, family):
        assert CONFIGS[family]['evaluations'] == CONFIGS[family]['experiments']

    def test_select_by_prefix(self):
        cfg = reproduce.select(reproduce.load_family('main'), 'main_sequences')
        assert cfg['experiments'] == [f'main_sequences_({n})' for n in range(1, 6)]
        assert cfg['evaluations'] == cfg['experiments']

    def test_paper_experiments_are_the_default(self):
        from benchmarks.utils import meta
        for kind in ('experiments', 'evaluations'):
            enabled = set(meta.load(kind, only_enabled=True).keys())
            assert enabled == set(ALL_EXPERIMENTS)


class TestExperimentResolution:

    @pytest.mark.parametrize('name', sorted(EXPECTED))
    def test_resolves_as_intended(self, name):
        pairs, train_ds, val_ds, epochs, protocol = EXPECTED[name]
        args = experiments.get_meta(name)['args']
        assert _pairs(name) == pairs
        assert args['train_dataset'] == train_ds
        assert args['val_dataset'] == val_ds
        assert args['epochs'] == epochs
        assert experiments.get_protocol(name) == protocol
        # protocol keys are valid config keys
        with config.overrides(protocol):
            pass

    def test_sequences_5_has_six_layers(self):
        (model, _), = _pairs('main_sequences_(5)')
        assert model == 'minmax_s__l_(6)'
        assert models.get_meta(model)['args']['n_layers'] == 6

    @pytest.mark.parametrize('name', sorted(EXPECTED))
    def test_datasets_trainers_and_models_resolve(self, name):
        args = experiments.get_meta(name)['args']
        for ds in (args['train_dataset'], args['val_dataset']):
            assert datasets.get_meta(ds)['class']
        for model, trainer in _pairs(name):
            assert models.get_meta(model)['class'] == 'MinMaxRNC_LM'
            targs = trainers.get_meta(trainer)['args']
            assert optimizers.get_meta(targs['optimizer'])
            assert schedulers.get_meta(targs['scheduler'])

    @pytest.mark.parametrize('name', sorted(ALL_EVALUATIONS))
    def test_evaluation_resolves(self, name):
        args = evaluations.get_meta(name)['args']
        assert args['experiment'] == name
        family = name.split('_(')[0]
        n = name[len(family) + 2:-1] if '_(' in name else None
        assert args['dataset'] == EVALUATION_DATASET[family].format(n)
        assert datasets.get_length(args['dataset']) == 1_024_000


class TestModelRecipes:

    def test_main_battery_recipe(self):
        for name in ('minmax_s__l_(2)', 'minmax_m__l_(2)'):
            a = models.get_meta(name)['args']
            assert (a['output_gate'], a['conv_type'], a['s_r_init']) == (False, 'basic', 'asymmetric')
            assert a['d_model'] == 90
        assert models.get_meta('minmax_s__l_(2)')['args']['scalar_neurons'] == 40
        assert models.get_meta('minmax_m__l_(2)')['args']['scalar_neurons'] == 90

    def test_amp_variants_differ_only_in_precision(self):
        base = models.get_meta('minmax_s__l_(3)')['args']
        amp = models.get_meta('minmax_s__l_(3)__amp_bf16')['args']
        lr = models.get_meta('minmax_s__l_(3)__amp_bf16__rec_fp16')['args']
        assert amp == base | {'amp_dtype': 'bf16'}
        assert lr == base | {'amp_dtype': 'bf16', 'recurrence_dtype': 'fp16'}

    def test_trainers(self):
        get = lambda t: trainers.get_meta(t)['args']
        assert get('trainer_minmax')['train_batch_size'] == 64
        assert get('trainer_minmax_bs8')['train_batch_size'] == 8
        assert get('trainer_minmax_compiled_amp')['compile_model'] is True
        assert get('trainer_groups')['scheduler'] == 'cosine'
        assert get('trainer_groups_bs8')['train_batch_size'] == 8
        longseq = get('trainer_minmax_longseq')
        assert (longseq['train_batch_size'], longseq['grad_accum_steps'], longseq['max_unroll_steps'],
                longseq['compile_model']) == (64, 2, 65536, True)
        for t in ('trainer_minmax', 'trainer_groups', 'trainer_minmax_longseq'):
            assert get(t)['optimizer'] == 'adam'
            assert get(t)['lr'] == 0.001
        assert optimizers.get_meta('adam')['args']['weight_decay'] == 1e-4


def _n_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


class TestModelsBuild:
    """Build every model of every paper experiment at its benchmark's
    vocabulary size, without warnings."""

    @pytest.mark.parametrize('name', sorted(EXPECTED))
    def test_builds_without_warnings(self, name):
        iosize = datasets.get_iosize(experiments.get_meta(name)['args']['train_dataset'])
        for model, _ in _pairs(name):
            with warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                models.instantiate(model, iosize)

    @pytest.mark.parametrize('model, dataset, trainable', [
        # Appendix tab:groups_models (one layer): 76k, 91k, 108k on S_3; 92k, 110k on S_4
        ('minmax_groups__l_(1)__d_(3)', 'groupstatetracking_s3_train', 76_166),
        ('minmax_groups__l_(1)__d_(4)', 'groupstatetracking_s3_train', 90_710),
        ('minmax_groups__l_(1)__d_(5)', 'groupstatetracking_s3_train', 108_166),
        ('minmax_groups__l_(1)__d_(4)', 'groupstatetracking_s4_train__large', 92_330),
        ('minmax_groups__l_(1)__d_(5)', 'groupstatetracking_s4_train__large', 109_786),
        # Appendix app:experiments-longrange: 122,582 trainable parameters
        ('minmax_s__l_(2)__amp_bf16__rec_fp16', 'sequences_grad_train_(1)', 122_582),
    ])
    def test_parameter_counts(self, model, dataset, trainable):
        assert _n_params(models.instantiate(model, datasets.get_iosize(dataset))) == trainable

    def test_groups_vocabulary(self):
        assert datasets.get_iosize('groupstatetracking_s3_train') == 8
        assert datasets.get_iosize('groupstatetracking_s4_train__large') == 26

    def test_inductor_workaround_only_for_neuron_arrays(self):
        assert models.instantiate('minmax_groups__l_(1)__d_(3)', 8).inductor_config() == \
            {'triton.coalesce_tiling_analysis': False}
        assert models.instantiate('minmax_s__l_(2)', 20).inductor_config() == {}

    def test_compile_applies_the_workaround(self, monkeypatch):
        import torch
        import torch._inductor.config as inductor_config
        from benchmarks.utils import compile as compile_module
        monkeypatch.setattr(inductor_config.triton, 'coalesce_tiling_analysis', True)
        monkeypatch.setattr(torch, 'compile', lambda m: ('compiled', m))
        scalar = models.instantiate('minmax_s__l_(2)', 20)
        assert compile_module.compile_model(scalar) == ('compiled', scalar)
        assert inductor_config.triton.coalesce_tiling_analysis is True
        arrays = models.instantiate('minmax_groups__l_(1)__d_(3)', 8)
        assert compile_module.compile_model(arrays) == ('compiled', arrays)
        assert inductor_config.triton.coalesce_tiling_analysis is False

    def test_unknown_inductor_entries_are_skipped(self):
        from benchmarks.utils.compile import apply_inductor_config
        assert apply_inductor_config({'triton.no_such_flag_in_any_version': False}) == {}
