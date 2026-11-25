# Benchmarks

A model-agnostic benchmarking framework for sequence models. It provides synthetic tasks that stress-test long-range memory, context binding and state tracking:

- **Latching** — remember and reproduce a value across a long sequence
- **Sequences** — match multiple ordered sub-sequences (also in a long-range variant, with training sequences of 131,072 steps)
- **Induction Heads** — retrieve a token that appeared after a repeated context (as in the [Mamba paper](https://arxiv.org/pdf/2312.00752))
- **Word problems on permutation groups** — track the running product of a sequence of elements of S_3 or S_4

Training uses bootstrap confidence intervals to automatically determine how many random seeds are needed, making comparisons statistically rigorous.

---

## Reproducing the paper's MinMax experiments

Version 0.2.0 contains the configurations of the MinMax experiments of the paper's appendices, with the
[`minmaxrnc`](https://github.com/minmaxrnc/model) package: the main battery (Latching, Sequences,
InductionHeads), the compiled+AMP throughput runs, the word problems on S_3 and S_4, and the long-range
Sequences experiment with its majority-class baseline, event-gap statistics and gradient probe.

```bash
python main.py reproduce main          # generate -> train -> evaluate; also: throughput, groups, longrange
python main.py reproduce main --tiny   # CPU smoke run of the whole pipeline
```

See [docs/reproducing.md](docs/reproducing.md) for the experiment → table/figure map, the
configurations, where the numbers are in the outputs, and compute estimates, and
[docs/known-issues.md](docs/known-issues.md) for the behaviours kept as run (generator quirks, the
best-model option, per-chunk minimum accuracy) and the differences from the paper's text. The ablation
of Appendix K and a script that generates the paper's tables are not part of this version.

---

## Installation

```bash
git clone git@github.com:minmaxrnc/benchmarks.git
cd benchmarks
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Plugging in a model

The framework ships two models out of the box: a vanilla RNN (included as a basic example) and
[MinMax RNC](https://github.com/minmaxrnc/model) (installed via the requirements above).
You can also plug in your own model in two steps.

### Step 1 — register the Python class

Your model package should call `register_model` when imported:

```python
# mymodel/__init__.py
from benchmarks import register_model
from .model import MyModel_LM

register_model(MyModel_LM)
```

Your class must subclass `benchmarks.models.model.Model` (which itself subclasses `torch.nn.Module`):

```python
from benchmarks.models.model import Model

class MyModel_LM(Model):
    def __init__(self, name, d_model, n_blocks, vocab_size=256, **kwargs):
        super().__init__(name)
        # build your model here

    def forward(self, x, state=None, unroll_steps=-1, return_state=False):
        # x:           (batch, seq_len) int tensor
        # return:      (batch, seq_len, vocab_size) float logits
        #              or (logits, new_state) if return_state is True
        ...

    def supports_unroll_steps(self):
        # Return True if forward() uses the unroll_steps argument.
        # The trainer and evaluator only pass unroll_steps when this returns True.
        # The base class returns False; override only when your model supports it.
        return False
```

The `unroll_steps` argument chunks the sequence into pieces of that length before passing them
through the model recurrently. It is useful for models with an explicit recurrent state that can
process the sequence incrementally — it trades a small amount of extra sequential work for a large
reduction in peak memory. For models not supporting `unroll_steps`, the parameter should have no
effect. 

### Step 2 — add config entries

**`meta/meta.models.yaml`** — define named model configurations:

```yaml
'mymodel__layers_{N:int}__small':
  class: MyModel_LM
  args:
    n_blocks: '{N}'
    d_model: 90

'mymodel__layers_{N:int}__medium':
  class: MyModel_LM
  args:
    n_blocks: '{N}'
    d_model: 256
```

Template keys (`{N:int}`) are expanded when an experiment references a specific instance like `mymodel__layers_(2)__small`.
Template parameters may also appear inside list-valued arguments (e.g. `neurons: [['{S}', '{N}']]`); each element is substituted with its type.

The MinMax RNC wrapper (`models/minmax_rnc`, `minmaxrnc>=0.4.1`) takes `n_layers`, `d_model`, `scalar_neurons` (or its legacy name `d_state`),
`neurons` (further neuron arrays, as `[[d_state, num_neurons], ...]`), `output_gate`, `conv_type`, `ffn_dropout`, `s_r_init`,
`amp_dtype` (model-level autocast: `null`, `bf16` or `fp16`) and `recurrence_dtype` (`null`, `fp32`, `bf16` or `fp16`).
Mixed precision is set per model through `amp_dtype`/`recurrence_dtype`; the trainer and the evaluator use no autocast of their own.

**`meta/meta.experiments.t.yaml`** — add your model to the benchmark experiments:

```yaml
'latching_{N:int}':
  class: Experiment
  args:
    epochs: 30
    train_dataset: latching_train_({N})
    val_dataset: latching_val_({N})
    loss: CrossEntropyLoss
    metric: TokenAccuracy
    models_trainers:
      - model: mymodel__layers_(2)__small
        trainer: trainer_default
```

---

## Running benchmarks

```bash
# Generate and save all datasets to disk
python main.py generate

# Train all enabled experiments
python main.py train

# Run evaluations on trained models
python main.py evaluate

# Aggregate throughput and gradient/state norms from the outputs on disk
python main.py throughput_report

# Run one family of the paper's experiments (configs/paper/<family>.yaml)
python main.py reproduce <family> [--tiny] [--steps generate train evaluate] [--select PREFIX]

# List available runnables
python main.py ls
```

### Selecting a subset at runtime

By default `generate`, `train` and `evaluate` run every entry that has `enabled: true` in the corresponding meta file. To override this without editing the meta files, create a `config/run.yaml` file at the project root:

```yaml
experiments:
  - latching_(1)
  - latching_(3)
  - inductionheads_(0)

evaluations:
  - latching_(1)
  - inductionheads_(0)
```

Each list contains the exact experiment or evaluation names as they appear in `meta/meta.experiments.yaml` / `meta/meta.evaluations.yaml`. When a key is present the corresponding `enabled` flags in the meta files are ignored — only the listed names are run. When a key is absent the meta's `enabled` flags are used as normal. You may include only `experiments`, only `evaluations`, or both.

The `config/` directory is listed in `.gitignore` so the file is treated as a local runtime override and never committed.

A custom path can be passed directly on the command line with `--config` / `-c`:

```bash
python main.py train --config path/to/my_run.yaml
python main.py evaluate -c path/to/my_run.yaml
python main.py generate -c configs/paper/groups.yaml
```

The runtime configs of the paper's experiments are in `configs/paper/` (`main`, `throughput`, `groups`,
`longrange`).

When calling the runnables programmatically you can pass a dictionary instead of a file via the `config` keyword argument:

```python
from benchmarks.runnables import train, evaluate

train.run(config={
    'experiments': ['latching_(1)', 'inductionheads_(0)'],
})

evaluate.run(config={
    'evaluations': ['latching_(1)'],
})
```

The `config` keyword argument takes priority over any CLI `--config` path and over the default `config/run.yaml` file.

### Outputs

#### `generate`

Datasets are written to `datasets/`. For each dataset name and seed the runnable produces:

| File | Description |
|---|---|
| `<dataset>__seed_<NN>.zip` | Serialised dataset samples |
| `stats__<dataset>__seed_<NN>.json` | Summary statistics for the dataset split |

Up to `ci_max_n` seeds are generated (default 10, or the value in the `protocol:` block of the experiment). Training and validation splits use short sequences; evaluation splits use long sequences.

#### `train`

One directory is created per experiment / model / trainer combination under `outputs/experiments/<experiment>/<model>__<trainer>/`:

| File | Description |
|---|---|
| `train_log__lr_*__bs_*__opt_*__sc_*__seed_*.csv` | Per-seed epoch log — columns: `epoch`, `lr_*`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `runtime`, `timelimit`, `earlystop`, then `tokens` (input tokens of the epoch, padding included), `epoch_time` (seconds, excluding data loading and statistics), `stats_time`, `data_time`, `throughput` (`tokens / epoch_time`), `grad_norm_mean/min/max` (pre-clipping), `state_norm_mean/max` |
| `run_env__lr_*__bs_*__opt_*__sc_*__seed_*.json` | Per-seed run environment — GPU, Python/torch/CUDA/minmaxrnc versions, model arguments, whether `torch.compile` was on, effective `unroll_steps` and `grad_accum_steps`, `best_model`, `ci_seed` (compiled runs also write `…__after_training.json` with dynamo's counters) |
| `checkpoint__lr_*__bs_*__opt_*__sc_*__seed_*__epoch_*.pt` | Weights after the epochs listed in the trainer's `save_checkpoint_epochs` (none by default) |
| `train_log__lr_*__bs_*__opt_*__sc_*.json` | Per-hyperparameter-combo summary — best and CI-bounded val accuracy / loss, total time, timelimit flag |
| `best_model__lr_*__bs_*__opt_*__sc_*.pt` | Best model checkpoint for that hyperparameter combo |
| `best_model.pt` | Best model overall (loaded by `evaluate`) |
| `train_log__summary.json` | Overall summary — runtime, `best_model` option, winning hyperparameters, best val accuracy / loss / train accuracy |
| `run_info.json` | Completion timestamp and total runtime (seconds) |
| `completed.txt` | Presence of this file marks the run as done and skips re-execution |

#### `evaluate`

One directory is created per evaluation / model / trainer combination under `outputs/evaluations/<evaluation>/<model>__<trainer>/`:

| File | Description |
|---|---|
| `eval_log__seed_*.csv` | Per-seed chunk-level accuracy log — columns: `step` (tokens processed), `acc` (accuracy at that chunk), `acc_cum` (cumulative accuracy), `runtime`, `timelimit`, `state_norm` (L2 norm of the recurrent state after that chunk, summed over the batches) |
| `runinfo__seed_*.json` | Per-seed runtime in seconds, `tokens`, `throughput` (tokens/s over the whole seed, excluding data loading), `stats_time`, `data_time`, `batch_size` |
| `completed__seed_*.txt` | Presence marks that seed as done and its results are loaded from disk on re-runs |
| `eval_log.csv` | Aggregate accuracy across all seeds — columns: `step`, `acc_mean`, `acc_low`, `acc_high` (bootstrap CI bounds) |
| `eval_summary.json` | Overall summary — total runtime, number of seeds used, CI bounds for `acc_min` (minimum per-chunk accuracy) and `acc_cum` (mean accuracy) |
| `run_info.json` | Completion timestamp and total runtime |
| `completed.txt` | Presence marks the full evaluation as done and skips re-execution |

---

## Trainers

Trainers are defined in `meta/meta.trainers.yaml`. The built-in ones are:

| Name | Optimizer | Scheduler | LR | Batch size | Other |
|---|---|---|---|---|---|
| `trainer_default` | Adam | Step | 1e-3 | 64 | |
| `trainer_warmup` | Adam | Step + warmup | 5e-4 | 64 | |
| `trainer_bs8` | Adam | Step | 1e-3 | 8 | |
| `trainer_minmax` | Adam | Step | 1e-3 | 64 | paper, main battery |
| `trainer_minmax_bs8` | Adam | Step | 1e-3 | 8 | paper, main battery |
| `trainer_minmax_compiled_amp` | Adam | Step | 1e-3 | 64 | paper, throughput: compiled (pair with the `__amp_bf16` models) |
| `trainer_groups` | Adam | Cosine | 1e-3 | 64 | paper, word problems |
| `trainer_groups_bs8` | Adam | Cosine | 1e-3 | 8 | paper, word problems |
| `trainer_minmax_longseq` | Adam | Step | 1e-3 | 64 | paper, long-range: compiled, `grad_accum_steps` 2, `max_unroll_steps` 65536 |

Adam has weight decay 1e-4 (`meta/meta.optimizers.yaml`); Step halves the learning rate every 5 epochs; Cosine anneals it to 0 over the epochs.

You can add your own trainer entry in `meta/meta.trainers.yaml` using the same schema. Besides `optimizer`, `scheduler`, `lr` and
`train_batch_size`, a trainer accepts:

| Argument | Default | Description |
|---|---|---|
| `grad_clip` | `null` | Clip the global gradient norm to this value |
| `grad_accum_steps` | 1 | Split each batch into this many micro-batches and accumulate their gradients (same effective batch size and number of optimizer steps) |
| `max_unroll_steps` | `null` | Process training sequences in chunks of this many steps, each under gradient checkpointing, so that memory is bounded by the chunk size (gradients are unchanged). `null`: the longest sequence |
| `compile_model` | `null` | `torch.compile` the model in training and in the evaluations of this trainer's models. `null`: `compile_model` in `meta/meta.config.yaml` |
| `save_checkpoint_epochs` | `null` | Epochs after which to save the weights (0: before training) |
| `best_model` | `null` | Weights kept for each seed: `best_epoch` (the epoch with the best validation accuracy, then loss) or `last_epoch` (the weights at the end of training, which is what runs before v0.2.0 saved). `null`: `best_model` in `meta/meta.config.yaml` |

On an out-of-memory error, training restarts the seed with twice the `grad_accum_steps`, and once micro-batches have a single
sequence, with half the chunk size. Checkpoints are always saved from the uncompiled module; the evaluator also accepts checkpoints
with torch.compile's `_orig_mod.` prefix.

---

## Adding a new benchmark task

1. Implement a `Dataset` subclass in `benchmarks/datasets/`
2. Add dataset configs (train/val/evaluation splits) to `meta/meta.datasets.yaml`
3. Add experiment entries (enabled instances) to `meta/meta.experiments.yaml`
4. Add a template for the new task to `meta/meta.experiments.t.yaml`
5. Add a corresponding evaluation template to `meta/meta.evaluations.t.yaml`

---

## Configuration

Global runtime settings live in `meta/meta.config.yaml`:

| Key | Default | Description |
|---|---|---|
| `ci_min_n` | 3 | Minimum seeds before CI is evaluated |
| `ci_max_n` | 10 | Maximum seeds per hyperparameter combo |
| `ci_confidence` | 0.99 | Bootstrap CI confidence level |
| `ci_accuracy_val` | 0.001 | CI convergence tolerance |
| `max_train_runtime` | 14400 | Per-experiment wall-clock limit (seconds) |
| `max_train_runtime__per_params` | 14400 | Per-hyperparameter-combo wall-clock limit (seconds) |
| `max_train_runtime__per_seed` | 14400 | Per-seed wall-clock limit (seconds) |
| `max_eval_runtime` | 72000 | Per-evaluation wall-clock limit (seconds) |
| `stopper__enabled` | true | Use early stopping |
| `stopper__mode` | min | Whether to minimise (`min`) or maximise (`max`) the tracked metric |
| `stopper__window` | 3 | Rolling window size used to smooth the tracked metric |
| `stopper__patience` | 5 | Early stopping patience (epochs) |
| `stopper__min_delta` | 1e-5 | Minimum improvement to reset the patience counter |
| `stopper__debias` | false | Apply EMA debias correction |
| `num_workers` | 2 | DataLoader worker processes |
| `learning_dynamics_tracker__enabled` | false | Record per-step learning dynamics |
| `ci_seed` | `null` | Seed of the bootstrap RNG; an integer makes every CI a deterministic function of its scores. `null`: unseeded |
| `val_acc_giveup_threshold` | `null` | Stop trying seeds once the upper bound of the val-accuracy CI is below this value. `null`: disabled |
| `best_model` | `best_epoch` | Default of the trainers' `best_model` argument |
| `compile_model` | false | Default of the trainers' `compile_model` argument |
| `eval_max_batch_size` | `null` | Largest batch size tried first in evaluation. `null`: the whole evaluation set |

### Per-experiment protocol

An experiment can override any of these keys for itself with a `protocol:` block in its arguments; its evaluations inherit the
same overrides, and an evaluation can add its own `protocol:` block on top. The number of epochs is the experiment's `epochs`
argument. Unknown keys raise an error.

```yaml
'my_experiment_{N:int}':
  class: Experiment
  args:
    epochs: 100
    # ...
    protocol:
      stopper__enabled: false
      ci_max_n: 5
      max_train_runtime: 43200
      max_train_runtime__per_params: 43200
      max_train_runtime__per_seed: 43200
      eval_max_batch_size: 256
```

---

## Project layout

```
main.py                      Entry point
config/                      Runtime overrides (git-ignored)
  run.yaml                   Optional subset of experiments / evaluations to run
configs/paper/               Runtime configs of the paper's experiments, one per family
docs/                        Reproducing the paper; known issues
scripts/                     Majority-class baseline, event-gap statistics, gradient probe
models/                      Model plugins (auto-discovered at startup)
  vanilla_rnn/               Example: plain RNN baseline
  minmax_rnc/                MinMax RNC wrapper
meta/
  meta.models.yaml           Named model configurations
  meta.experiments.yaml      Enabled experiment instances
  meta.experiments.t.yaml    Experiment templates (tasks + model lists)
  meta.evaluations.yaml      Enabled evaluations
  meta.evaluations.t.yaml    Evaluation templates
  meta.datasets.yaml         Dataset configurations
  meta.trainers.yaml         Trainer configurations
  meta.evaluators.yaml       Evaluator configurations
  meta.config.yaml           Global runtime settings
  meta.tests.yaml            Test groups (enable / disable per group)
benchmarks/
  models/
    model.py                 Abstract Model base class
    models.py                Model factory + register_model()
  datasets/                  Dataset implementations
  trainers/                  Training loop
  experiments/               Experiment orchestration
  evaluations/               Evaluation orchestration
  evaluators/                Evaluator implementations
  runnables/                 CLI entry points (generate, train, evaluate, throughput_report, reproduce)
  metrics/                   Accuracy metrics
  losses/                    Loss functions
  optimizers/                Optimizers
  schedulers/                LR schedulers
  stoppers/                  Early-stopping implementations
  stats/                     Dataset statistics; throughput and norm aggregation
  ldtracker/                 Learning-dynamics tracker
tests/                       pytest test suite
datasets/                    Pre-generated dataset files (zip archives)
outputs/
  experiments/               Training checkpoints and logs
  evaluations/               Evaluation results (CSV + JSON)
```

---

## Testing

The test suite uses [pytest](https://pytest.org). Run it from the project root:

```bash
pytest          # run all enabled groups
pytest -q       # same, less output
pytest tests/test_models.py   # run a single file regardless of group settings
```

Test groups are defined in **`meta/meta.tests.yaml`**. Set `enabled: false` on any group to skip it across all future runs:

```yaml
groups:
  models:
    enabled: true          # flip to false to skip model tests
    description: ...
  datasets:
    enabled: false         # off by default — requires saved datasets on disk
    description: ...
```

| Group | Default | What it covers |
|---|---|---|
| `meta` | on | Template matching, YAML scope loading, parameter substitution |
| `factory` | on | `instantiate`, `get_meta`, `get_required_kwargs`, class registration |
| `models` | on | Model construction, forward shapes, weight-decay groups; MinMax wrapper parameter counts, variants, unrolling, precision |
| `losses` | on | `CrossEntropyLoss` output, masking, `clone` |
| `metrics` | on | `TokenAccuracy` accumulation, reset, edge cases |
| `optimizers` | on | `Adam` / `AdamW` construction and parameter groups |
| `schedulers` | on | `NoneScheduler` / `StepLR` construction and lr-decay |
| `paper` | on | Every paper experiment resolves to the intended models, trainers, datasets and protocol; models build with the paper's parameter counts |
| `permutation_groups` | on | Cayley tables of S_n and A_n |
| `groups` | on | Word-problem labels (brute force on short sequences), cosine scheduler |
| `scripts` | on | Majority-class baseline, event-gap statistics, gradient probe on tiny inputs; tiny mode; `generate` with a runtime config; `reproduce` |
| `trainer` | on | Chunked/checkpointed forward, gradient accumulation, OOM retry, end-to-end CPU fits with in-memory data |
| `evaluator` | on | Compiled-checkpoint loading, history CSVs, end-to-end CPU evaluation with in-memory data |
| `config` | on | Config overrides, `protocol:` precedence, seeded CI |
| `stats` | on | Throughput and norm extraction and aggregation |
| `integration` | on | Synthetic end-to-end training step (no disk data needed) |
| `datasets` | **off** | Dataset generation and DataLoader collation — needs saved datasets |


 ## How to cite

 ```bibtex
 @software{ronca2026minmaxbenchmarks,
   author  = {Alessandro Ronca},
   title   = {{MinMax RNC} Benchmarks},
   year    = {2026},
   url     = {https://github.com/minmaxrnc/benchmarks},
   version = {0.2.0},
 }
 ```

 ## License

This project is licensed under the GNU General Public License v3.0 or later.
See the `LICENSE` file for details.
