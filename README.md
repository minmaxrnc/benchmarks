# Benchmarks

A model-agnostic benchmarking framework for sequence models. It provides three synthetic tasks that stress-test long-range memory and context binding:

- **Latching** — remember and reproduce a value across a long sequence
- **Sequences** — match multiple ordered sub-sequences
- **Induction Heads** — retrieve a token that appeared after a repeated context (as in the [Mamba paper](https://arxiv.org/pdf/2312.00752))

Training uses bootstrap confidence intervals to automatically determine how many random seeds are needed, making comparisons statistically rigorous.

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

# List available runnables
python main.py ls
```

### Selecting a subset at runtime

By default `train` and `evaluate` run every entry that has `enabled: true` in the corresponding meta file. To override this without editing the meta files, create a `config/run.yaml` file at the project root:

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
```

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

Up to `ci_max_n` seeds are generated (default 10). Training and validation splits use short sequences; evaluation splits use long sequences.

#### `train`

One directory is created per experiment / model / trainer combination under `outputs/experiments/<experiment>/<model>__<trainer>/`:

| File | Description |
|---|---|
| `train_log__lr_*__bs_*__opt_*__sc_*__seed_*.csv` | Per-seed epoch log — columns: `epoch`, `lr_*`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `runtime`, `timelimit`, `earlystop` |
| `train_log__lr_*__bs_*__opt_*__sc_*.json` | Per-hyperparameter-combo summary — best and CI-bounded val accuracy / loss, total time, timelimit flag |
| `best_model__lr_*__bs_*__opt_*__sc_*.pt` | Best model checkpoint for that hyperparameter combo |
| `best_model.pt` | Best model overall (loaded by `evaluate`) |
| `train_log__summary.json` | Overall summary — runtime, winning hyperparameters, best val accuracy / loss / train accuracy |
| `run_info.json` | Completion timestamp and total runtime (seconds) |
| `completed.txt` | Presence of this file marks the run as done and skips re-execution |

#### `evaluate`

One directory is created per evaluation / model / trainer combination under `outputs/evaluations/<evaluation>/<model>__<trainer>/`:

| File | Description |
|---|---|
| `eval_log__seed_*.csv` | Per-seed chunk-level accuracy log — columns: `step` (tokens processed), `acc` (accuracy at that chunk), `acc_cum` (cumulative accuracy), `runtime`, `timelimit` |
| `runinfo__seed_*.json` | Per-seed runtime in seconds |
| `completed__seed_*.txt` | Presence marks that seed as done and its results are loaded from disk on re-runs |
| `eval_log.csv` | Aggregate accuracy across all seeds — columns: `step`, `acc_mean`, `acc_low`, `acc_high` (bootstrap CI bounds) |
| `eval_summary.json` | Overall summary — total runtime, number of seeds used, CI bounds for `acc_min` (minimum per-chunk accuracy) and `acc_cum` (mean accuracy) |
| `run_info.json` | Completion timestamp and total runtime |
| `completed.txt` | Presence marks the full evaluation as done and skips re-execution |

---

## Trainers

Trainers are defined in `meta/meta.trainers.yaml`. The built-in one is:

| Name | Optimizer | Scheduler | LR | Batch size |
|---|---|---|---|---|
| `trainer_default` | Adam | Step | 1e-3 | 64 |

You can add your own trainer entry in `meta/meta.trainers.yaml` using the same schema.

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

---

## Project layout

```
main.py                      Entry point
config/                      Runtime overrides (git-ignored)
  run.yaml                   Optional subset of experiments / evaluations to run
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
  runnables/                 CLI entry points (generate, train, evaluate)
  metrics/                   Accuracy metrics
  losses/                    Loss functions
  optimizers/                Optimizers
  schedulers/                LR schedulers
  stoppers/                  Early-stopping implementations
  stats/                     Dataset statistics utilities
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
| `models` | on | Model construction, forward shapes, weight-decay groups |
| `losses` | on | `CrossEntropyLoss` output, masking, `clone` |
| `metrics` | on | `TokenAccuracy` accumulation, reset, edge cases |
| `optimizers` | on | `Adam` / `AdamW` construction and parameter groups |
| `schedulers` | on | `NoneScheduler` / `StepLR` construction and lr-decay |
| `integration` | on | Synthetic end-to-end training step (no disk data needed) |
| `datasets` | **off** | Dataset generation and DataLoader collation — needs saved datasets |


 ## How to cite

 ```bibtex
 @software{ronca2026minmaxbenchmarks,
   author  = {Alessandro Ronca},
   title   = {Benchmarks},
   year    = {2026},
   url     = {https://github.com/minmaxrnc/benchmarks},
   version = {0.1.2},
 }
 ```

 ## License

This project is licensed under the GNU General Public License v3.0 or later.
See the `LICENSE` file for details.
