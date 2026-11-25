# Reproducing the paper's MinMax experiments

This release (v0.2.0) contains the configurations and tools to re-run every MinMax experiment of the
paper's appendices on the synthetic benchmarks, with the `minmaxrnc` package (`minmaxrnc>=0.4.1,<0.5`):

| Family | Paper | Config | Experiments |
|---|---|---|---|
| `main` | `tab:experiments-all`, `tab:experiments-latching`, `tab:experiments-sequences`, `tab:experiments-inductionheads`; row "MinMax" of `tab:experiments-throughput` | `configs/paper/main.yaml` | `main_latching_(1..8)`, `main_sequences_(1..5)`, `main_inductionheads_(0..1)` |
| `throughput` | row "MinMax (compiled+AMP)" of `tab:experiments-throughput`, and the claim that compilation and AMP leave the accuracy unchanged | `configs/paper/throughput.yaml` | `throughput_latching_(1..8)`, `throughput_sequences_(1..5)`, `throughput_inductionheads_(0..1)` |
| `groups` | `tab:groups_results`, `fig:groups_eval` (`tab:groups_models` for the parameter counts) | `configs/paper/groups.yaml` | `groups_s3`, `groups_s4` |
| `longrange` | `tab:longrange_results`, `tab:longrange_stats`, the majority-class baseline, the gradient probe | `configs/paper/longrange.yaml` | `longrange_sequences_(1)` |

Not part of this version:

- the ablation of Appendix K (output gate, gated convolution, set/reset initialisation): the
  og/gc/s_r_init templates of earlier releases are still in `meta/meta.experiments.t.yaml`
  (`latching_{N}`, `sequences_{N}`, `inductionheads_{N}`), but they are not the paper's
  configuration of that appendix and have not been re-run;
- a script that turns the outputs into the paper's tables and figures: the numbers have to be read
  from the output files (see [Where the numbers are](#where-the-numbers-are));
- the other models of the paper (Mamba, xLSTM, DeltaNet, DeltaProduct) and the OpenWebText
  experiment.

## Running

```bash
pip install -r requirements.txt

# everything for one family: generate -> train -> evaluate
python main.py reproduce main
python main.py reproduce throughput
python main.py reproduce groups
python main.py reproduce longrange

# one step, or a subset of the experiments
python main.py reproduce groups --steps train evaluate
python main.py reproduce main --select main_inductionheads

# the same, one runnable at a time
python main.py generate -c configs/paper/main.yaml
python main.py train    -c configs/paper/main.yaml
python main.py evaluate -c configs/paper/main.yaml

python main.py reproduce --list
```

Every step skips what is already on disk (saved datasets, completed seeds, completed runs), so an
interrupted run resumes from the last completed seed. Datasets are written to `$DATA_DIR/datasets`
and outputs to `$DATA_DIR/outputs` (`DATA_DIR` defaults to the current directory). Each `reproduce`
call also writes a record of the family, steps, timings and run environment (Python, torch, CUDA,
GPU, `minmaxrnc` versions) to `outputs/reproduce/<family>__<timestamp>.json`; each training seed
writes its own `run_env__*.json` next to its logs.

The paper's experiments are the ones enabled in `meta/meta.experiments.yaml` and
`meta/meta.evaluations.yaml`, so a bare `python main.py train` (or `evaluate`) runs all of them; the
runtime configs select one family at a time by name.

### Smoke run

```bash
python main.py reproduce main --tiny
```

`--tiny` runs the whole pipeline on a CPU in seconds to minutes per family: datasets of a few
sequences of a few dozen steps (evaluation: 256 steps), 2 epochs, one seed, batches of at most 4
sequences, training chunks of at most 16 steps where the trainer sets `max_unroll_steps`, evaluation
chunks of 64 steps (`benchmarks/utils/tiny.py`). Models, trainers' recipes, compilation, precision
and protocol overrides are the real ones. Everything goes to `$DATA_DIR/outputs-tiny`, so tiny
datasets never replace real ones. The numbers are meaningless.

## What each family runs

All runs use the training and evaluation protocol of the paper's Appendix `app:experiments`
(`meta/meta.config.yaml`): seeds 1 to 10 until the 99% bootstrap CI of the validation loss shows
that a new seed would not improve on the best one (at least 3 seeds), 30 epochs with patience-based
early stopping, a 4-hour limit per model–trainer pair; evaluation on 1,000 sequences of 1,024,000
steps per seed, in chunks of 1,024 steps, until the CIs of the step-average and chunk-minimum
accuracies are narrow enough. The weights evaluated are those of the epoch with the best validation
accuracy of the best seed (`best_model: best_epoch`; see [known issues](known-issues.md#best-model)).
Differences are set per experiment with a `protocol:` block (`meta/meta.experiments.t.yaml`).

### Main battery (`main`)

| Benchmark | Model | Trainer | Layers |
|---|---|---|---|
| Latching(4, 8, …, 512): `main_latching_(1..8)` | `minmax_s` | `trainer_minmax` | 2 |
| Sequences(2, 3, 4, 8, 16): `main_sequences_(1..5)` | `minmax_s` | `trainer_minmax` | 2, 3, 4, 5, 6 (N + 1) |
| InductionHeads(16, 30): `main_inductionheads_(0..1)` | `minmax_s`; `minmax_s`; `minmax_m`; `minmax_s` | `trainer_minmax`; `trainer_minmax_bs8`; `trainer_minmax`; `trainer_minmax` | 2; 2; 2; 3 |

- `minmax_s__l_{L}`: `d_model` 90, 40 scalar-state neurons; `minmax_m__l_{L}`: `d_model` 90, 90
  scalar-state neurons (the paper's "medium"). Both have `output_gate=False`, `conv_type='basic'`,
  `s_r_init='asymmetric'` and the package defaults otherwise.
- `trainer_minmax`: Adam with weight decay 1e-4, learning rate 1e-3 halved every 5 epochs, batch
  size 64; `trainer_minmax_bs8`: batch size 8.
- The training throughput of these runs (eager, fp32) is the row "MinMax" of
  `tab:experiments-throughput`.

### Throughput (`throughput`)

The configurations of the main battery with batch size 64 and two layers (Sequences: N + 1 layers;
InductionHeads: small and medium), i.e. 17 model–benchmark pairs, again with `torch.compile`
(`trainer_minmax_compiled_amp`) and bf16 automatic mixed precision. Mixed precision is a setting of
the model (`amp_dtype: bf16` in `minmax_s__l_{L}__amp_bf16`, `minmax_m__l_{L}__amp_bf16`; model-level
autocast, with the logits and the loss in fp32), not of the trainer. After training and evaluation:

```bash
python main.py throughput_report -c configs/paper/throughput.yaml   # compiled + AMP
python main.py throughput_report -c configs/paper/main.yaml         # eager
```

Training throughput counts input tokens, padding included, per second of the training epoch
(excluding data loading and statistics); evaluation throughput counts the tokens of a seed per
second of its evaluation. The report gives mean ± SE across seeds, per benchmark family, and across
families. Throughput depends on the GPU; the paper's numbers are from a single A100.

### Word problems on groups (`groups`)

| Experiment | Models | Trainers | Data | Protocol |
|---|---|---|---|---|
| `groups_s3` | MinMax(L, d) for L ∈ {1, 2}, d ∈ {3, 4, 5} | `trainer_groups` (batch 64), `trainer_groups_bs8` | 20k sequences of length 128; validation 1k × 512 | 30 epochs, patience 3, seed-skip below 50% |
| `groups_s4` | MinMax(1, d) for d ∈ {4, 5} | `trainer_groups` | 200k sequences of length 128; validation 1k × 512 | 100 epochs, early stopping off, seed-skip below 50% |

- `minmax_groups__l_{L}__d_{D}`: `d_model` 90, one array of 16 neurons of state dimension d per
  layer (`neurons: [[d, 16]]`, `scalar_neurons: 0`), `s_r_init='asymmetric'`, the package defaults
  otherwise (`output_gate=True`, `conv_type='basic'`). One-layer parameter counts: 76,166 / 90,710 /
  108,166 on Word(S_3) and 92,330 / 109,786 on Word(S_4) (d = 3 / 4 / 5), as in `tab:groups_models`.
- `trainer_groups`: Adam with weight decay 1e-4, learning rate 1e-3 annealed to 0 with a cosine
  schedule over the epochs.
- Seed-skip: no further seeds once the upper bound of the 99% CI of the validation accuracy is below
  50% (`val_acc_giveup_threshold`). The 4-hour limit is checked before a new seed, so Word(S_4) may
  end with fewer than 3 seeds.
- Labels: γ_t = γ_{t−1} ∘ σ_t = σ_1 ∘ … ∘ σ_t, with (p ∘ q)(i) = p(q(i))
  (`benchmarks/datasets/groupstatetracking.py`, tested in `tests/test_groupstatetracking.py`).
- Evaluation on 1,000 sequences of 1,024,000 steps; every step is labelled.

### Long-range Sequences (`longrange`)

- `longrange_sequences_(1)`: Sequences(2) with training and validation sequences of length 131,072
  and p_set = 2⁻¹⁶ (`sequences_grad_{train,val}_(1)`); evaluation on `sequences_grad_evaluation_(1)`,
  which has the parameters of `sequences_evaluation_(1)`.
- Model `minmax_s__l_(2)__amp_bf16__rec_fp16` (MinMax-S, two layers, bf16 AMP, recurrence in fp16;
  122,582 trainable parameters).
- Trainer `trainer_minmax_longseq`: the `trainer_minmax` recipe, compiled, with 2 micro-batches of 32
  sequences and a forward pass in chunks of 65,536 steps under gradient checkpointing; neither
  changes the gradients. It also saves the weights before training and after epochs 1 and 2, for the
  gradient probe.
- Protocol: at most 5 seeds, patience 3, 12-hour limits (43,200 s), evaluation batches of at most
  256 sequences.

Tools:

```bash
# majority-class baseline per dataset and seed, and the gain of the trained model over it
python scripts/majority_baseline.py --gains --out outputs/longrange_baseline.json
# event gap / retention span (tab:longrange_stats), Monte Carlo on the generator's distribution
python scripts/sequences_gap_stats.py --out outputs/sequences_gap_stats.json
# gradient probe: input-to-loss sensitivity across the gap and the MinMax state Jacobian
python scripts/gradient_probe.py --out outputs/gradient_probe
python scripts/gradient_probe.py --checkpoints init --length 2048 --gaps 1 10 100 1000 \
    --init-seeds 1 --device cpu --out /tmp/probe      # CPU smoke test
```

The baseline reads the saved datasets (`python main.py generate -c configs/paper/longrange.yaml`
first). The gradient probe reads `best_model.pt` and the epoch checkpoints of
`longrange_sequences_(1)`; with `--checkpoints init` it needs no training. Both write data files
(CSV/JSON) only.

## Where the numbers are

Under `outputs/experiments/<experiment>/<model>__<trainer>/` and
`outputs/evaluations/<evaluation>/<model>__<trainer>/` (see the README for every file):

| Paper column | File and field |
|---|---|
| Train (best model) | `train_log__summary.json`: `values_for_best_model.train_acc` |
| Validation (best model) | `train_log__summary.json`: `values_for_best_model.val_acc` |
| Validation, mean and CI across seeds | `train_log__lr_*__bs_*__opt_*__sc_*.json`: `val_acc.{mean,low,high}` |
| Evaluation, step average | `eval_summary.json`: `acc_cum.{mean,low,high}` |
| Evaluation, minimum | `eval_summary.json`: `acc_min.{mean,low,high}` (minimum over chunks of 1,024 steps) |
| `fig:groups_eval` | `eval_log.csv` (mean and CI per chunk across seeds) and `eval_log__seed_*.csv` |
| Throughput | `outputs/throughput_report.json` |
| Time-out (asterisk) | `train_log__lr_*.json`: `timelimit` |

## Compute

From the logs of the original runs; the GPU was not recorded (a single NVIDIA A100 according to the
paper). Dataset generation is not included.

| Family | Training | Evaluation | Notes |
|---|---|---|---|
| `main` (eager) | about 31 h (Latching 10 h, Sequences 8 h, InductionHeads 13 h) | about 30 h (Sequences 17 h: it grows with the number of layers) | original runs with the earlier implementation |
| `throughput` (compiled + AMP) | about 4.6 h | about 4.9 h | 17 runs |
| `groups` | about 9 h (Word(S_3)) + about 11 h (Word(S_4), time-limited) | about 5 h (Word(S_3)); Word(S_4) not run originally | |
| `longrange` | about 14 h (time limit reached after 3 seeds) | not run originally (at least about 1 h, estimated) | |

Disk: each evaluation dataset seed holds 1,000 × 1,024,000 tokens; the long-range training and
validation datasets take about 13 GB for 5 seeds. The 12-hour long-range limit exceeds a typical
hosted-notebook session; runs resume from the last completed seed.

## Versions

- This repository: v0.2.0.
- `minmaxrnc` 0.4.1 (`requirements.txt` allows `>=0.4.1,<0.5`), `torch>=2.4`; the tests run on CPU
  with Python 3.11–3.13 (`.github/workflows/tests.yml`).
- The minmaxrnc 0.4.0 known issue (Inductor failing for neuron arrays of state dimension d > 1 in
  fp32 inference) is worked around automatically when such a model is compiled
  (`benchmarks/utils/compile.py`); the groups runs are not compiled by default.

## Comparison with the paper

The configurations above are the ones that produced the paper's numbers, except for the main
battery, whose original numbers came from an earlier implementation (see
[known issues](known-issues.md#provenance-of-the-main-table-numbers)). Results of re-runs with this
release are to be recorded here.

| Family | Re-run with v0.2.0 | Differences from the paper |
|---|---|---|
| `main` | not yet | |
| `throughput` | not yet | |
| `groups` | not yet | |
| `longrange` | not yet | |

See [known-issues.md](known-issues.md) for the generator quirks, the best-model option, and the
places where the code and the paper's text differ.
