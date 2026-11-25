# Changelog

## 0.2.0 (2026-11-25)

Configurations and tools to reproduce the MinMax experiments of the paper with the `minmaxrnc`
package: the main battery, the compiled+AMP throughput runs, the word problems on permutation groups
and the long-range Sequences experiment. See `docs/reproducing.md` and `docs/known-issues.md`.

### Added

- `CITATION.cff`, with the paper as the preferred citation (GitHub shows it as "Cite this
  repository").
- **Paper experiments** (MinMax only), enabled by default and also selectable with the runtime configs
  `configs/paper/{main,throughput,groups,longrange}.yaml`:
  - main battery: `main_latching_(1..8)`, `main_sequences_(1..5)` (L = N + 1 layers),
    `main_inductionheads_(0..1)` (two-layer small with batch sizes 64 and 8, two-layer medium,
    three-layer small);
  - throughput: `throughput_{latching,sequences,inductionheads}_(N)`, the batch-size-64 configurations
    of the main battery, compiled and under bf16 AMP;
  - word problems: `groups_s3` (MinMax(L, d), L ∈ {1, 2}, d ∈ {3, 4, 5}, batch sizes 64 and 8) and
    `groups_s4` (MinMax(1, d), d ∈ {4, 5}, 200k training sequences, 100 epochs, early stopping off);
  - long-range Sequences: `longrange_sequences_(1)` (training and validation length 131,072).
- **Models** `minmax_s__l_{L}` (d_model 90, 40 scalar-state neurons) and `minmax_m__l_{L}` (90
  neurons), both with `output_gate=False`, `conv_type='basic'`, `s_r_init='asymmetric'`; their bf16
  AMP variants `__amp_bf16`; `minmax_s__l_{L}__amp_bf16__rec_fp16` (fp16 recurrence); and
  `minmax_groups__l_{L}__d_{D}` (one array of 16 neurons of state dimension D per layer).
- **Trainers** `trainer_minmax`, `trainer_minmax_bs8`, `trainer_minmax_compiled_amp`,
  `trainer_groups`, `trainer_groups_bs8` (cosine schedule) and `trainer_minmax_longseq`.
- **Datasets**: the word problems on S_3 and S_4 (`GroupStateTracking`, with the Cayley tables of
  S_n and A_n in `datasets/utils/permutation_groups.py`) and the long-range Sequences datasets
  `sequences_grad_{train,val,evaluation}_(1)`.
- **Cosine learning-rate schedule** (`cosine`).
- **`python main.py reproduce <family>`**: generate → train → evaluate for one family, with `--steps`,
  `--select` and `--tiny` (a CPU smoke run with tiny datasets, 2 epochs and one seed, under
  `$DATA_DIR/outputs-tiny`); each call writes a record with the run environment to
  `outputs/reproduce/`.
- **Scripts**: `scripts/majority_baseline.py` (majority-class baseline of saved datasets and the gain
  of a trained model over it), `scripts/sequences_gap_stats.py` (event gap and retention span of the
  Sequences datasets), `scripts/gradient_probe.py` (input-to-loss sensitivity across a gap and the
  MinMax state Jacobian).
- **Per-experiment protocol**: a `protocol:` block in an experiment (or evaluation) overrides keys of
  `meta/meta.config.yaml` for it (e.g. early stopping, patience, `ci_max_n`, time limits), with unknown
  keys rejected.
- **Seed-skip rule** (`val_acc_giveup_threshold`, off by default): no further seeds once the upper
  bound of the validation-accuracy CI is below the threshold.
- **Trainer**: `grad_accum_steps`, `max_unroll_steps` (chunked forward with gradient
  checkpointing; same gradients), out-of-memory retry (more micro-batches, then shorter chunks),
  per-trainer `compile_model`, `save_checkpoint_epochs`, and per-epoch throughput, gradient-norm and
  state-norm columns in the training logs.
- **Evaluator**: compiles the model once per evaluation, accepts checkpoints of compiled models
  (`_orig_mod.` prefix), `eval_max_batch_size`, throughput and state-norm logging.
- **`python main.py throughput_report`**: throughput and gradient/state norms aggregated across seeds,
  benchmark families and models.
- **Run environment**: `run_env__*.json` per training seed (Python, torch, CUDA, GPU, `minmaxrnc`
  versions, model arguments, compilation, effective chunking and accumulation).
- `ci_seed`: optional seed of the bootstrap RNG (unseeded by default, as before).
- Tests for the MinMax wrapper, the trainer and evaluator on CPU, the protocol overrides, the
  statistics, the paper configurations, the group tasks and the scripts; GitHub Actions running the
  tests on CPU with Python 3.11–3.13.
- `docs/reproducing.md` and `docs/known-issues.md`.

### Changed

- Requires `minmaxrnc>=0.4.1,<0.5` and `torch>=2.4`.
- The MinMax wrapper follows the minmaxrnc 0.4 API: `scalar_neurons` (with `d_state` as a legacy
  alias), neuron arrays (`neurons: [[d_state, num_neurons], ...]`), `amp_dtype` and
  `recurrence_dtype`. Template parameters can be used inside list-valued arguments.
- **Best model**: the trainer now keeps a copy of the weights of the best epoch. Before, it kept a
  reference to the live parameters, so the saved "best model" was the model at the end of training
  of the best seed. `best_model: last_epoch` (in `meta/meta.config.yaml` or a trainer's arguments)
  restores the previous behaviour.
- Checkpoints are saved from the uncompiled module, so they load into a fresh model whether or not
  training was compiled.
- `python main.py generate` reads the runtime config (`config/run.yaml` or `-c`) like `train` and
  `evaluate`, and generates each dataset for the `ci_max_n` of its experiment's protocol.
- When a model with neuron arrays of state dimension d > 1 is compiled, Inductor's
  `triton.coalesce_tiling_analysis` is turned off, the workaround for a known issue of minmaxrnc
  0.4.0 (Inductor failing in fp32 inference).

### Not included

- The ablation of the paper's Appendix K and a script that generates the paper's tables and figures.
  The og/gc/s_r_init templates of earlier releases are kept unchanged.
