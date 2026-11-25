# Known issues and differences from the paper

Behaviours of the code that a reader of the paper might not expect, kept as they are because the
paper's numbers were produced with them, and places where the paper's text and the configurations
of this repository differ. See [reproducing.md](reproducing.md) for how to run the experiments.

## Provenance of the main-table numbers

The MinMax numbers of the main tables (`tab:experiments-all`, `-latching`, `-sequences`,
`-inductionheads`) and the eager row "MinMax" of `tab:experiments-throughput` were produced with an
earlier implementation of MinMax, from before the `minmaxrnc` package: no output gate, a simple
convolution, PyTorch's default initialisation of the set, reset and output projections (with biases),
no FFN dropout, the neuron input without the convolution, and an initialisation of some projections
that changed during the runs. No configuration of the package reproduces it exactly.

This release reproduces those experiments with the package's recipe instead: `output_gate=False`,
`conv_type='basic'`, `s_r_init='asymmetric'`, package defaults otherwise (`minmax_s__l_{L}`,
`minmax_m__l_{L}`). The compiled+AMP re-runs of the paper (row "MinMax (compiled+AMP)") already used
this recipe, on the 17 batch-size-64 configurations, and reached perfect validation and evaluation
accuracy on all of them; the batch-size-8, three-layer and InductionHeads "medium" configurations
were not run with the package before this release. Results that were close to a threshold in the original
runs (e.g. reached by only some of the seeds) may differ.

## Best model

Before v0.2.0 the trainer kept a reference to the model's `state_dict()` instead of a copy, and the
tensors of a state dict are the live parameters. The "best model" saved for evaluation was therefore
the model at the end of training of the best seed, not the model of its best epoch; the validation
scores reported with it were those of the best epoch. All the paper's evaluations used these
last-epoch weights.

v0.2.0 saves a copy of the best epoch (`best_model: best_epoch`, the default in
`meta/meta.config.yaml`). To reproduce the original behaviour, set `best_model: last_epoch` there or
in a trainer's arguments. Training and validation numbers are not affected; evaluation numbers can
differ when training continued after the best epoch and the last epoch is worse.

## Sequences generator

Kept as is, since the paper's Sequences and long-range numbers were produced with it
(`benchmarks/datasets/sequences.py`):

- **Completion check.** A sequence counts as matched once `n_distinct_seqs` of its events have
  arrived in order (`completion_status[seq_id] == self.n_distinct_seqs`), not all `n_ordered_seq` of
  them. With `n_distinct_seqs = 2` in every configuration, Sequences(n) is in effect Sequences(2) with
  the vocabulary of Sequences(n) (4n + 19 tokens): the later events of a sequence are irrelevant
  tokens.
- **Resets are never emitted.** A token is a reset when `p_set <= r < p_reset`, which never happens
  when `p_reset < p_set`, as in every configuration; so no dataset contains reset tokens, whatever
  its `p_reset`.
- **Lengths exclude the upper bound.** Lengths are drawn with `rng.integers(low, high)`, i.e. in
  [min, max − 1] (also for Latching, InductionHeads and the group tasks; the latter have fixed
  lengths). The Sequences validation datasets use [1025, 2048].

## Sequences: number of layers

Sequences(n) for n = 2, 3, 4, 8, 16 (`main_sequences_(1..5)`) uses L = N + 1 = 2, 3, 4, 5, 6 layers,
as in all the runs. The paper's `tab:concrete_model_sizes` lists L = 8 and L = 16 for n = 8 and n = 16;
its parameter counts for those rows correspond to L = 5 and L = 6. This is a discrepancy between the
paper's text and the configurations that produced its numbers.

## Evaluation metrics

- **Minimum accuracy is per chunk.** The evaluation splits each sequence of 1,024,000 steps into
  1,000 chunks of 1,024 steps; `acc_min` in `eval_summary.json` is the minimum over chunks of the
  accuracy within a chunk (across the sequences of a seed), not the minimum over single steps.
- **InductionHeads.** Only the last step is labelled, so most chunks have no labelled step and
  `acc_min` is NaN; the evaluation's stopping rule is then decided by the step-average accuracy
  `acc_cum` alone.
- **Length.** Evaluation sequences have 1,024,000 = 1,000 · 2¹⁰ steps (the paper writes "2¹⁰ (≈ 1M)").
- **CIs are not deterministic by default.** The bootstrap RNG is unseeded, as in the original runs,
  so the CI bounds, and hence the number of seeds tried, can differ between runs. Set `ci_seed` in
  `meta/meta.config.yaml` to make them deterministic.

## Throughput

- Training throughput counts input tokens, padding included, per second of the training part of an
  epoch (data loading and statistics excluded). Evaluation throughput (`runinfo__seed_*.json`) is the
  number of tokens of a seed (1,000 sequences evaluated in parallel) per second, data loading
  excluded; the evaluation column of `tab:experiments-throughput` is in steps per second of a single
  sequence and includes data loading, so it is about 1,000 times smaller.
- The paper's training figure for MinMax (compiled+AMP), 1,136,600 tokens/s, is not exactly the
  result of any aggregation of the original logs that was tried (the mean of the family means comes
  closest).
- The original compiled+AMP runs used a global `torch.autocast` around training and evaluation. This
  release uses the package's model-level autocast (`amp_dtype: bf16`), which keeps the logits and the
  loss in fp32; the two have not been compared on a GPU.
- Throughput depends on the GPU, which the original runs did not record (v0.2.0 records it in
  `run_env__*.json`).

## Word problems on groups

- **Weight decay.** MinMax is trained with Adam with weight decay 1e-4 (`trainer_groups`), as run.
- **Composition order.** The label at step t is γ_t = γ_{t−1} ∘ σ_t = σ_1 ∘ … ∘ σ_t, where
  (p ∘ q)(i) = p(q(i)). The other order, σ_t ∘ … ∘ σ_1, gives an equivalent task (it is the inverse
  of the running product of the inverse elements) with different labels. Whether the generator of
  DeltaProduct uses the same order has not been checked.
- **Word(S_4).** The original runs reached the 4-hour limit (MinMax(1, 4): 3 seeds; MinMax(1, 5): 2
  seeds, below the usual minimum of 3), and the evaluation of MinMax(1, 4) on sequences of 1,024,000
  steps was not run; `groups_s4` includes it.
- **Compilation.** The groups runs are not compiled. If `compile_model` is turned on, the workaround
  for the minmaxrnc 0.4.0 known issue (Inductor failing for neuron arrays with d > 1 in fp32
  inference) is applied automatically: `torch._inductor.config.triton.coalesce_tiling_analysis =
  False` (`benchmarks/utils/compile.py`).
- DeltaProduct, the other model of `tab:groups_results`, is not part of this repository.

## Long-range Sequences

- **Compiled.** The original MinMax run was compiled with `torch.compile` (its checkpoints carry the
  `_orig_mod.` prefix), and `trainer_minmax_longseq` compiles. Checkpoints are now saved from the
  uncompiled module, and the evaluator also accepts the prefixed ones.
- **Precision.** bf16 AMP with the MinMax recurrence in fp16 (`amp_dtype: bf16`,
  `recurrence_dtype: fp16`); in the original code these were called `precision` and `scan_dtype`.
- **Parameters.** 122,582 trainable parameters, plus 260 non-trainable initial-state entries.
- **Evaluation.** The original run was not evaluated on the 1,024,000-step dataset (its evaluator
  could not load compiled checkpoints); `longrange_sequences_(1)` includes the evaluation.
- **Chunked training.** Gradient accumulation (2 micro-batches) and chunked forward passes with
  gradient checkpointing (65,536 steps) leave the gradients unchanged up to floating-point rounding
  (`tests/test_trainer.py`).

## Other

- `python main.py generate` now reads the runtime config (`config/run.yaml` or `-c`) like `train` and
  `evaluate`, and generates each dataset for the `ci_max_n` of its experiment's `protocol:` block
  (5 seeds for the long-range datasets).
- The og/gc/s_r_init templates `latching_{N}`, `sequences_{N}`, `inductionheads_{N}` of earlier
  releases are kept but disabled by default; they are not the configuration of the paper's ablation (Appendix K), which is
  not part of this release. Outputs of those templates produced before v0.1.3 do not distinguish the
  output-gate and convolution variants: the wrapper passed `output_gate` and `conv_type` where they
  were overridden by the defaults.
