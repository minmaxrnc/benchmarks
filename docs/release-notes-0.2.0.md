# v0.2.0

This release makes it possible to re-run the MinMax experiments of the paper with the
[`minmaxrnc`](https://github.com/minmaxrnc/model) package (`minmaxrnc>=0.4.1,<0.5`):

- the **main battery**: Latching(4–512), Sequences(2, 3, 4, 8, 16) and InductionHeads(16, 30);
- the **throughput** runs with `torch.compile` and bf16 mixed precision;
- the **word problems** on S_3 and S_4;
- the **long-range Sequences** experiment, with scripts for its majority-class baseline, its event-gap
  statistics and the gradient probe.

```bash
pip install -r requirements.txt
python main.py reproduce main            # generate -> train -> evaluate
python main.py reproduce groups --tiny   # CPU smoke run
```

Each family has a runtime config in `configs/paper/`. `docs/reproducing.md` maps the
experiments to the paper's tables and figures, describes the configurations and gives compute
estimates; `docs/known-issues.md` lists the behaviours kept as run and the
differences from the paper's text.

## Highlights

- `python main.py reproduce <family>` with `--tiny`, `--steps` and `--select`.
- New models, trainers and datasets for the paper's configurations; a cosine learning-rate schedule;
  the word-problem task on permutation groups.
- Per-experiment `protocol:` overrides of the training and evaluation protocol, and an optional
  seed-skip rule.
- Long-sequence training: gradient accumulation, chunked forward passes with gradient checkpointing,
  out-of-memory retry, per-trainer `torch.compile`, epoch checkpoints.
- Throughput, gradient-norm and state-norm logging, and `python main.py throughput_report`.
- The run environment (versions, GPU, compilation) is recorded with every training seed.
- The MinMax wrapper supports the minmaxrnc 0.4 API (neuron arrays, `amp_dtype`,
  `recurrence_dtype`).

## Changes in behaviour

- **Best model.** The weights saved for evaluation are now those of the best epoch of the best seed.
  Before, they were those at the end of training of that seed; set `best_model: last_epoch` to
  reproduce that.
- `generate` honours the runtime config (`config/run.yaml` or `-c`) and the experiments' `protocol:`
  blocks.
- Requires `minmaxrnc>=0.4.1,<0.5` and `torch>=2.4`.

## Notes

- The paper's original main-table numbers were produced with an earlier implementation of MinMax;
  this release reproduces those experiments with the package's recipe (no output gate, basic
  convolution, asymmetric set/reset initialisation).
- The Sequences generator is unchanged (a sequence is matched after its first two events, resets are
  never emitted, length ranges exclude their upper bound), since the paper's numbers were produced
  with it.
- The ablation of Appendix K and a script that generates the paper's tables are not part of this
  release.

See `CHANGELOG.md` for the full list of changes.
