# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Gradient diagnostics of MinMax on the long-range Sequences(2) experiment
(longrange_sequences_(1)), in support of the non-vanishing gradient result.

What it measures (plain autograd)
---------------------------------
Probe sequences are built with the dataset's own vocabulary (Sequences with
n_ordered_seq=2, n_distinct_seqs=2, multiplicity=2, n_other_tokens=10): random
irrelevant ("Other_*") tokens everywhere, except a position-0 token of
sequence s=0 at step A and a position-1 token of s at step B. At B the correct
output switches to Completed_[0], which can only be predicted by remembering A.

B is fixed at the last position (B = T-1; by default T = 131072, the training
length) and A = B - gap. Fixing B keeps the evaluated position, the amount of
history before it and its local context identical across gaps: only where the
A token sits changes. A null probe (no A token) gives the background
sensitivity.

For every probe, one forward and one backward per objective give
    g_t = || d obj_B / d e_t ||_2   for every t <= B,
where e_t is the input-embedding vector at step t (captured with a forward
hook on the model's nn.Embedding, made a leaf with requires_grad) and obj_B is
  * 'loss'  : cross-entropy at B against Completed_[0];
  * 'margin': logit(Completed_[0]) - logit(Completed_[]) at B (it does not
              saturate when a trained model is already confident).
Reported: g_A, g_B, ratio = g_A / g_B (per gap), and a log-spaced profile of
g_t.

Optionally, the state-level Jacobian d h_B / d h_A of the MinMax neuron states
(per layer pair), via the explicit per-layer state of the model
(forward(..., state, return_state=True)): its Frobenius norm (Rademacher /
Hutchinson estimate) and, for same-layer blocks -- which are exactly diagonal
with entries in {0, 1} -- the fraction of state components whose derivative
is exactly 1.

Precision: 'train' = the model as configured (bf16 AMP, fp16 recurrence:
amp_dtype='bf16', recurrence_dtype='fp16'); 'fp32' = the same weights with
amp_dtype=None and recurrence_dtype='fp32'.

Checkpoints: 'init' (freshly initialised models, one per --init-seeds),
'trained' (best_model.pt of the experiment), 'epochs' (the
checkpoint__*__epoch_*.pt files that trainer_minmax_longseq saves before
training and after epochs 1 and 2).

Outputs (in --out): probe_ratio.csv, probe_profile.csv, state_jacobian.csv,
and probe_env.json.

Usage (from the repository root):
    python scripts/gradient_probe.py --data-dir <DATA_DIR> --out <dir>
CPU smoke test (untrained models, short probes):
    python scripts/gradient_probe.py --checkpoints init --length 2048 \\
        --gaps 1 10 100 1000 --init-seeds 1 --device cpu --out /tmp/probe
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

EXPERIMENT = 'longrange_sequences_(1)'
MODEL      = 'minmax_s__l_(2)__amp_bf16__rec_fp16'
TRAINER    = 'trainer_minmax_longseq'
DATASET    = 'sequences_grad_train_(1)'
FP32_ARGS  = {'amp_dtype': None, 'recurrence_dtype': 'fp32'}
TRAIN_DTYPE_LABEL = 'bf16-amp+fp16-recurrence'

DEFAULT_GAPS = [1, 10, 100, 1000, 10000, 30000, 100000]
TRAIN_LENGTH = 131072

_EPOCH_CKPT_RE = re.compile(r'^checkpoint__(?P<params>.+)__seed_(?P<seed>\d+)__epoch_(?P<epoch>\d+)\.pt$')

_booted = False


def bootstrap(data_dir=None):
    """Make the benchmarks package importable (meta paths are relative to the
    repository root) and register the model plugins. DATA_DIR must be set
    before benchmarks.definitions is first imported."""
    global _booted
    if _booted:
        return
    if data_dir is not None:
        os.environ['DATA_DIR'] = str(data_dir)
    os.chdir(REPO_DIR)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    import main
    main.load_models()
    _booted = True


# ---------------------------------------------------------------------------
# Probe sequences
# ---------------------------------------------------------------------------

def build_vocab(dataset=DATASET):
    from benchmarks.datasets.datasets import datasets
    from benchmarks.datasets.sequences import Vocab
    args = datasets.get_property(dataset, 'args')
    return Vocab.build(n_ordered_seq=args['n_ordered_seq'], n_distinct_seqs=args['n_distinct_seqs'],
                       multiplicity=args['multiplicity'], n_other_tokens=args['n_other_tokens'])


def probe_tokens(vocab, T, gap, filler_seed, seq_id=0):
    """(1, T) LongTensor: random Other_* tokens, Set_<s>_0_* at A = B - gap
    (unless gap is None: null probe), Set_<s>_1_* at B = T - 1. Returns
    (x, A, B)."""
    import numpy as np
    import torch
    other_ids = [vocab.token_to_id[t] for t in vocab.other_tokens]
    rng = np.random.default_rng(filler_seed)
    x = torch.as_tensor(rng.choice(other_ids, size=T), dtype=torch.long)
    pos0 = [t for t in vocab.set_tokens
            if vocab.set_token_to_seq_id[t] == seq_id and vocab.set_token_to_pos_id[t] == 0][0]
    pos1 = [t for t in vocab.set_tokens
            if vocab.set_token_to_seq_id[t] == seq_id and vocab.set_token_to_pos_id[t] == 1][0]
    B = T - 1
    A = None
    if gap is not None:
        A = B - gap
        assert A >= 0, f'gap {gap} too large for length {T}'
        x[A] = vocab.token_to_id[pos0]
    x[B] = vocab.token_to_id[pos1]
    return x[None], A, B


def target_ids(vocab, seq_id=0):
    matched = vocab.token_to_id[vocab.seq_idxs_to_output_token[str([seq_id])]]
    none    = vocab.token_to_id[vocab.seq_idxs_to_output_token[str([])]]
    return matched, none


def profile_offsets(B, gaps, n=48):
    import numpy as np
    offs = {0} | {int(round(v)) for v in np.logspace(0, math.log10(max(B, 1)), n)}
    offs |= {g for g in gaps if g is not None}
    return sorted(o for o in offs if 0 <= o <= B)


# ---------------------------------------------------------------------------
# Models and checkpoints
# ---------------------------------------------------------------------------

def iosize(dataset=DATASET):
    from benchmarks.datasets.datasets import datasets
    return datasets.get_iosize(dataset)


def dtype_label(mode):
    return TRAIN_DTYPE_LABEL if mode == 'train' else 'fp32'


def instantiate(model_name, mode):
    """The model as configured ('train') or with fp32 precision ('fp32')."""
    from benchmarks.models.models import models
    if mode == 'train':
        return models.instantiate(model_name, iosize())
    args = models.get_meta(model_name)['args'] | FP32_ARGS
    return models._get_class(model_name)(model_name, iosize(), **args)


def build_model(model_name, mode, state_dict, device):
    model = instantiate(model_name, mode)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def init_state_dict(model_name, seed):
    from benchmarks.utils.seed import seed_everything
    seed_everything(seed)
    model = instantiate(model_name, 'train')
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_state_dict_file(path):
    import torch
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('model', sd) if isinstance(sd, dict) else sd
    return {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v for k, v in sd.items()}


def enumerate_checkpoints(groups, init_seeds, experiment=EXPERIMENT, model_name=MODEL, trainer=TRAINER):
    """Yield (checkpoint label, seed or None, state_dict loader). Missing
    trained checkpoints are reported and skipped."""
    from benchmarks.experiments import utils as eu
    if 'init' in groups:
        for s in init_seeds:
            yield 'init', s, (lambda s=s: init_state_dict(model_name, s))
    out_dir = Path(eu.get_output_dir(experiment, model_name, trainer))
    if 'trained' in groups:
        best = out_dir / 'best_model.pt'
        if best.exists():
            yield 'trained:best', None, (lambda p=best: load_state_dict_file(p))
        else:
            print(f'[skip] trained: no {best}')
    if 'epochs' in groups:
        found = []
        for p in sorted(out_dir.glob('checkpoint__*__seed_*__epoch_*.pt')):
            m = _EPOCH_CKPT_RE.match(p.name)
            if m:
                found.append((int(m['epoch']), int(m['seed']), p))
        if not found:
            print(f'[skip] epochs: no checkpoint__*__epoch_*.pt in {out_dir}')
        for epoch, seed, p in sorted(found):
            yield f'trained:epoch_{epoch}', seed, (lambda p=p: load_state_dict_file(p))


# ---------------------------------------------------------------------------
# Input-to-loss sensitivity
# ---------------------------------------------------------------------------

def _input_embedding(model):
    import torch.nn as nn
    embs = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    if not embs:
        raise RuntimeError('model has no nn.Embedding')
    return embs[0]


def sensitivity(model, x, B, matched_id, none_id):
    """One forward (a single chunk: its gradients equal those of the
    chunked, checkpointed training forward), one backward per objective.
    Returns per-objective (T,) CPU float64 tensors of ||d obj_B / d e_t||
    plus scalar diagnostics."""
    import torch
    import torch.nn.functional as F
    captured = []

    def hook(module, inputs, output):
        leaf = output.detach().requires_grad_(True)
        captured.append(leaf)
        return leaf

    handle = _input_embedding(model).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = model(x, unroll_steps=x.shape[1])
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f'expected one embedding call, got {len(captured)}')
    e = captured[0]
    zB = logits[0, B].float()
    target = torch.tensor([matched_id], device=zB.device)
    objectives = {
        'loss':   F.cross_entropy(zB[None], target),
        'margin': zB[matched_id] - zB[none_id],
    }
    norms = {}
    names = list(objectives)
    for i, name in enumerate(names):
        (g,) = torch.autograd.grad(objectives[name], e, retain_graph=i < len(names) - 1)
        norms[name] = g[0].double().norm(dim=-1).cpu()
    info = {
        'loss':     objectives['loss'].item(),
        'margin':   objectives['margin'].item(),
        'p_target': torch.softmax(zB.detach(), -1)[matched_id].item(),
    }
    return norms, info


# ---------------------------------------------------------------------------
# MinMax state Jacobian
# ---------------------------------------------------------------------------

def _forward_with_state_checkpointed(model, x, state, chunk):
    from torch.utils.checkpoint import checkpoint

    def _step(xc, st):
        return model(xc, st, unroll_steps=xc.shape[1], return_state=True)

    for xc in x.split(chunk, dim=1):
        _, state = checkpoint(_step, xc, state, use_reentrant=False)
    return state


def minmax_state_jacobian(model, x, A, B, K=4, chunk=16384, seed=0):
    """d h_B / d h_A for the scalar-state MinMax neurons (h_A = state after
    step A, h_B = after step B), with the convolution state held fixed.
    Returns a list of row dicts."""
    import torch
    with torch.no_grad():
        _, st = model(x[:, :A + 1], unroll_steps=A + 1, return_state=True)
    L, D = len(st), st[0]['neuron'].shape[-1]
    nb = L * K + L
    leaves = [st[l]['neuron'].detach().expand(nb, D).clone().requires_grad_(True) for l in range(L)]
    state = [{'neuron': leaves[l], 'conv': st[l]['conv'].detach().expand(nb, -1).contiguous()}
             for l in range(L)]
    suffix = x[:, A + 1:B + 1].expand(nb, -1)
    with torch.enable_grad():
        final = _forward_with_state_checkpointed(model, suffix, state, chunk)
        g = torch.Generator().manual_seed(seed)
        obj = 0.0
        for l_out in range(L):
            V = torch.zeros(nb, D)
            V[l_out * K:(l_out + 1) * K] = torch.randint(0, 2, (K, D), generator=g).float() * 2 - 1
            V[L * K + l_out] = 1.0
            obj = obj + (V.to(final[l_out]['neuron'].device) * final[l_out]['neuron'].float()).sum()
        grads = torch.autograd.grad(obj, leaves)
    rows = []
    for l_out in range(L):
        for l_in in range(L):
            vjp = grads[l_in][l_out * K:(l_out + 1) * K].double()
            frob = float(vjp.pow(2).sum(-1).mean().sqrt())
            rows.append({'l_out': l_out, 'l_in': l_in, 'quantity': 'frob_est', 'value': frob})
            if l_in == l_out:
                d = grads[l_in][L * K + l_out].double()
                rows.append({'l_out': l_out, 'l_in': l_in, 'quantity': 'diag_frac_one',
                             'value': float(((d - 1).abs() < 1e-3).double().mean())})
                rows.append({'l_out': l_out, 'l_in': l_in, 'quantity': 'diag_frac_zero',
                             'value': float((d.abs() < 1e-6).double().mean())})
                rows.append({'l_out': l_out, 'l_in': l_in, 'quantity': 'diag_mean', 'value': float(d.mean())})
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _env(device):
    from benchmarks.utils.run_env import environment
    env = environment()
    env['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    env['device'] = str(device)
    return env


class _CsvAppender:
    def __init__(self, path, fields):
        self.path, self.fields = Path(path), fields
        new = not self.path.exists()
        self.f = open(self.path, 'a', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fields)
        if new:
            self.w.writeheader()

    def write(self, rows):
        for r in rows:
            self.w.writerow({k: r.get(k) for k in self.fields})
        self.f.flush()

    def close(self):
        self.f.close()


RATIO_FIELDS = ['checkpoint', 'ckpt_seed', 'filler_seed', 'dtype', 'objective', 'gap', 'A', 'B',
                'g_A', 'g_B', 'ratio', 'n_exact_zero_before_B', 'loss', 'margin', 'p_target']
PROFILE_FIELDS = ['checkpoint', 'ckpt_seed', 'filler_seed', 'dtype', 'objective', 'gap', 't',
                  'offset', 'grad_norm', 'grad_norm_rel']
STATE_FIELDS = ['checkpoint', 'ckpt_seed', 'filler_seed', 'dtype', 'gap', 'l_out', 'l_in',
                'quantity', 'value']


def run(checkpoints=('init', 'trained', 'epochs'), init_seeds=(1, 2, 3), gaps=DEFAULT_GAPS,
        length=TRAIN_LENGTH, dtypes=('train', 'fp32'), filler_seeds=(0, 1), null_probe=True,
        state_jacobian=True, state_K=4, state_chunk=16384, state_max_gap=None,
        out='gradient_probe_out', data_dir=None, device=None,
        experiment=EXPERIMENT, model_name=MODEL, trainer=TRAINER):
    bootstrap(data_dir)
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gaps = [g for g in gaps if g <= length - 1]
    probe_gaps = list(gaps) + ([None] if null_probe else [])
    vocab = build_vocab()
    matched_id, none_id = target_ids(vocab)
    env = _env(device)
    env['args'] = dict(experiment=experiment, model=model_name, trainer=trainer,
                       checkpoints=list(checkpoints), init_seeds=list(init_seeds), gaps=list(gaps),
                       length=length, dtypes=list(dtypes), filler_seeds=list(filler_seeds),
                       null_probe=null_probe, state_jacobian=state_jacobian, state_K=state_K,
                       data_dir=os.environ.get('DATA_DIR'))
    (out / 'probe_env.json').write_text(json.dumps(env, indent=2, default=str))

    ratio_w = _CsvAppender(out / 'probe_ratio.csv', RATIO_FIELDS)
    prof_w = _CsvAppender(out / 'probe_profile.csv', PROFILE_FIELDS)
    state_w = _CsvAppender(out / 'state_jacobian.csv', STATE_FIELDS)
    probes = {(gap, fs): probe_tokens(vocab, length, gap, fs) for gap in probe_gaps for fs in filler_seeds}
    n_rows = 0
    try:
        for ckpt, ckpt_seed, loader in enumerate_checkpoints(checkpoints, init_seeds, experiment,
                                                             model_name, trainer):
            sd = loader()
            for mode in dtypes:
                label = dtype_label(mode)
                t0 = time.time()
                model = build_model(model_name, mode, sd, device)
                base = {'checkpoint': ckpt, 'ckpt_seed': ckpt_seed, 'dtype': label}
                for (gap, fs), (x, A, B) in probes.items():
                    x = x.to(device)
                    norms, info = sensitivity(model, x, B, matched_id, none_id)
                    offs = profile_offsets(B, gaps)
                    for obj, n in norms.items():
                        gB = float(n[B])
                        gA = float(n[A]) if A is not None else None
                        ratio_w.write([{**base, 'filler_seed': fs, 'objective': obj,
                                        'gap': gap if gap is not None else 'none', 'A': A, 'B': B,
                                        'g_A': gA, 'g_B': gB,
                                        'ratio': (gA / gB if (gA is not None and gB > 0) else None),
                                        'n_exact_zero_before_B': int((n[:B] == 0).sum()), **info}])
                        prof_w.write([{**base, 'filler_seed': fs, 'objective': obj,
                                       'gap': gap if gap is not None else 'none', 't': B - o, 'offset': o,
                                       'grad_norm': float(n[B - o]),
                                       'grad_norm_rel': (float(n[B - o]) / gB if gB > 0 else None)}
                                      for o in offs])
                        n_rows += 1
                    if (state_jacobian and gap is not None and fs == filler_seeds[0]
                            and (state_max_gap is None or gap <= state_max_gap)):
                        try:
                            rows = minmax_state_jacobian(model, x, A, B, K=state_K, chunk=state_chunk)
                        except Exception as e:  # e.g. out of memory at the largest gaps
                            print(f'[fail] state jacobian {ckpt} {label} gap={gap}: {type(e).__name__}: {e}')
                        else:
                            state_w.write([{**base, 'filler_seed': fs, 'gap': gap, **r} for r in rows])
                print(f'{ckpt:18s} seed={ckpt_seed} {label:26s} {time.time() - t0:6.1f}s')
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    finally:
        ratio_w.close(); prof_w.close(); state_w.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoints', nargs='+', default=['init', 'trained', 'epochs'],
                    choices=['init', 'trained', 'epochs'])
    ap.add_argument('--init-seeds', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--gaps', nargs='+', type=int, default=DEFAULT_GAPS)
    ap.add_argument('--length', type=int, default=TRAIN_LENGTH)
    ap.add_argument('--dtypes', nargs='+', default=['train', 'fp32'], choices=['train', 'fp32'])
    ap.add_argument('--filler-seeds', nargs='+', type=int, default=[0, 1])
    ap.add_argument('--no-null-probe', action='store_true')
    ap.add_argument('--no-state-jacobian', action='store_true')
    ap.add_argument('--state-K', type=int, default=4)
    ap.add_argument('--state-chunk', type=int, default=16384)
    ap.add_argument('--state-max-gap', type=int, default=None)
    ap.add_argument('--experiment', default=EXPERIMENT)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--trainer', default=TRAINER)
    ap.add_argument('--out', default='gradient_probe_out')
    ap.add_argument('--data-dir', default=None,
                    help='DATA_DIR holding outputs/experiments (default: env DATA_DIR or .)')
    ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()  # resolve before bootstrap() changes the cwd
    data_dir = Path(a.data_dir).resolve() if a.data_dir else None
    run(checkpoints=a.checkpoints, init_seeds=a.init_seeds, gaps=a.gaps, length=a.length,
        dtypes=a.dtypes, filler_seeds=a.filler_seeds, null_probe=not a.no_null_probe,
        state_jacobian=not a.no_state_jacobian, state_K=a.state_K, state_chunk=a.state_chunk,
        state_max_gap=a.state_max_gap, out=out, data_dir=data_dir, device=a.device,
        experiment=a.experiment, model_name=a.model, trainer=a.trainer)


if __name__ == '__main__':
    main()
