# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import os
import csv
import glob
from collections import defaultdict
from pathlib import Path

from ..definitions import EXPERIMENTS_OUTPUT_DIR, OUTPUT_DIR
from ..utils.ci import ConfidenceIntervals
from ..utils import config

TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _scan_families():
    """Return {family: sorted [(instance_num, exp_dir_name), ...]}."""
    families = defaultdict(list)
    if not os.path.isdir(EXPERIMENTS_OUTPUT_DIR):
        return {}
    for entry in sorted(os.listdir(EXPERIMENTS_OUTPUT_DIR)):
        if not os.path.isdir(os.path.join(EXPERIMENTS_OUTPUT_DIR, entry)):
            continue
        idx = entry.rfind('_')
        if idx == -1:
            continue
        prefix, suffix = entry[:idx], entry[idx + 1:]
        if suffix.lstrip('-').isdigit():
            families[prefix].append((int(suffix), entry))
    return {f: sorted(v) for f, v in sorted(families.items())}


def _list_model_trainer_dirs(exp_dir):
    """Return sorted model-trainer directory names inside an experiment dir."""
    result = []
    for entry in sorted(os.listdir(exp_dir)):
        if os.path.isdir(os.path.join(exp_dir, entry)) and entry.endswith('__trainer_default'):
            result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Parameter parsing
# ---------------------------------------------------------------------------

def _parse_params(dirname):
    """Parse model parameters from a model-trainer directory name.

    Returns a dict with keys such as d, l, s, og, cv.
    Strips the trailing __trainer_<name> suffix first.
    """
    model_part = dirname
    idx = model_part.rfind('__trainer_')
    if idx != -1:
        model_part = model_part[:idx]

    params = {}
    for part in model_part.split('__'):
        for prefix in ('og_', 'cv_', 's_', 'l_', 'd_'):
            if part.startswith(prefix):
                key = prefix.rstrip('_')
                raw = part[len(prefix):]
                try:
                    params[key] = int(raw)
                except ValueError:
                    params[key] = raw
                break
    return params


def _find_row_params(exp_to_dirs):
    """Return the set of parameter names that vary *within* at least one experiment.

    These are the params that distinguish model variants (rows) rather than
    dataset instances (columns).
    """
    within_varying = set()
    for dirs in exp_to_dirs.values():
        param_vals = defaultdict(set)
        for d in dirs:
            for k, v in _parse_params(d).items():
                param_vals[k].add(v)
        for k, vals in param_vals.items():
            if len(vals) > 1:
                within_varying.add(k)
    return within_varying


def _row_key(dirname, row_params):
    """Hashable row identifier: only the within-varying params."""
    p = _parse_params(dirname)
    return tuple(sorted((k, p[k]) for k in row_params if k in p))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_seed_results(mt_path):
    """Return list of best val_acc (float) for each seed CSV found."""
    pattern = os.path.join(mt_path, 'train_log__*__seed_*.csv')
    results = []
    for csv_file in sorted(glob.glob(pattern)):
        with open(csv_file, newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            results.append(max(float(r['val_acc']) for r in rows))
    return results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _count_perfect(seed_results):
    """Return (n_perfect, n_total)."""
    return sum(1 for v in seed_results if v >= 1.0), len(seed_results)


def _bootstrap_ci(seed_results):
    """Bootstrap CI for convergence probability using binary [0,1] scores."""
    binary = [1.0 if v >= 1.0 else 0.0 for v in seed_results]
    delta = 1.0 - config.get('ci_confidence')
    _, low, high = ConfidenceIntervals().mean_bootstrap(binary, delta)
    return low, high


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------

_PARAM_ORDER = ('og', 'cv', 's', 'l', 'd')

_PARAM_LABELS = {
    'og': {True: r'\cmark', 'true': r'\cmark', False: r'\xmark', 'false': r'\xmark'},
    'cv': {'basic': 'basic', 'gated': 'gated'},
}


def _fmt_param(key, val):
    if key == 'og':
        mapped = _PARAM_LABELS['og'].get(val, str(val))
        return f'OG={mapped}'
    if key == 'cv':
        return _PARAM_LABELS['cv'].get(val, str(val))
    return f'${key}\\!=\\!{val}$'


def _row_label(row_key_tuple, row_params):
    """Human-readable LaTeX label for a model variant row."""
    d = dict(row_key_tuple)
    parts = []
    for k in _PARAM_ORDER:
        if k in row_params and k in d:
            parts.append(_fmt_param(k, d[k]))
    return ', '.join(parts) if parts else str(row_key_tuple)


def _col_label(family, instance_num):
    abbrev = {'latching': 'L', 'sequences': 'S', 'inductionheads': 'I'}
    prefix = abbrev.get(family, family[0].upper())
    return f'{prefix}{instance_num}'


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _convergence_cell(n, N):
    if N == 0:
        return r'\textemdash'
    if n == N:
        return r'\checkmark'
    return f'{n}/{N}'


def _ci_cell(seed_results):
    if not seed_results:
        return r'\textemdash'
    lo, hi = _bootstrap_ci(seed_results)
    return f'$[{lo:.2f},\\,{hi:.2f}]$'


def _render_table(family, col_labels, row_keys, row_labels, cells, caption, label):
    n_cols = len(col_labels)
    col_spec = 'l' + 'c' * n_cols
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')
    header = 'Model & ' + ' & '.join(col_labels) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    for rk in row_keys:
        row_str = row_labels[rk] + ' & ' + ' & '.join(cells[rk][c] for c in col_labels) + r' \\'
        lines.append(row_str)
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_family_tables(family, instances):
    exp_to_dirs = {}
    for inst_num, exp_name in instances:
        exp_dir = os.path.join(EXPERIMENTS_OUTPUT_DIR, exp_name)
        exp_to_dirs[exp_name] = _list_model_trainer_dirs(exp_dir)

    row_params = _find_row_params(exp_to_dirs)

    # Collect all row keys in a stable order (first seen, sorted)
    seen_row_keys = {}
    for inst_num, exp_name in instances:
        for d in exp_to_dirs[exp_name]:
            rk = _row_key(d, row_params)
            if rk not in seen_row_keys:
                seen_row_keys[rk] = d

    # Sort row keys by their parameter values
    def _sort_key(rk):
        d = dict(rk)
        return tuple(str(d.get(k, '')) for k in _PARAM_ORDER)

    row_keys = sorted(seen_row_keys.keys(), key=_sort_key)

    col_labels = [_col_label(family, inst_num) for inst_num, _ in instances]

    # Build label maps
    row_labels = {rk: _row_label(rk, row_params) for rk in row_keys}

    # Load cell data
    conv_cells = {rk: {} for rk in row_keys}
    ci_cells = {rk: {} for rk in row_keys}

    for inst_num, exp_name in instances:
        col = _col_label(family, inst_num)
        exp_dir = os.path.join(EXPERIMENTS_OUTPUT_DIR, exp_name)
        # Build a lookup: row_key -> model_trainer_dir for this instance
        rk_to_dir = {}
        for d in exp_to_dirs[exp_name]:
            rk = _row_key(d, row_params)
            rk_to_dir[rk] = d

        for rk in row_keys:
            if rk not in rk_to_dir:
                conv_cells[rk][col] = r'\textemdash'
                ci_cells[rk][col] = r'\textemdash'
                continue
            mt_path = os.path.join(exp_dir, rk_to_dir[rk])
            seed_results = _load_seed_results(mt_path)
            n, N = _count_perfect(seed_results)
            conv_cells[rk][col] = _convergence_cell(n, N)
            ci_cells[rk][col] = _ci_cell(seed_results)

    family_title = family.capitalize()

    conv_table = _render_table(
        family, col_labels, row_keys, row_labels, conv_cells,
        caption=f'Convergence to perfect accuracy on {family_title} datasets. '
                r'\checkmark: all seeds converged; $n/N$: only $n$ out of $N$ seeds converged.',
        label=f'tab:{family}_convergence',
    )
    ci_table = _render_table(
        family, col_labels, row_keys, row_labels, ci_cells,
        caption=f'Bootstrap confidence intervals for convergence probability on '
                f'{family_title} datasets (confidence level: {config.get("ci_confidence")}).',
        label=f'tab:{family}_ci',
    )
    return conv_table, ci_table


def run():
    os.makedirs(TABLES_DIR, exist_ok=True)
    families = _scan_families()

    if not families:
        print(f'No experiment results found in {EXPERIMENTS_OUTPUT_DIR}')
        return

    for family, instances in families.items():
        print(f'\n# Family: {family}  ({len(instances)} dataset instances)')
        conv_table, ci_table = _build_family_tables(family, instances)

        conv_path = os.path.join(TABLES_DIR, f'{family}_convergence.tex')
        ci_path = os.path.join(TABLES_DIR, f'{family}_ci.tex')

        Path(conv_path).write_text(conv_table + '\n', encoding='utf-8')
        Path(ci_path).write_text(ci_table + '\n', encoding='utf-8')

        print(f'  Written: {conv_path}')
        print(f'  Written: {ci_path}')
