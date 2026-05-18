# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import math
import os
import csv
import glob
import re
import yaml
from collections import defaultdict
from pathlib import Path

from ..definitions import EXPERIMENTS_OUTPUT_DIR, OUTPUT_DIR, PROJECT_ROOT

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
# Parameter parsing  (used for row-key grouping across experiment instances)
# ---------------------------------------------------------------------------

def _parse_params(dirname):
    """Parse model parameters from a model-trainer directory name."""
    model_part = dirname
    idx = model_part.rfind('__trainer_')
    if idx != -1:
        model_part = model_part[:idx]

    params = {}
    for part in model_part.split('__'):
        for prefix in ('og_', 'cv_', 's_', 'l_', 'd_', 'sri_'):
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
    """Return parameter names that vary within at least one experiment."""
    within_varying = set()
    for dirs in exp_to_dirs.values():
        all_keys = set(k for d in dirs for k in _parse_params(d))
        param_vals = defaultdict(set)
        for d in dirs:
            p = _parse_params(d)
            for k in all_keys:
                param_vals[k].add(p.get(k, ''))  # '' when param is absent
        for k, vals in param_vals.items():
            if len(vals) > 1:
                within_varying.add(k)
    return within_varying


def _row_key(dirname, row_params):
    """Hashable row identifier: only the within-varying params."""
    p = _parse_params(dirname)
    return tuple(sorted((k, p.get(k, '')) for k in row_params))


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


def _wilson_ci(n, N):
    """Wilson score interval for proportion n/N at 95% confidence."""
    if N == 0:
        return 0.0, 1.0
    z = 1.96
    p = n / N
    denom = 1 + z * z / N
    center = (p + z * z / (2 * N)) / denom
    spread = z / denom * math.sqrt(p * (1 - p) / N + z * z / (4 * N * N))
    return max(0.0, center - spread), min(1.0, center + spread)


# ---------------------------------------------------------------------------
# Pretty name registry
# ---------------------------------------------------------------------------

def _split_model_trainer(dirname):
    """Split 'model__trainer_X' into ('model', 'trainer_X')."""
    idx = dirname.rfind('__trainer_')
    if idx == -1:
        return dirname, ''
    return dirname[:idx], dirname[idx + 2:]


def _template_to_regex(template):
    """Convert a meta template key to a regex that matches repr (parenthesis-free) dirnames."""
    type_patterns = {
        'int':   r'-?\d+',
        'float': r'-?[\d.]+',
        'str':   r'[^_]+',
        'bool':  r'(?:true|false)',
    }
    result = ''
    i = 0
    while i < len(template):
        if template[i] == '{':
            j = template.index('}', i)
            inside = template[i + 1:j]
            type_name = inside.split(':')[1] if ':' in inside else 'str'
            result += type_patterns.get(type_name, r'.+')
            i = j + 1
        else:
            result += re.escape(template[i])
            i += 1
    return re.compile('^' + result + '$')


def _build_name_registry(all_dir_names):
    """Map every directory name to a short pretty label.

    Returns:
      dir_to_label: {dirname: 'MinMax/Default 1'}
      label_to_full: ordered dict {label: (model_name, trainer_name)}
    """
    models_path  = os.path.join(PROJECT_ROOT, 'meta', 'meta.models.yaml')
    trainers_path = os.path.join(PROJECT_ROOT, 'meta', 'meta.trainers.yaml')

    with open(models_path) as f:
        models_raw = yaml.safe_load(f) or {}
    with open(trainers_path) as f:
        trainers_raw = yaml.safe_load(f) or {}

    model_patterns = [
        (_template_to_regex(key), val.get('pretty_name', key))
        for key, val in models_raw.items()
        if isinstance(val, dict)
    ]

    trainer_pretty = {
        key: val.get('pretty_name', key)
        for key, val in trainers_raw.items()
        if isinstance(val, dict)
    }

    def _model_pretty(model_name):
        for pattern, pretty in model_patterns:
            if pattern.match(model_name):
                return pretty
        return model_name

    sorted_dirs = sorted(set(all_dir_names))

    # First pass: resolve pretty names and find models with multiple trainers
    dir_info = {}
    model_trainers = defaultdict(set)
    for dirname in sorted_dirs:
        model_name, trainer_name = _split_model_trainer(dirname)
        m_pretty = _model_pretty(model_name)
        t_pretty = trainer_pretty.get(trainer_name, trainer_name)
        dir_info[dirname] = (model_name, trainer_name, m_pretty, t_pretty)
        model_trainers[m_pretty].add(t_pretty)

    multi_trainer = {m for m, ts in model_trainers.items() if len(ts) > 1}

    # Second pass: assign labels
    group_counts = defaultdict(int)
    dir_to_label = {}
    label_to_full = {}

    for dirname in sorted_dirs:
        model_name, trainer_name, m_pretty, t_pretty = dir_info[dirname]
        base = f'{m_pretty}/{t_pretty}' if m_pretty in multi_trainer else m_pretty
        group_counts[base] += 1
        label = f'{base} {group_counts[base]}'
        dir_to_label[dirname] = label
        label_to_full[label] = (model_name, trainer_name)

    return dir_to_label, label_to_full


# ---------------------------------------------------------------------------
# Cell rendering
# ---------------------------------------------------------------------------

def _result_cell(seed_results):
    if not seed_results:
        return r'\textemdash'
    n, N   = _count_perfect(seed_results)
    lo, hi = _wilson_ci(n, N)
    return f'${n}/{N}\\ [{lo:.2f},\\,{hi:.2f}]$'


# ---------------------------------------------------------------------------
# Column label
# ---------------------------------------------------------------------------

def _col_label(family, instance_num):
    abbrev = {'latching': 'L', 'sequences': 'S', 'inductionheads': 'I'}
    prefix = abbrev.get(family, family[0].upper())
    return f'{prefix}{instance_num}'


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _latex_tt(s):
    return r'\texttt{' + s.replace('_', r'\_') + '}'


def _render_table(family, col_labels, row_keys, row_labels, cells, caption, label):
    n_cols = len(col_labels)
    col_spec = 'l' + 'c' * n_cols
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')
    lines.append('Model & ' + ' & '.join(col_labels) + r' \\')
    lines.append(r'\midrule')
    for rk in row_keys:
        lines.append(row_labels[rk] + ' & ' + ' & '.join(cells[rk][c] for c in col_labels) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


def _render_legend_table(label_to_full):
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\begin{tabular}{lll}')
    lines.append(r'\toprule')
    lines.append(r'Name & Model & Trainer \\')
    lines.append(r'\midrule')
    for label, (model_name, trainer_name) in label_to_full.items():
        lines.append(label + ' & ' + _latex_tt(model_name) + ' & ' + _latex_tt(trainer_name) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{Model--trainer abbreviations used in the benchmark tables.}')
    lines.append(r'\label{tab:legend}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_family_table(family, instances, exp_to_dirs, dir_to_label):
    row_params = _find_row_params(exp_to_dirs)

    seen_row_keys = {}
    for inst_num, exp_name in instances:
        for d in exp_to_dirs[exp_name]:
            rk = _row_key(d, row_params)
            if rk not in seen_row_keys:
                seen_row_keys[rk] = d

    row_keys   = sorted(seen_row_keys.keys())
    col_labels = [_col_label(family, inst_num) for inst_num, _ in instances]
    row_labels = {rk: dir_to_label[seen_row_keys[rk]] for rk in row_keys}

    cells = {rk: {} for rk in row_keys}

    for inst_num, exp_name in instances:
        col     = _col_label(family, inst_num)
        exp_dir = os.path.join(EXPERIMENTS_OUTPUT_DIR, exp_name)
        rk_to_dir = {_row_key(d, row_params): d for d in exp_to_dirs[exp_name]}

        for rk in row_keys:
            if rk not in rk_to_dir:
                cells[rk][col] = r'\textemdash'
                continue
            mt_path      = os.path.join(exp_dir, rk_to_dir[rk])
            seed_results = _load_seed_results(mt_path)
            cells[rk][col] = _result_cell(seed_results)

    family_title = family.capitalize()
    return _render_table(
        family, col_labels, row_keys, row_labels, cells,
        caption=(f'Convergence to perfect accuracy on {family_title} datasets. '
                 r'Cells show $n/N$ (seeds converged / total) followed by the Wilson 95\% CI.'),
        label=f'tab:{family}',
    )


_MACROS_CONTENT = r"""% Macro definitions for benchmark tables.
% Add \input{macros} to your LaTeX preamble.
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
"""


def _write_macros(tables_dir):
    path = os.path.join(tables_dir, 'macros.tex')
    Path(path).write_text(_MACROS_CONTENT, encoding='utf-8')
    print(f'  Written: {path}')


def run():
    os.makedirs(TABLES_DIR, exist_ok=True)
    _write_macros(TABLES_DIR)
    families = _scan_families()

    if not families:
        print(f'No experiment results found in {EXPERIMENTS_OUTPUT_DIR}')
        return

    # Pre-scan all experiment directories
    all_dir_names = []
    family_data   = {}
    for family, instances in families.items():
        exp_to_dirs = {}
        for inst_num, exp_name in instances:
            exp_dir  = os.path.join(EXPERIMENTS_OUTPUT_DIR, exp_name)
            dirs     = _list_model_trainer_dirs(exp_dir)
            exp_to_dirs[exp_name] = dirs
            all_dir_names.extend(dirs)
        family_data[family] = (instances, exp_to_dirs)

    dir_to_label, label_to_full = _build_name_registry(all_dir_names)

    legend_path = os.path.join(TABLES_DIR, 'legend.tex')
    Path(legend_path).write_text(_render_legend_table(label_to_full) + '\n', encoding='utf-8')
    print(f'  Written: {legend_path}')

    for family, (instances, exp_to_dirs) in family_data.items():
        print(f'\n# Family: {family}  ({len(instances)} dataset instances)')
        table = _build_family_table(family, instances, exp_to_dirs, dir_to_label)

        path = os.path.join(TABLES_DIR, f'{family}.tex')
        Path(path).write_text(table + '\n', encoding='utf-8')
        print(f'  Written: {path}')
