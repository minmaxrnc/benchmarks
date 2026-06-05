# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import json
import math
import os
import csv
import glob
import re
import yaml
from collections import defaultdict
from pathlib import Path

from ..definitions import EXPERIMENTS_OUTPUT_DIR, EVALUATIONS_OUTPUT_DIR, OUTPUT_DIR, PROJECT_ROOT

TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _scan_families(root_dir):
    """Return {family: sorted [(instance_num, dir_name), ...]} under root_dir."""
    families = defaultdict(list)
    if not os.path.isdir(root_dir):
        return {}
    for entry in sorted(os.listdir(root_dir)):
        if not os.path.isdir(os.path.join(root_dir, entry)):
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


def _list_eval_model_trainer_dirs(eval_dir):
    """Return sorted model-trainer directory names inside an evaluation dir (completed only)."""
    result = []
    for entry in sorted(os.listdir(eval_dir)):
        path = os.path.join(eval_dir, entry)
        if os.path.isdir(path) and entry.endswith('__trainer_default'):
            if os.path.exists(os.path.join(path, 'completed.txt')):
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


def _load_oscillation_data(mt_path):
    """Return list of (converged, max_drop, n_dips) per seed CSV.

    max_drop: largest drop below 1.0 after first convergence (0.0 if none).
    n_dips:   number of distinct dip events (transitions from >=1.0 to <1.0)
              after first convergence.
    Seeds that never converged are included as (False, 0.0, 0).
    """
    pattern = os.path.join(mt_path, 'train_log__*__seed_*.csv')
    results = []
    for csv_file in sorted(glob.glob(pattern)):
        with open(csv_file, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        accs = [float(r['val_acc']) for r in rows]
        first_conv = next((i for i, a in enumerate(accs) if a >= 1.0), None)
        if first_conv is None:
            results.append((False, 0.0, 0))
        else:
            after = accs[first_conv + 1:]
            max_drop = max(0.0, 1.0 - min(after)) if after else 0.0
            # count transitions from >=1.0 to <1.0 (treat the convergence epoch as prev=1.0)
            n_dips = 0
            prev = 1.0
            for a in after:
                if a < 1.0 and prev >= 1.0:
                    n_dips += 1
                prev = a
            results.append((True, max_drop, n_dips))
    return results


# ---------------------------------------------------------------------------
# Evaluation data loading
# ---------------------------------------------------------------------------

def _load_eval_summary(mt_path):
    """Return (acc_cum, acc_min) dicts with mean/low/high, or None if missing."""
    path = os.path.join(mt_path, 'eval_summary.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        s = json.load(f)
    return s.get('acc_cum'), s.get('acc_min')


def _eval_cell(metric_dict):
    """Format a metric dict {mean, low, high} as $mean [low, high]$ (2 d.p.)."""
    if metric_dict is None:
        return r'\textemdash'
    mean = metric_dict.get('mean')
    low  = metric_dict.get('low')
    high = metric_dict.get('high')
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return r'\textemdash'
    return f'${mean:.2f}\\ [{low:.2f},\\,{high:.2f}]$'


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


_OSC_EPS = 1e-9


def _oscillation_freq_cell(osc_data):
    """n_osc/n_conv [Wilson CI] for seeds that dipped below 1.0 after converging."""
    n_conv = sum(1 for conv, *_ in osc_data if conv)
    if n_conv == 0:
        return r'\textemdash'
    n_osc  = sum(1 for conv, drop, *_ in osc_data if conv and drop > _OSC_EPS)
    lo, hi = _wilson_ci(n_osc, n_conv)
    return f'${n_osc}/{n_conv}\\ [{lo:.2f},\\,{hi:.2f}]$'


def _oscillation_magnitude_cell(osc_data):
    """Mean ± s.d. of the max drop below 1.0 after first convergence, over converged seeds."""
    drops = [drop for conv, drop, *_ in osc_data if conv]
    if not drops:
        return r'\textemdash'
    mean = sum(drops) / len(drops)
    if len(drops) >= 2:
        std = math.sqrt(sum((d - mean) ** 2 for d in drops) / (len(drops) - 1))
        return f'${mean:.3f} \\pm {std:.3f}$'
    return f'${mean:.3f}$'


def _oscillation_count_cell(osc_data):
    """Mean ± s.d. of the number of dip events per converged seed."""
    counts = [n_dips for conv, _, n_dips in osc_data if conv]
    if not counts:
        return r'\textemdash'
    mean = sum(counts) / len(counts)
    if len(counts) >= 2:
        std = math.sqrt(sum((c - mean) ** 2 for c in counts) / (len(counts) - 1))
        return f'${mean:.1f} \\pm {std:.1f}$'
    return f'${mean:.1f}$'


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


def _build_family_oscillation_tables(family, instances, exp_to_dirs, dir_to_label):
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

    freq_cells  = {rk: {} for rk in row_keys}
    mag_cells   = {rk: {} for rk in row_keys}
    count_cells = {rk: {} for rk in row_keys}

    for inst_num, exp_name in instances:
        col       = _col_label(family, inst_num)
        exp_dir   = os.path.join(EXPERIMENTS_OUTPUT_DIR, exp_name)
        rk_to_dir = {_row_key(d, row_params): d for d in exp_to_dirs[exp_name]}

        for rk in row_keys:
            if rk not in rk_to_dir:
                freq_cells[rk][col]  = r'\textemdash'
                mag_cells[rk][col]   = r'\textemdash'
                count_cells[rk][col] = r'\textemdash'
                continue
            mt_path  = os.path.join(exp_dir, rk_to_dir[rk])
            osc_data = _load_oscillation_data(mt_path)
            freq_cells[rk][col]  = _oscillation_freq_cell(osc_data)
            mag_cells[rk][col]   = _oscillation_magnitude_cell(osc_data)
            count_cells[rk][col] = _oscillation_count_cell(osc_data)

    family_title = family.capitalize()

    freq_table = _render_table(
        family, col_labels, row_keys, row_labels, freq_cells,
        caption=(f'Oscillation after first convergence on {family_title} datasets. '
                 r'Cells show $n_{\mathrm{osc}}/n_{\mathrm{conv}}$ (seeds that dipped '
                 r'below perfect accuracy after first converging / seeds that converged) '
                 r'followed by the Wilson 95\% CI.'),
        label=f'tab:{family}_oscillation',
    )
    mag_table = _render_table(
        family, col_labels, row_keys, row_labels, mag_cells,
        caption=(f'Magnitude of oscillation after first convergence on {family_title} datasets. '
                 r'Cells show mean\,$\pm$\,s.d.\ of the maximum drop in validation accuracy '
                 r'below 1.0 after first convergence, averaged over converged seeds.'),
        label=f'tab:{family}_oscillation_magnitude',
    )
    count_table = _render_table(
        family, col_labels, row_keys, row_labels, count_cells,
        caption=(f'Number of oscillation events after first convergence on {family_title} datasets. '
                 r'Cells show mean\,$\pm$\,s.d.\ of the number of distinct dips below perfect '
                 r'accuracy after first convergence, averaged over converged seeds.'),
        label=f'tab:{family}_oscillation_count',
    )
    return freq_table, mag_table, count_table


def _build_eval_family_tables(family, instances, eval_to_dirs, dir_to_label):
    row_params = _find_row_params(eval_to_dirs)

    seen_row_keys = {}
    for inst_num, eval_name in instances:
        for d in eval_to_dirs[eval_name]:
            rk = _row_key(d, row_params)
            if rk not in seen_row_keys:
                seen_row_keys[rk] = d

    row_keys   = sorted(seen_row_keys.keys())
    col_labels = [_col_label(family, inst_num) for inst_num, _ in instances]
    row_labels = {rk: dir_to_label[seen_row_keys[rk]] for rk in row_keys}

    cum_cells = {rk: {} for rk in row_keys}
    min_cells = {rk: {} for rk in row_keys}

    for inst_num, eval_name in instances:
        col      = _col_label(family, inst_num)
        eval_dir = os.path.join(EVALUATIONS_OUTPUT_DIR, eval_name)
        rk_to_dir = {_row_key(d, row_params): d for d in eval_to_dirs[eval_name]}

        for rk in row_keys:
            if rk not in rk_to_dir:
                cum_cells[rk][col] = r'\textemdash'
                min_cells[rk][col] = r'\textemdash'
                continue
            mt_path          = os.path.join(eval_dir, rk_to_dir[rk])
            acc_cum, acc_min = _load_eval_summary(mt_path) or (None, None)
            cum_cells[rk][col] = _eval_cell(acc_cum)
            min_cells[rk][col] = _eval_cell(acc_min)

    # Drop columns where every row is missing
    col_labels = [
        c for c in col_labels
        if any(cum_cells[rk][c] != r'\textemdash' for rk in row_keys)
    ]

    family_title = family.capitalize()

    cum_table = _render_table(
        family, col_labels, row_keys, row_labels, cum_cells,
        caption=(f'Cumulative accuracy on {family_title} evaluation datasets. '
                 r'Cells show mean\,[low,\,high] (bootstrap CI across seeds).'),
        label=f'tab:{family}_eval_cum',
    )
    min_table = _render_table(
        family, col_labels, row_keys, row_labels, min_cells,
        caption=(f'Minimum per-step accuracy on {family_title} evaluation datasets. '
                 r'Cells show mean\,[low,\,high] (bootstrap CI across seeds). '
                 r'\textemdash\ indicates the metric was not defined at some steps.'),
        label=f'tab:{family}_eval_min',
    )
    return cum_table, min_table


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

    # ------------------------------------------------------------------ training
    train_families = _scan_families(EXPERIMENTS_OUTPUT_DIR)

    if not train_families:
        print(f'No experiment results found in {EXPERIMENTS_OUTPUT_DIR}')
    else:
        all_dir_names = []
        family_data   = {}
        for family, instances in train_families.items():
            exp_to_dirs = {}
            for inst_num, exp_name in instances:
                exp_dir = os.path.join(EXPERIMENTS_OUTPUT_DIR, exp_name)
                dirs    = _list_model_trainer_dirs(exp_dir)
                exp_to_dirs[exp_name] = dirs
                all_dir_names.extend(dirs)
            family_data[family] = (instances, exp_to_dirs)

        dir_to_label, label_to_full = _build_name_registry(all_dir_names)

        legend_path = os.path.join(TABLES_DIR, 'legend.tex')
        Path(legend_path).write_text(_render_legend_table(label_to_full) + '\n', encoding='utf-8')
        print(f'  Written: {legend_path}')

        for family, (instances, exp_to_dirs) in family_data.items():
            print(f'\n# Training family: {family}  ({len(instances)} instances)')

            conv_table                       = _build_family_table(family, instances, exp_to_dirs, dir_to_label)
            freq_table, mag_table, cnt_table = _build_family_oscillation_tables(family, instances, exp_to_dirs, dir_to_label)

            for fname, content in [
                (f'{family}.tex',                        conv_table),
                (f'{family}_oscillation.tex',            freq_table),
                (f'{family}_oscillation_magnitude.tex',  mag_table),
                (f'{family}_oscillation_count.tex',      cnt_table),
            ]:
                path = os.path.join(TABLES_DIR, fname)
                Path(path).write_text(content + '\n', encoding='utf-8')
                print(f'  Written: {path}')

    # ------------------------------------------------------------------ evaluation
    eval_families = _scan_families(EVALUATIONS_OUTPUT_DIR)

    if not eval_families:
        print(f'\nNo evaluation results found in {EVALUATIONS_OUTPUT_DIR}')
        return

    all_eval_dir_names = []
    eval_family_data   = {}
    for family, instances in eval_families.items():
        eval_to_dirs = {}
        for inst_num, eval_name in instances:
            eval_dir = os.path.join(EVALUATIONS_OUTPUT_DIR, eval_name)
            dirs     = _list_eval_model_trainer_dirs(eval_dir)
            eval_to_dirs[eval_name] = dirs
            all_eval_dir_names.extend(dirs)
        eval_family_data[family] = (instances, eval_to_dirs)

    eval_dir_to_label, eval_label_to_full = _build_name_registry(all_eval_dir_names)

    eval_legend_path = os.path.join(TABLES_DIR, 'eval_legend.tex')
    Path(eval_legend_path).write_text(_render_legend_table(eval_label_to_full) + '\n', encoding='utf-8')
    print(f'\n  Written: {eval_legend_path}')

    for family, (instances, eval_to_dirs) in eval_family_data.items():
        print(f'\n# Evaluation family: {family}  ({len(instances)} instances)')

        cum_table, min_table = _build_eval_family_tables(family, instances, eval_to_dirs, eval_dir_to_label)

        for fname, content in [
            (f'{family}_eval_cum.tex', cum_table),
            (f'{family}_eval_min.tex', min_table),
        ]:
            path = os.path.join(TABLES_DIR, fname)
            Path(path).write_text(content + '\n', encoding='utf-8')
            print(f'  Written: {path}')
