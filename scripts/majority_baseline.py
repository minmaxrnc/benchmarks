# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Majority-class baseline of saved datasets (Appendix
subsec:longrange-baseline), and the gain of a trained model over it.

The baseline predicts, at every labelled step, the most frequent label of the
dataset; its accuracy is the fraction of labelled steps with that label. On
the Sequences datasets the most frequent label is normally the empty set (no
sequence matched, output token `Completed_[]`), whose fraction is reported
separately. The gain of a model with accuracy acc over a baseline with
accuracy base is (acc - base) / (1 - base).

The statistics are computed per dataset and seed from the datasets saved by
`python main.py generate` (under $DATA_DIR/datasets). With --gains, the best
validation accuracy of each seed of a trained experiment is read from its
train_log CSVs and compared with the baseline of that seed's validation
dataset.

Usage (from the repository root):
    python scripts/majority_baseline.py                      # long-range datasets, seeds 1..5
    python scripts/majority_baseline.py --gains              # also the gains of longrange_sequences_(1)
    python scripts/majority_baseline.py --datasets 'sequences_val_(1)' --seeds 1 2 3
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATASETS = ['sequences_grad_train_(1)', 'sequences_grad_val_(1)']
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
EXPERIMENT = 'longrange_sequences_(1)'
EMPTY_SET_TOKEN = 'Completed_[]'


def bootstrap(data_dir=None):
    if data_dir is not None:
        os.environ['DATA_DIR'] = str(data_dir)
    os.chdir(REPO_DIR)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    import main
    main.load_models()


def label_counts(samples, vocab_size, max_samples=None):
    """Counts of the labels at the labelled (mask=True) steps of the samples
    (dicts with 'outputs' and 'mask' tensors), as a length-vocab_size list."""
    import torch
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for i, sample in enumerate(samples):
        if max_samples is not None and i >= max_samples:
            break
        labels = sample['outputs'].long()[sample['mask'].bool()]
        counts += torch.bincount(labels, minlength=vocab_size)
    return counts.tolist()


def baseline_from_counts(counts, id_to_token=None, empty_id=None):
    """Majority-class accuracy and label, and the empty-set fraction."""
    n = sum(counts)
    if n == 0:
        raise ValueError('no labelled steps')
    majority_id = max(range(len(counts)), key=lambda i: counts[i])
    result = {
        'n_labelled_steps': n,
        'majority_label_id': majority_id,
        'majority_label': id_to_token[majority_id] if id_to_token is not None else None,
        'baseline_accuracy': counts[majority_id] / n,
    }
    if empty_id is not None:
        result['empty_set_fraction'] = counts[empty_id] / n
    return result


def gain(acc, base):
    """(acc - base) / (1 - base): 1 for a perfect model, 0 for the baseline."""
    if base >= 1.0:
        return float('nan')
    return (acc - base) / (1.0 - base)


def dataset_baseline(name, seed, max_samples=None):
    from benchmarks.datasets.datasets import datasets
    dataset = datasets.instantiate(name, seed=seed)
    vocab = getattr(dataset, 'vocab', None)
    id_to_token = getattr(vocab, 'id_to_token', None)
    token_to_id = getattr(vocab, 'token_to_id', None) or {}
    empty_id = token_to_id.get(EMPTY_SET_TOKEN)
    samples = (dataset[i] for i in range(len(dataset)))
    counts = label_counts(samples, dataset.vocab_size, max_samples=max_samples)
    return {'dataset': name, 'seed': seed, 'n_sequences': len(dataset) if max_samples is None
            else min(len(dataset), max_samples), **baseline_from_counts(counts, id_to_token, empty_id)}


def best_val_acc_per_seed(experiment_dir):
    """{seed: max val_acc over the epochs} from train_log__*__seed_<s>.csv."""
    import re
    result = {}
    for path in glob.glob(os.path.join(experiment_dir, 'train_log__*__seed_*.csv')):
        seed = int(re.search(r'__seed_(\d+)\.csv$', path).group(1))
        with open(path, newline='') as f:
            accs = [float(r['val_acc']) for r in csv.DictReader(f) if r.get('val_acc') not in (None, '')]
        if accs:
            result[seed] = max(result.get(seed, float('-inf')), max(accs))
    return result


def experiment_gains(experiment, baselines, max_samples=None):
    """Per model/trainer pair and seed: best val_acc, val baseline, gain."""
    from benchmarks.experiments.experiments import experiments
    from benchmarks.experiments import utils as eu
    val_dataset = experiments.get_meta(experiment)['args']['val_dataset']
    by_seed = {b['seed']: b for b in baselines if b['dataset'] == val_dataset}
    rows = []
    for mt in experiments.get_models_trainers(experiment):
        out_dir = eu.get_output_dir(experiment, mt['model'], mt['trainer'])
        for seed, acc in sorted(best_val_acc_per_seed(out_dir).items()):
            base = by_seed.get(seed)
            if base is None:
                base = dataset_baseline(val_dataset, seed, max_samples)
            rows.append({'model': mt['model'], 'trainer': mt['trainer'], 'seed': seed,
                         'val_acc': acc, 'baseline_accuracy': base['baseline_accuracy'],
                         'gain': gain(acc, base['baseline_accuracy'])})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--max-samples', type=int, default=None, help='only the first N sequences of each dataset')
    ap.add_argument('--gains', action='store_true', help=f'gains of the trained experiment (--experiment)')
    ap.add_argument('--experiment', default=EXPERIMENT)
    ap.add_argument('--data-dir', default=None, help='DATA_DIR holding datasets/ and outputs/ (default: env DATA_DIR or .)')
    ap.add_argument('--out', default=None, help='JSON output path (default: print only)')
    a = ap.parse_args(argv)
    out = Path(a.out).resolve() if a.out else None
    bootstrap(Path(a.data_dir).resolve() if a.data_dir else None)

    baselines = []
    for name in a.datasets:
        for seed in a.seeds:
            try:
                b = dataset_baseline(name, seed, a.max_samples)
            except Exception as e:
                print(f'[skip] {name} seed {seed}: {e}')
                continue
            baselines.append(b)
            extra = f", empty set {b['empty_set_fraction']:.4%}" if 'empty_set_fraction' in b else ''
            print(f"{name} seed {seed}: baseline {b['baseline_accuracy']:.4%} "
                  f"({b['majority_label']}){extra}, {b['n_labelled_steps']:,} steps")
    report = {'baselines': baselines}
    if a.gains:
        report['gains'] = experiment_gains(a.experiment, baselines, a.max_samples)
        for r in report['gains']:
            print(f"{r['model']} [{r['trainer']}] seed {r['seed']}: val_acc {r['val_acc']:.4%}, "
                  f"baseline {r['baseline_accuracy']:.4%}, gain {r['gain']:.3%}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f'written to {out}')
    return report


if __name__ == '__main__':
    main()
