# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

"""Reproduce one family of the paper's experiments: generate -> train ->
evaluate, with the runtime config configs/paper/<family>.yaml.

    python main.py reproduce main                # the main battery
    python main.py reproduce groups --steps train evaluate
    python main.py reproduce longrange --tiny    # CPU smoke run
    python main.py reproduce --list

Every step skips what is already on disk (saved datasets, completed seeds and
runs), so an interrupted run resumes where it stopped. With --tiny, datasets,
epochs and seeds are shrunk (benchmarks/utils/tiny.py) and everything is
written under $DATA_DIR/outputs-tiny instead of $DATA_DIR (main.py sets this
up before the package is imported). A JSON record of each invocation (family,
steps, run environment, timings) is written to
outputs/reproduce/<family>__<timestamp>.json.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import yaml

from . import generate, train, evaluate
from ..definitions import PAPER_CONFIGS_DIR, OUTPUT_DIR, TINY
from ..utils.run_env import environment

STEPS = ('generate', 'train', 'evaluate')


def families():
    return sorted(p.stem for p in Path(PAPER_CONFIGS_DIR).glob('*.yaml'))


def load_family(family):
    path = Path(PAPER_CONFIGS_DIR) / f'{family}.yaml'
    if not path.exists():
        raise SystemExit(f"Unknown family '{family}'. Available: {', '.join(families())}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return {'experiments': list(cfg.get('experiments') or []),
            'evaluations': list(cfg.get('evaluations') or [])}


def select(cfg, prefix):
    if not prefix:
        return cfg
    return {k: [n for n in v if n.startswith(prefix)] for k, v in cfg.items()}


def _write_record(family, record):
    out_dir = Path(OUTPUT_DIR) / 'reproduce'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    path = out_dir / f'{family}__{stamp}.json'
    path.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
    return path


def run(*argv):
    parser = argparse.ArgumentParser(prog='main.py reproduce', description=__doc__.split('\n\n')[0])
    parser.add_argument('family', nargs='?', help='configs/paper/<family>.yaml')
    parser.add_argument('--steps', nargs='+', choices=STEPS, default=list(STEPS),
                        help='steps to run, in this order (default: all)')
    parser.add_argument('--select', '-s', metavar='PREFIX', default=None,
                        help='only experiments/evaluations whose name starts with PREFIX')
    parser.add_argument('--tiny', action='store_true',
                        help='smoke run with tiny datasets, 2 epochs and 1 seed, under $DATA_DIR/outputs-tiny')
    parser.add_argument('--list', action='store_true', help='list the families and their experiments')
    args = parser.parse_args(argv)

    if args.tiny and not TINY:
        raise SystemExit("--tiny must be given on the command line of main.py "
                         "(it is set up before the package is imported)")

    if args.list or args.family is None:
        for family in families():
            cfg = load_family(family)
            print(f"{family}: {len(cfg['experiments'])} experiments, {len(cfg['evaluations'])} evaluations")
            for name in cfg['experiments']:
                print(f"    {name}")
        return

    cfg = select(load_family(args.family), args.select)
    steps = [s for s in STEPS if s in args.steps]
    print(f"Reproducing '{args.family}'{' (tiny)' if TINY else ''}: steps {', '.join(steps)}")
    print(f"  experiments: {', '.join(cfg['experiments']) or '-'}")
    print(f"  evaluations: {', '.join(cfg['evaluations']) or '-'}")
    print(f"  data dir:    {os.path.abspath(os.environ.get('DATA_DIR', '.'))}")

    record = {
        'family': args.family, 'tiny': TINY, 'steps': steps, 'select': args.select,
        'config': cfg, 'environment': environment(), 'started': datetime.now().astimezone().isoformat(),
        'step_seconds': {},
    }
    runners = {
        'generate': lambda: generate.run(config=cfg),
        'train':    lambda: train.run(config=cfg),
        'evaluate': lambda: evaluate.run(config=cfg),
    }
    try:
        for step in steps:
            print(f"\n{'=' * 60}\n# reproduce {args.family}: {step}\n{'=' * 60}")
            t0 = time.perf_counter()
            runners[step]()
            record['step_seconds'][step] = time.perf_counter() - t0
        record['status'] = 'completed'
    except BaseException as e:
        record['status'] = f'failed: {type(e).__name__}: {e}'
        raise
    finally:
        record['finished'] = datetime.now().astimezone().isoformat()
        path = _write_record(args.family, record)
        print(f"\nRun record written to {path}")
