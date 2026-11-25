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
import time
import torch
import math
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from ..utils.device import device
from ..utils import config
from ..utils.seed import Seeds, seed_everything
from ..utils.ci import ConfidenceIntervals
from ..utils.timer import Timer
from ..utils.misc import pjoin
from ..utils.gpumem import gpu_memory, bytes2miB
from ..utils.tqdm import tqdm, trange
from ..utils.norms import sum_of_squares
from ..utils.oom import OOM_EXCEPTION_TYPES, is_oom_error
from ..utils.compile import compile_model as _compile_model

from ..models.utils import _nbytes
from ..datasets import datasets
from ..metrics.metrics import metrics
from ..models.models import models

from ..definitions import TQDM_MININTERVAL


class Evaluator:
    def __init__(
        self,
        name:         str,
        model:        str,
        metric:       str,
        dataset:      str,
        csv_log_path: str,
        model_state_dict,
        # torch.compile the model; None: meta.config.yaml's compile_model flag
        # (Evaluation passes the compile_model argument of the trainer).
        compile_model: Optional[bool] = None,
        # ---- loaded args below
        chunk_size:   int = 1024
    ):

        length = datasets.get_length(dataset)

        if (length % chunk_size) != 0:
            raise ValueError('Length not divisible by chunk_size')

        self.length     = length
        self.chunk_size = chunk_size
        self.num_chunks = length // chunk_size

        self.name = name

        self.model            = model
        self.model_state_dict = model_state_dict
        self.metric           = metric
        self.dataset          = dataset
        self.csv_log_path     = csv_log_path

        self.ci                = ConfidenceIntervals()
        self.ci_min_n          = config.get('ci_min_n')
        self.ci_max_n          = config.get('ci_max_n')
        self.ci_accuracy__min  = config.get('ci_accuracy__min')
        self.ci_accuracy__cum  = config.get('ci_accuracy__cum')
        self.ci_confidence     = config.get('ci_confidence')
        self.max_eval_runtime  = config.get('max_eval_runtime')

        # Largest batch size tried first (None: the whole dataset in one batch).
        self.eval_max_batch_size = config.get('eval_max_batch_size')

        if compile_model is None:
            compile_model = config.get('compile_model')
        self.compile_model = bool(compile_model)


    @staticmethod
    def strip_compile_prefix(state_dict):
        """Remove the '_orig_mod.' prefix that torch.compile adds to the
        state_dict() keys, so that checkpoints saved from a compiled module
        load into a plain one. Checkpoints without the prefix are unchanged."""
        prefix = '_orig_mod.'
        if not any(k.startswith(prefix) for k in state_dict):
            return state_dict
        return {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }


    def _prepare_model(self):
        """Instantiate, load and (optionally) compile the model, once for the
        whole evaluation: torch.compile guards specialise on the module
        instance, so recompiling a new model for every seed would exhaust
        dynamo's recompile limit after a few seeds and fall back to eager."""
        print(f"Preparing model {models.str(self.model)}")
        model = models.instantiate(
            self.model,
            datasets.get_iosize(self.dataset)
        )
        model.load_state_dict(self.strip_compile_prefix(self.model_state_dict), strict=True)
        model.train(False)
        model.to(device)
        if self.compile_model:
            model = _compile_model(model)
        print(f"torch.compile: {'enabled' if self.compile_model else 'disabled'}")
        return model


    ## ---------- Public API ----------
    def evaluate(self) -> None:

        delta = (1 - self.ci_confidence) / (2 * self.ci_max_n + self.num_chunks)

        timer = Timer()
        seeds = Seeds(self.ci_max_n)

        prepared_model = []
        def get_model():
            if not prepared_model:
                prepared_model.append(self._prepare_model())
            return prepared_model[0]

        histories        = []
        acc_min__history = []
        acc_cum__history = []

        for iteration_idx, seed in enumerate(seeds):
            timer_seed = Timer(timer, max_time=self.max_eval_runtime)

            print(f"\n\n# Evaluating with seed={seed}\n")
            history  = self._evaluate(seed, timer_seed, get_model)
            acc_min = history.get_min('acc')
            acc_cum  = history.get_last('acc_cum')
            tqdm.write(f"  accuracy for seed {seed}: min={acc_min:.3f}, mean={acc_cum:.3f}")

            histories.append(history)
            acc_min__history.append(acc_min)
            acc_cum__history.append(acc_cum)

            if iteration_idx + 1 < self.ci_min_n:
                continue

            acc_min__confidence_interval = self.ci.mean_bootstrap(acc_min__history, delta)
            acc_min__mean, acc_min__low, acc_min__high = acc_min__confidence_interval

            acc_cum__confidence_interval = self.ci.mean_bootstrap(acc_cum__history, delta)
            acc_cum__mean, acc_cum__low, acc_cum__high = acc_cum__confidence_interval

            acc_cum__ci_size  = acc_cum__high - acc_cum__low
            acc_min__ci_size = acc_min__high - acc_min__low
            tqdm.write(
                f"CI acc_cum:     [{acc_cum__low},{acc_cum__high}] (confidence: {self.ci_confidence})"
            )
            tqdm.write(
                f"CI for acc_min: [{acc_min__low},{acc_min__high}] (confidence: {self.ci_confidence})"
            )

            ci_accuracy__min__goal = False
            for ub, size in self.ci_accuracy__min:
                ci_accuracy__min__goal |= (acc_min__high <= ub and acc_min__ci_size <= size)
            ci_accuracy__cum__goal = False
            for ub, size in self.ci_accuracy__cum:
                ci_accuracy__cum__goal |= (acc_cum__high <= ub and acc_cum__ci_size <= size)
            ci_accuracy__goal  = ci_accuracy__min__goal and ci_accuracy__cum__goal
            ci_accuracy__goal |= math.isnan(acc_min__high) or math.isnan(acc_min__low)

            if ci_accuracy__goal:
                tqdm.write(
                    'No more seeds will be tried since the confidence intervals are as required.'
                )
                break

            if timer_seed.is_over_limit():
                tqdm.write(
                    'No more seeds will be tried since the time limit has been reached.'
                )
                break

        timer.stop()

        overall_history = {
            'mean': [],
            'low':  [],
            'high': [],
            'step': history.get('step')
        }
        with trange(self.num_chunks, desc='CI', leave=False, mininterval=TQDM_MININTERVAL) as pbar:
            for step in pbar:
                accuracies = [h.get_step('acc', step) for h in histories]
                confidence_interval = self.ci.mean_bootstrap(accuracies, delta)
                acc__mean, acc__low, acc__high = confidence_interval
                overall_history['mean'].append(acc__mean)
                overall_history['low'].append(acc__low)
                overall_history['high'].append(acc__high)

        self._csv__write_overall_history(overall_history)
        self._json__write_summary({
            'runtime': timer.get_elapsed(),
            'n_seeds': seeds.get_used_seeds(),
            'acc_min': {
                'mean': acc_min__mean,
                'high': acc_min__high,
                'low':  acc_min__low
            },
            'acc_cum': {
                'mean': acc_cum__mean,
                'high': acc_cum__high,
                'low':  acc_cum__low
            }
        })


    def _evaluate(self, seed, timer, get_model):

        already_completed, history, runtime = self._check_completed_seed(seed)
        if already_completed:
            print('  already run')
            timer.add(runtime)
            timer.stop()
            return history

        ## Preparing dataset

        print(f"Preparing dataset {datasets.str(self.dataset, seed)}")
        dataset = datasets.instantiate(
            self.dataset,
            seed = seed
        )

        model = get_model()

        # Function to have an attempt with a given batch_size
        def __evaluate(batch_size):

            loader = dataset.get_loader(batch_size)

            ## Preparing metric

            metric = metrics.instantiate(self.metric)
            metric.set_steps_and_reset(self.num_chunks)

            ## Running evaluation

            torch.set_grad_enabled(False)

            timer.resume()

            runtimes    = [0]  * self.num_chunks
            timelimits  = [''] * self.num_chunks

            # Throughput bookkeeping: tokens from tensor shapes; the whole seed
            # is timed once, with a device sync before reading the clock.
            # Time spent waiting for batches (data_time) and computing the
            # state norms (stats_time) is subtracted.
            tokens_processed   = 0
            stats_time         = 0.0
            data_time          = 0.0
            state_sq_per_chunk = [None] * self.num_chunks
            throughput_t0      = time.perf_counter()
            batch_wait_t0      = time.perf_counter()

            with tqdm(total=len(loader), desc='batches', leave=False, mininterval=TQDM_MININTERVAL) as pb:
                for batch_idx, batch in enumerate(loader):
                    data_time += time.perf_counter() - batch_wait_t0

                    x    = batch['inputs']
                    y    = batch['outputs']
                    mask = batch['mask']

                    # move batch to device
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)


                    # split batches
                    x    = x.split(self.chunk_size, dim=1)
                    y    = y.split(self.chunk_size, dim=1)
                    mask = mask.split(self.chunk_size, dim=1)

                    model_state = None # init the state of the model

                    # Process batch chunk-wise
                    with tqdm(total=self.length,
                              desc='  steps',
                              leave=False,
                              unit_scale=True,
                              mininterval=TQDM_MININTERVAL) as chunk_pbar:
                        for chunk_idx, (x_chunk, y_chunk, mask_chunk) in enumerate(zip(x,y,mask)):

                            # evaluate model, providing the previous state
                            if model.supports_unroll_steps():
                                logits_chunk, model_state = model(x_chunk,
                                                                  model_state,
                                                                  return_state=True,
                                                                  unroll_steps=self.chunk_size)
                            else:
                                logits_chunk, model_state = model(x_chunk,
                                                                  model_state,
                                                                  return_state=True)

                            # update metric
                            metric.update(logits_chunk, y_chunk, mask_chunk, chunk_idx)

                            tokens_processed += x_chunk.numel()

                            stats_t0 = time.perf_counter()
                            state_sq = sum_of_squares(model_state)
                            if state_sq is not None:
                                prev = state_sq_per_chunk[chunk_idx]
                                state_sq_per_chunk[chunk_idx] = state_sq if prev is None else prev + state_sq
                            stats_time += time.perf_counter() - stats_t0

                            chunk_pbar.update(self.chunk_size)
                            runtimes[chunk_idx] += timer.get_elapsed()
                            if timer.is_over_limit():
                                timelimits[chunk_idx] = 'reached'
                                break

                    pb.update()

                    if timer.is_over_limit():
                        break

                    batch_wait_t0 = time.perf_counter()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            throughput_time = (time.perf_counter() - throughput_t0) - stats_time - data_time
            throughput = tokens_processed / throughput_time if throughput_time > 0 else 0.0

            state_norm = [float('nan')] * self.num_chunks
            present = [i for i, s in enumerate(state_sq_per_chunk) if s is not None]
            if present:
                norms = torch.stack([state_sq_per_chunk[i] for i in present]).sqrt().cpu().tolist()
                for i, v in zip(present, norms):
                    state_norm[i] = v

            timer.stop()

            history = HistorySeed(
                step       = [self.chunk_size * (chunk_idx+1) for chunk_idx in range(self.num_chunks)],
                acc        = metric.compute(),
                acc_cum    = metric.compute_cumulative(),
                runtime    = runtimes,
                timelimit  = timelimits,
                state_norm = state_norm
            )

            self._csv__write_history_per_seed(seed, history)
            self._json_write_runinfo_per_seed(
                seed,
                runtime        = timer.get_elapsed(),
                runtime_pretty = timer.get_elapsed(pretty=True),
                tokens         = tokens_processed,
                stats_time     = stats_time,
                data_time      = data_time,
                throughput     = throughput,
                batch_size     = batch_size
            )
            self._write_completed_per_seed(seed)

            return history

        # We will call the above function with more and more batches until memory suffices
        n_batches = 1
        if self.eval_max_batch_size is not None:
            n_batches = max(n_batches, math.ceil(len(dataset) / self.eval_max_batch_size))
        success = False
        while not success:
            try:
                batch_size = math.ceil(len(dataset) / n_batches)
                history = __evaluate(batch_size)
                success = True
            except OOM_EXCEPTION_TYPES as e:
                if not is_oom_error(e):
                    raise
                if n_batches == len(dataset):
                    raise Exception("Not enough memory even with a batch per sample... Giving up!")
                timer.pause(ok_if_already_paused=True)
                n_batches += 1
                print(f"Reducing batch size to {math.ceil(len(dataset) / n_batches)}")

        return history



    @staticmethod
    def get_required_kwargs():
        return []


    def _get_path__history_per_seed(self, seed):
        return pjoin(
            self.csv_log_path,
            'eval_log__seed_{}'.format(seed),
            ext='csv'
        )

    def _get_path__runinfo_per_seed(self, seed):
        return pjoin(
            self.csv_log_path,
            'runinfo__seed_{}'.format(seed),
            ext='json'
        )

    def _get_path__completed_per_seed(self, seed):
        return pjoin(
            self.csv_log_path,
            'completed__seed_{}'.format(seed),
            ext='txt'
        )

    def _json_write_runinfo_per_seed(self, seed, runtime, runtime_pretty, **stats):
        """stats: throughput bookkeeping of the seed (tokens, stats_time,
        data_time, throughput in tokens/s, batch_size)."""
        path = self._get_path__runinfo_per_seed(seed)
        data = {
            'runtime': runtime,
            'runtime_pretty': runtime_pretty,
            **stats
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def _check_completed_seed(self, seed):
        completed_path = self._get_path__completed_per_seed(seed)
        has_been_completed = Path(completed_path).exists()
        runtime = None
        history = None

        if has_been_completed:
            # Load runtime
            runinfo_path = self._get_path__runinfo_per_seed(seed)
            with open(runinfo_path) as f:
                runinfo = json.load(f)
                runtime = runinfo['runtime']

            # Load history
            history_path = self._get_path__history_per_seed(seed)
            history = HistorySeed.load_from_csv(history_path)

        return has_been_completed, history, runtime


    def _write_completed_per_seed(self, seed):
        ts = datetime.now().astimezone().isoformat(timespec="seconds")

        completed_text_fmt  = "Evaluation {} for model '{}' with seed {} completed at"
        completed_text_fmt += "the following time:\n{}\n"
        completed_text      = completed_text_fmt.format(self.name, self.model, seed, ts)

        path = self._get_path__completed_per_seed(seed)
        Path(path).write_text(completed_text, encoding="utf-8")


    def _csv__write_history_per_seed(self, seed, history):
        path = self._get_path__history_per_seed(seed)
        history.write_to_csv(path)


    def _csv__write_overall_history(self, history):
        header = ['step', 'acc_mean', 'acc_low', 'acc_high']
        path = os.path.join(
            self.csv_log_path,
            'eval_log.csv'
        )
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            for step, mean, low, high in zip(
                history['step'], history['mean'], history['low'], history['high']
            ):
                row = [step, mean, low, high]
                csv.writer(f).writerow(row)


    def _json__write_summary(self, data):
        path = os.path.join(
            self.csv_log_path,
            'eval_summary.json'
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class HistorySeed:

    @staticmethod
    def keys():
        return ['step', 'acc', 'acc_cum', 'runtime', 'timelimit', 'state_norm']

    @staticmethod
    def datatypes():
        return [int, float, float, float, str, float]

    # Keys of the histories written before state_norm was recorded; loading
    # one of them fills state_norm with NaN.
    LEGACY_KEYS = ['step', 'acc', 'acc_cum', 'runtime', 'timelimit']

    def _check_keys(self, kwargs):
        for key in kwargs:
            if key not in self.keys():
                raise ValueError(f"Given invalid key: {key}")
        for key in self.keys():
            if key not in kwargs:
                raise ValueError(f"Missing key: {key}")

    def __init__(self, **kwargs):
        self.data = {}

        if len(kwargs) == 0:
            for key in self.keys():
                self.data[key] = []
        else:
            self._check_keys(kwargs)
            for key in self.keys():
                self.data[key] = kwargs[key]

    def append(self, **kwargs):
        self._check_keys(kwargs)
        for key in self.keys():
            self.data[key].append(kwargs[key])

    def get_last(self, key):
        return self.data[key][-1]

    def get_min(self, key):
        return min(self.data[key])

    def get_max(self, key):
        return max(self.data[key])

    def get_step(self, key, step):
        return self.data[key][step]

    def get(self, key):
        return self.data[key]

    def write_to_csv(self, path):
        header = self.keys()
        data = [self.data[key] for key in self.keys()]
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(header)
            for row in zip(*data):
                csv.writer(f).writerow(row)

    @staticmethod
    def load_from_csv(path):
        with open(path, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header == HistorySeed.keys():
                legacy = False
            elif header == HistorySeed.LEGACY_KEYS:
                legacy = True
            else:
                raise Exception('Invalid header found in csv file')
            history = HistorySeed()
            for row in reader:
                if legacy:
                    row = row + ['nan']
                row_data = {
                    key: dt(row[key_idx])
                    for key_idx, (key, dt) in enumerate(zip(HistorySeed.keys(),
                                                            HistorySeed.datatypes()))
                }
                history.append(**row_data)
            return history

