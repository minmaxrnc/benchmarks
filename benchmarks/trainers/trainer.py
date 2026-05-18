# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from typing import Callable, Dict, Optional, Any, List, Union
from torch import cuda
import os, csv
import torch
import json
from itertools import product

from ..optimizers.optimizers import optimizers
from ..stoppers import stoppers
from ..schedulers.schedulers import schedulers
from ..metrics.metrics import metrics
from ..losses.losses import losses
from ..models.models import models
from ..datasets.datasets import datasets

from ..utils import config
from ..utils.device import device
from ..utils.seed import Seeds, seed_everything
from ..utils.ci import ConfidenceIntervals
from ..utils.timer import Timer
from ..utils.format import format_accuracy, format_loss
from ..utils.tqdm import tqdm

from ..definitions import TQDM_MININTERVAL

from ..ldtracker.ldtracker import LearningDynamicsTracker, DynamicsConfig

ldt_cfg = DynamicsConfig()


class Trainer:
    def __init__(
        self,
        name:             str,
        *,
        model:            str,
        loss:             str,
        metric:           str,
        train_dataset:    str,
        val_dataset:      str,
        epochs:           int,
        csv_log_path:     str,
        best_model_path:  str,
        # --- Below are loaded args ---
        optimizer:        Union[str,List[str]],
        scheduler:        Union[str,List[str]],
        lr:               Union[float, List[float]],
        train_batch_size: Union[int, List[int]],
        grad_clip:        Optional[float] = None
    ):

        self.ci            = ConfidenceIntervals()
        self.ci_min_n      = config.get('ci_min_n')
        self.ci_max_n      = config.get('ci_max_n')
        # self.ci_accuracy   = config.get('ci_accuracy_val')
        self.ci_confidence = config.get('ci_confidence')

        # for a single combination of training parameters
        self.max_time            = config.get('max_train_runtime')
        self.max_time_per_params = config.get('max_train_runtime__per_params')
        self.max_time_per_seed   = config.get('max_train_runtime__per_seed')

        self.tracker_enabled     = config.get('learning_dynamics_tracker__enabled')

        if type(lr) == float:
            self.lr = [lr]
        else:
            self.lr = lr

        if type(train_batch_size) == int:
            self.train_batch_size = [train_batch_size]
        else:
            self.train_batch_size = train_batch_size

        self.name            = name
        self.model           = model
        self.loss            = loss
        self.metric          = metric
        self.train_dataset   = train_dataset
        self.val_dataset     = val_dataset
        self.epochs          = epochs
        self.csv_log_path    = csv_log_path
        self.best_model_path = best_model_path
        self.grad_clip       = grad_clip

        if type(scheduler) == list:
            self.scheduler = scheduler
        else:
            self.scheduler = [scheduler]

        if type(optimizer) == list:
            self.optimizer = optimizer
        else:
            self.optimizer = [optimizer]


    # ---------- Public API ----------
    def fit(self) -> Dict[str, list]:

        timer = Timer(max_time = self.max_time)

        n_ci_tests  = self.ci_max_n
        n_ci_tests *= len(self.lr)
        n_ci_tests *= len(self.train_batch_size)
        n_ci_tests *= len(self.optimizer)
        n_ci_tests *= len(self.scheduler)
        delta = (1 - self.ci_confidence) / n_ci_tests

        val_acc__best          = -1.0
        val_loss__best         = float('inf')
        train_acc__best        = None
        lr__best               = None
        train_batch_size__best = None
        optimizer__best        = None
        scheduler__best        = None
        val_acc__mean          = None

        for lr, train_batch_size, optimizer, scheduler in product(self.lr,
                                                                  self.train_batch_size,
                                                                  self.optimizer,
                                                                  self.scheduler):
            val_acc, val_loss, train_acc, model_dict = self._fit_params(lr,
                                                                        train_batch_size,
                                                                        optimizer,
                                                                        scheduler,
                                                                        timer,
                                                                        delta
                                                                        )

            if val_acc > val_acc__best or (val_acc == val_acc__best and val_loss < val_loss__best):
                val_acc__best          = val_acc
                val_loss__best         = val_loss
                train_acc__best        = train_acc
                lr__best               = lr
                train_batch_size__best = train_batch_size
                optimizer__best        = optimizer
                scheduler__best        = scheduler
                best_model_dict        = model_dict
                print(
                    f"New overall best:",
                    f"val_loss={format_loss(val_loss__best)},",
                    f"val_acc={format_accuracy(val_acc__best)}"
                )
            print(
                '  Time for current training:',
                timer.get_elapsed(pretty=True)
            )
            if timer.is_over_limit():
                print('Time limit exceeded')
                break

        timer.stop()

        torch.save(best_model_dict, self.best_model_path)

        print(
            f"Overall best:",
            f"val_loss={format_loss(val_loss__best)},",
            f"val_acc={format_accuracy(val_acc__best)},",
            f"lr={lr__best},",
            f"train_batch_size={train_batch_size__best},"
            f"optimizer={optimizer__best},",
            f"scheduler={scheduler__best}"
        )
        print(f"Total train time: {timer.get_elapsed(pretty=True)}")

        self._write_overall({
            'runtime':        timer.get_elapsed(),
            'runtime_pretty': timer.get_elapsed(pretty=True),
            'values_for_best_model': {
                'lr':               lr__best,
                'train_batch_size': train_batch_size__best,
                'optimizer':        optimizer__best,
                'scheduler':        scheduler__best,
                'val_acc':          val_acc__best,
                'val_loss':         val_loss__best,
                'train_acc':        train_acc__best
            }
        })


    def _fit_params(self, lr, train_batch_size, optimizer, scheduler, parent_timer, delta):

        timer = Timer(parent_timer, max_time=self.max_time_per_params)

        # Load and return if already run
        def __na_non_r(x): return None         if x == 'n/a' else x
        def __na_neg_r(x): return -1.0         if x == 'n/a' else x
        def __na_inf_r(x): return float('inf') if x == 'n/a' else x

        loaded = self._load_per_params(lr, train_batch_size, optimizer, scheduler)

        if loaded:
            print(
                f"\n\n# Loading results for training with lr={lr}, batch_size={train_batch_size},",
                f"optimizer={optimizer}, scheduler={scheduler}\n"
            )
            data, best_model_dict = loaded
            val_acc__best   = __na_neg_r(data['val_acc']['best'])
            val_loss__best  = __na_inf_r(data['val_loss']['best'])
            train_acc__best = __na_non_r(data['train_acc'])

            timer.add(data['time'])
            timer.stop()

            return val_acc__best, val_loss__best, train_acc__best, best_model_dict


        # Run if not loaded

        val_acc__best     = -1.0
        val_loss__best    = float('inf')
        train_acc__best   = None
        val_acc__mean,  val_acc__low,  val_acc__high  = None, None, None
        val_loss__mean, val_loss__low, val_loss__high = None, None, None
        val_acc__history  = []
        val_loss__history = []


        for seed in Seeds(self.ci_max_n):
            print(
                f"\n\n# Training with",
                f"lr={lr}, batch_size={train_batch_size}, optimizer={optimizer},",
                f"scheduler={scheduler}, seed={seed}\n"
            )

            val_acc, val_loss, train_acc, model_dict = self._fit(
                seed,
                lr,
                train_batch_size,
                optimizer,
                scheduler,
                timer
            )
            val_acc__history.append(val_acc)
            val_loss__history.append(val_loss)

            print(
                '\nTime for current params:',
                timer.get_elapsed(pretty=True)
            )

            if val_acc > val_acc__best or (val_acc == val_acc__best and val_loss < val_loss__best):
                val_acc__best   = val_acc
                val_loss__best  = val_loss
                train_acc__best = train_acc
                best_model_dict = model_dict
                print(
                    f"New best per params (",
                    f"val_loss={format_loss(val_loss__best)} ",
                    f"val_acc={format_accuracy(val_acc__best)})"
                )


            val_acc__mean, val_acc__low, val_acc__high = self.ci.mean_bootstrap(val_acc__history, delta)
            val_loss__mean, val_loss__low, val_loss__high = self.ci.mean_bootstrap(val_loss__history, delta)

            if len(val_loss__history) < self.ci_min_n:
                print(
                    f"  CI val_loss: [{format_loss(val_loss__low)}, {format_loss(val_loss__high)}]",
                    f"(not reliable)"
                )
                print(
                    f"  CI val_acc:  [{format_accuracy(val_acc__low)},"
                    f"{format_accuracy(val_acc__high)}]",
                    f"(not reliable)"
                )
            else:
                print(
                    f"  CI val_loss: [{format_loss(val_loss__low)}, {format_loss(val_loss__high)}]"
                )
                print(
                    f"  CI val_acc:  [{format_accuracy(val_acc__low)},",
                    f"{format_accuracy(val_acc__high)}]"
                )
                if val_loss__best <= val_loss__low:
                    print(
                        f"No more seeds will be tried since no better model will be found",
                        f"with confidence {self.ci_confidence}"
                    )
                    break
            if timer.is_over_limit():
                break

        timer.stop()

        def __na_non(x): return 'n/a' if x is None else x
        def __na_neg(x): return 'n/a' if x < 0 else x
        def __na_inf(x): return 'n/a' if x == float('inf') else x

        data_acc = {
            'best': __na_neg(val_acc__best),
            'mean': __na_non(val_acc__mean),
            'low':  __na_non(val_acc__low),
            'high': __na_non(val_acc__high)
        }
        data_loss = {
            'best': __na_inf(val_loss__best),
            'mean': __na_non(val_loss__mean),
            'low':  __na_non(val_loss__low),
            'high': __na_non(val_loss__high)
        }
        data = {
            'val_acc':           data_acc,
            'val_loss':          data_loss,
            'train_acc':         __na_non(train_acc__best),
            'time':              timer.get_elapsed(),
            'time_pretty':       timer.get_elapsed(pretty=True),
            'timelimit':         'reached' if timer.is_over_limit() else 'no'
        }

        self._write_per_params(data, best_model_dict, lr, train_batch_size, optimizer, scheduler)

        return val_acc__best, val_loss__best, train_acc__best, best_model_dict


    def _fit(self, seed, lr, train_batch_size, optimizer_name, scheduler_name, parent_timer):

        seed_everything(seed)

        print("Preparing train dataset...")
        train_dataset = datasets.instantiate(
            self.train_dataset,
            seed = seed
        )
        train_loader = train_dataset.get_loader(train_batch_size)
        print(f"Train dataset: {datasets.repr(train_dataset, seed)}")

        print("\nPreparing val_dataset...")
        val_dataset = datasets.instantiate(
            self.val_dataset,
            seed = seed
        )
        n_val_batches = 1
        val_loader = val_dataset.get_loader(len(val_dataset)//n_val_batches)
        print(f"Val dataset: {datasets.repr(val_dataset, seed)}")

        model_kwargs = {}
        if 'mode' in models.get_required_kwargs(self.model):
            model_kwargs['mode'] = 'training'
        if 'context_length' in models.get_required_kwargs(self.model):
            # model_kwargs['context_length'] = max(train_dataset.max_len, val_dataset.max_len)
            model_kwargs['context_length'] = train_dataset.max_len

        iosize = datasets.get_iosize(self.train_dataset)


        def __fit(model, unroll_steps: int = -1):

            __n_val_batches = n_val_batches
            __val_loader = val_loader

            model.to(device)

            loss = losses.instantiate(self.loss)
            metric = metrics.instantiate(self.metric)

            optimizer = self._init_optimizer(optimizer_name, model, lr)

            scheduler = self._init_scheduler(
                scheduler_name,
                optimizer,
                self.epochs,
                len(train_loader)
            )


            tracker = None
            if self.tracker_enabled:
                tracker = LearningDynamicsTracker(
                    model,
                    log_dir=self._path(
                        'ldtracker',
                        '',
                        lr,
                        train_batch_size,
                        optimizer_name,
                        scheduler_name,
                        seed
                    ),
                    cfg=ldt_cfg
                )

            history = []
            best_val_acc = -1.0
            best_val_loss = float('inf')
            train_acc_for_best_val_acc = None
            best_model_state_dict = None

            stopper = stoppers.instantiate('emastopper')

            timer = Timer(parent_timer, max_time = self.max_time_per_seed)
            timer.start()

            step = 0

            try:
                for epoch in range(1, self.epochs + 1):
                    epoch_msg  = "\nEpoch {}/{}".format(epoch, self.epochs)
                    epoch_msg += " (seed={} lr={} bs={}".format(seed, lr, train_batch_size)
                    epoch_msg += " opt={} sc={})".format(optimizer_name, scheduler_name)
                    tqdm.write(epoch_msg)

                    train_loss, _train_acc, step = self._run_train_epoch(
                        model        = model,
                        loader       = train_loader,
                        optimizer    = optimizer,
                        scheduler    = scheduler,
                        loss_fn      = loss,
                        metric       = metric,
                        tracker      = tracker,
                        step         = step,
                        unroll_steps = unroll_steps,
                    )
                    _train_loss, train_acc = self._run_validate(
                        model   = model,
                        loader  = train_loader,
                        metric  = metric,
                        loss_fn = loss,
                        desc    = 'Compute train metrics',
                        unroll_steps = unroll_steps,
                    )

                    tqdm.write(f"  Train:   loss={format_loss(train_loss)}  acc={format_accuracy(_train_acc)}")
                    tqdm.write(f"  TrainND: loss={format_loss(_train_loss)}  acc={format_accuracy(train_acc)}")

                    val_success = False
                    while not val_success:
                        try:
                            val_loss, val_acc = self._run_validate(
                                model   = model,
                                loader  = __val_loader,
                                metric  = metric,
                                loss_fn = loss,
                                desc    = 'Validate',
                                unroll_steps = unroll_steps,
                            )
                            val_success = True
                        except cuda.OutOfMemoryError as e:
                            if __n_val_batches == len(val_dataset):
                                raise Exception('Failed to validate even with one batch per sample')
                            __n_val_batches += 1
                            __val_loader = val_dataset.get_loader(len(val_dataset)//__n_val_batches)
                            print(f"Decreasing validation batch size to {len(val_dataset)//__n_val_batches}")

                    tqdm.write(f"  Val:     loss={format_loss(val_loss)}  acc={format_accuracy(val_acc)}")

                    tqdm.write(f"  Time for current seed: {timer.get_elapsed(pretty=True)}")
                    stopper.update(val_loss)


                    if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        train_acc_for_best_val_acc = train_acc
                        best_model_state_dict = model.state_dict()

                        new_best_msg = "\nNew best (val_loss={} val_acc={})".format(
                            format_loss(best_val_loss),format_accuracy(best_val_acc)
                        )
                        tqdm.write(new_best_msg)

                    # LR of each param group (before step, so it reflects what this epoch trained with)
                    cur_lr = [float(g['lr']) for g in optimizer.param_groups]

                    # ---- Scheduler step ----
                    if schedulers.get_property(scheduler, 'step') == 'plateau':
                        target = val_loss
                        scheduler.step(target)  # type: ignore[attr-defined]
                    elif schedulers.get_property(scheduler, 'step') == 'epoch':
                        scheduler.step()

                    history.append({
                        'epoch':      epoch,
                        'lr':         cur_lr,
                        'train_loss': train_loss,
                        'train_acc':  train_acc,
                        'val_loss':   val_loss,
                        'val_acc':    val_acc,
                        'runtime':    timer.get_elapsed(),
                        'timelimit':  'reached' if timer.is_over_limit() else '',
                        'earlystop':  'yes' if stopper.get_stop() else ''
                    })

                    if stopper.get_stop():
                        tqdm.write('  Training stopped because of no improvement')
                        break

                    if timer.is_over_limit():
                        tqdm.write('  Training stopped because max_runtime has been reached')
                        break

                timer.stop()

                self._csv_write_history(
                    history,
                    lr,
                    train_batch_size,
                    optimizer_name,
                    scheduler_name,
                    seed,
                    n_lr=len(optimizer.param_groups)
                )

                if self.tracker_enabled:
                    tracker.close()

                return (
                    best_val_acc,
                    best_val_loss,
                    train_acc_for_best_val_acc,
                    best_model_state_dict
                )
            except Exception as e:
                if self.tracker_enabled:
                    tracker.close()
                parent_timer.pause()
                raise e


        model = models.instantiate(self.model, iosize, **model_kwargs)

        if model.supports_unroll_steps():
            unroll_steps = max(train_dataset.max_len, val_dataset.max_len)
            while True:
                try:
                    return __fit(model, unroll_steps=unroll_steps)
                except cuda.OutOfMemoryError as e:
                    if unroll_steps > 1:
                        unroll_steps //= 2
                        model.reset()
                    else:
                        raise e
        else:
            return __fit(model)


    def _run_train_epoch(
        self,
        model,
        loader,
        metric,
        loss_fn,
        optimizer,
        scheduler,
        tracker,
        step,
        unroll_steps
    ):
        model.train(True)
        torch.set_grad_enabled(True)

        metric.reset()

        running_loss, n_samples = 0.0, 0

        per_sample_loss_fn = loss_fn.clone(reduction='none')

        with tqdm(loader, desc='Train', leave=False, mininterval=TQDM_MININTERVAL) as pb:
            for batch_idx, batch in enumerate(pb):
                x    = batch['inputs']
                y    = batch['outputs']
                mask = batch['mask']

                x    = x.to(device,    non_blocking=True)
                y    = y.to(device,    non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                optimizer.zero_grad()
                if model.supports_unroll_steps():
                    logits = model(x, unroll_steps=unroll_steps)
                else:
                    logits = model(x)

                loss = loss_fn(logits, y, mask)

                if self.tracker_enabled:
                    per_sample_loss = per_sample_loss_fn(logits, y, mask)
                    tracker.log_losses(step, loss, per_sample_losses=per_sample_loss, prefix="train")
                    tracker.log_optimizer(optimizer, step)
                    tracker.before_backward(step)

                loss.backward()

                if self.tracker_enabled:
                    tracker.after_backward(step)

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimizer.step()

                if self.tracker_enabled:
                    tracker.after_step(step)

                # Batch scheduler step (e.g., LinearCosine)
                if schedulers.get_property(scheduler, 'step') == 'batch':
                    scheduler.step()

                # Update loss
                batch_size    = y.size(0)
                running_loss += loss.item() * batch_size
                n_samples    += batch_size

                # Update acc
                metric.update(logits, y, mask)

                step += 1

        # Compute loss
        epoch_loss = running_loss / max(1, n_samples)

        # Compute acc
        epoch_acc = metric.compute()

        return epoch_loss, epoch_acc, step


    def _run_validate(self, model, loader, metric, loss_fn, desc, unroll_steps):
        model.train(False)
        torch.set_grad_enabled(False)

        running_loss, n_samples = 0.0, 0

        with torch.no_grad():
            metric.reset()
            with tqdm(loader, desc, leave=False, mininterval=TQDM_MININTERVAL) as pb:
                for batch_idx, batch in enumerate(pb):
                    x    = batch['inputs']
                    y    = batch['outputs']
                    mask = batch['mask']

                    x    = x.to(device,    non_blocking=True)
                    y    = y.to(device,    non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    if model.supports_unroll_steps():
                        logits = model(x, unroll_steps=unroll_steps)
                    else:
                        logits = model(x)

                    loss = loss_fn(logits, y, mask)

                    # Update loss
                    batch_size    = y.size(0)
                    running_loss += loss.item() * batch_size
                    n_samples    += batch_size

                    # Update acc
                    metric.update(logits, y, mask)

            # Compute loss
            val_loss = running_loss / max(1, n_samples)

            # Compute acc
            val_acc = metric.compute()

        torch.set_grad_enabled(True)
        return val_loss, val_acc


    def _init_scheduler(self, scheduler, optimizer, epochs, n_batches):
        scheduler_required_kwargs = schedulers.get_required_kwargs(scheduler)
        scheduler_kwargs = {}
        if 'steps' in scheduler_required_kwargs:
            scheduler_kwargs['steps'] = epochs * num_train_batches
        if 'batches' in scheduler_required_kwargs:
            scheduler_kwargs['batches'] = n_batches
        if 'epochs' in scheduler_required_kwargs:
            scheduler_kwargs['epochs'] = epochs

        return schedulers.instantiate(
            scheduler,
            optimizer=optimizer,
            **scheduler_kwargs
        )


    def _init_optimizer(self, optimizer, model, lr):
        if optimizers.get_property(optimizer, 'requires_decay_groups'):
            decay, no_decay = model.create_weight_decay_optim_groups()
            return optimizers.instantiate(
                optimizer,
                lr       = lr,
                decay    = decay,
                no_decay = no_decay
            )
        else:
            return optimizers.instantiate(
                optimizer,
                lr     = lr,
                params = model.parameters()
            )


    def _write_overall(self, data):
        filename = os.path.join(
            self.csv_log_path,
            'train_log__summary.json'
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def _load_per_params(self, lr, batch_size, optimizer_name, scheduler_name):
        param_names  = ['lr', 'bs', 'opt', 'sc']
        param_values = [lr, batch_size, optimizer_name, scheduler_name]
        params = [name + '_' + str(val) for name, val in zip(param_names, param_values)]
        params = '__'.join(params)

        path = os.path.join(
            self.csv_log_path,
            'train_log__' + params + '.json'
        )
        model_path = os.path.join(
            self.csv_log_path,
            'best_model__' + params + '.pt'
        )

        if not os.path.exists(path) or not os.path.exists(model_path):
            return None

        with open(path) as f:
            data = json.load(f)
        model_dict = torch.load(model_path)

        return data, model_dict


    def _write_per_params(self, data, model_dict, lr, batch_size, optimizer_name, scheduler_name):
        param_names  = ['lr', 'bs', 'opt', 'sc']
        param_values = [lr, batch_size, optimizer_name, scheduler_name]
        params = [name + '_' + str(val) for name, val in zip(param_names, param_values)]
        params = '__'.join(params)

        path = os.path.join(
            self.csv_log_path,
            'train_log__' + params + '.json'
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        model_path = os.path.join(
            self.csv_log_path,
            'best_model__' + params + '.pt'
        )
        torch.save(model_dict, model_path)


    def _path(self, prefix, suffix, lr, batch_size, optimizer_name, scheduler_name, seed):
        param_names  = ['lr', 'bs', 'opt', 'sc', 'seed']
        param_values = [lr, batch_size, optimizer_name, scheduler_name, seed]
        params = [name + '_' + str(val) for name, val in zip(param_names, param_values)]
        params = '__'.join(params)
        path = os.path.join(
            self.csv_log_path,
            prefix + '__' + params + suffix
        )
        return path

    def _csv_write_history(self, history, lr, batch_size, optimizer_name, scheduler_name, seed, n_lr):
        lr_header = ['lr_' + str(i) for i in range(n_lr)]
        header = ['epoch']
        header += lr_header
        header += ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        header += ['runtime', 'timelimit', 'earlystop']
        path = self._path('train_log', '.csv', lr, batch_size, optimizer_name, scheduler_name, seed)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for h in history:
                row  = [h['epoch']]
                row += h['lr']
                row += [h['train_loss'], h['train_acc'], h['val_loss'], h['val_acc']]
                row += [h['runtime'], h['timelimit'], h['earlystop']]
                writer.writerow(row)


    @staticmethod
    def get_required_kwargs():
        return [
            'model',
            'loss',
            'metric',
            'train_dataset',
            'val_dataset',
            'epochs',
            'csv_log_path',
            'best_model_path'
        ]

