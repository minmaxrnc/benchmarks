# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

from typing import Callable, Dict, Optional, Any, List, Union
import os, csv, time
import torch
import json
import platform
from itertools import product
from torch.utils.checkpoint import checkpoint

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
from ..utils.norms import sum_of_squares
from ..utils.oom import OOM_EXCEPTION_TYPES, is_oom_error
from ..utils.compile import compile_model as _compile_model

from ..definitions import TQDM_MININTERVAL

from ..ldtracker.ldtracker import LearningDynamicsTracker, DynamicsConfig

ldt_cfg = DynamicsConfig()

BEST_MODEL_OPTIONS = ('best_epoch', 'last_epoch')


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
        grad_clip:        Optional[float] = None,
        # Split each train_batch_size-sized batch into this many micro-batches,
        # accumulating their gradients before a single optimizer step. It
        # bounds memory along the batch dimension without changing the
        # effective batch size or the number of optimizer steps per epoch. It is
        # also the first lever of the OOM-retry loop (see _oom_retry).
        grad_accum_steps: int = 1,
        # Chunk size (in steps) the training forward pass starts from. Longer
        # sequences are processed chunk by chunk, each chunk under gradient
        # checkpointing (see _forward_checkpointed), so that backward memory is
        # bounded by the chunk size. None: the longest train/val sequence (no
        # chunking on the first attempt).
        max_unroll_steps: Optional[int] = None,
        # torch.compile for this trainer only. None: meta.config.yaml's
        # compile_model flag decides.
        compile_model:    Optional[bool] = None,
        # Epochs after which to also save a copy of the weights, per seed, as
        # checkpoint__<params>__seed_<s>__epoch_<e>.pt (0: the initial
        # weights). No effect on training.
        save_checkpoint_epochs: Optional[List[int]] = None,
        # Weights returned for each seed: 'best_epoch' or 'last_epoch'. None:
        # meta.config.yaml's best_model key decides.
        best_model:       Optional[str] = None
    ):

        self.ci            = ConfidenceIntervals()
        self.ci_min_n      = config.get('ci_min_n')
        self.ci_max_n      = config.get('ci_max_n')
        # self.ci_accuracy   = config.get('ci_accuracy_val')
        self.ci_confidence = config.get('ci_confidence')
        self.val_acc_giveup_threshold = config.get('val_acc_giveup_threshold')

        # for a single combination of training parameters
        self.max_time            = config.get('max_train_runtime')
        self.max_time_per_params = config.get('max_train_runtime__per_params')
        self.max_time_per_seed   = config.get('max_train_runtime__per_seed')

        self.tracker_enabled     = config.get('learning_dynamics_tracker__enabled')

        self.global_compile_model = config.get('compile_model')

        if best_model is None:
            best_model = config.get('best_model')
        if best_model not in BEST_MODEL_OPTIONS:
            raise ValueError(f"best_model must be one of {BEST_MODEL_OPTIONS}, got {best_model!r}")
        self.best_model = best_model

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

        self.grad_accum_steps       = grad_accum_steps
        self.max_unroll_steps       = max_unroll_steps
        self.compile_model          = compile_model
        self.save_checkpoint_epochs = sorted(set(save_checkpoint_epochs or []))

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
            'best_model':     self.best_model,
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
                if self._give_up(val_acc__high):
                    print(
                        f"No more seeds will be tried since the upper bound of the val_acc CI",
                        f"({format_accuracy(val_acc__high)}) is below",
                        f"{format_accuracy(self.val_acc_giveup_threshold)}",
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


    def _give_up(self, val_acc__high) -> bool:
        """Seed-skip rule: stop trying seeds once even the upper bound of the
        val_acc CI is below val_acc_giveup_threshold (disabled when None)."""
        if self.val_acc_giveup_threshold is None or val_acc__high is None:
            return False
        return val_acc__high < self.val_acc_giveup_threshold


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


        def __fit(model, unroll_steps: int = -1, grad_accum_steps: Optional[int] = None):
            if grad_accum_steps is None:
                grad_accum_steps = self.grad_accum_steps

            __n_val_batches = n_val_batches
            __val_loader = val_loader

            model.to(device)

            # The plain module: its state_dict() keys have no torch.compile
            # '_orig_mod.' prefix, so checkpoints saved from it load with
            # strict=True into a fresh model whether or not training was compiled.
            raw_model = model

            compile_active = self._compile_enabled()
            if compile_active:
                # Rewrapped on every call to __fit (e.g. each OOM retry), since a
                # new unroll_steps triggers a recompilation anyway.
                model = _compile_model(model)
            print(f"torch.compile: {'enabled' if compile_active else 'disabled'}")
            self._write_run_env(lr, train_batch_size, optimizer_name, scheduler_name, seed,
                                compile_active, unroll_steps, grad_accum_steps)

            loss = losses.instantiate(self.loss)
            metric = metrics.instantiate(self.metric)

            optimizer = self._init_optimizer(optimizer_name, raw_model, lr)

            scheduler = self._init_scheduler(
                scheduler_name,
                optimizer,
                self.epochs,
                len(train_loader)
            )


            tracker = None
            if self.tracker_enabled:
                tracker = LearningDynamicsTracker(
                    raw_model,
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
                if 0 in self.save_checkpoint_epochs:
                    self._save_epoch_checkpoint(raw_model, lr, train_batch_size, optimizer_name,
                                                scheduler_name, seed, 0)

                for epoch in range(1, self.epochs + 1):
                    epoch_msg  = "\nEpoch {}/{}".format(epoch, self.epochs)
                    epoch_msg += " (seed={} lr={} bs={}".format(seed, lr, train_batch_size)
                    epoch_msg += " opt={} sc={}".format(optimizer_name, scheduler_name)
                    if grad_accum_steps != 1:
                        epoch_msg += " grad_accum_steps={}".format(grad_accum_steps)
                    epoch_msg += ")"
                    tqdm.write(epoch_msg)

                    train_loss, _train_acc, step, epoch_stats = self._run_train_epoch(
                        model            = model,
                        loader           = train_loader,
                        optimizer        = optimizer,
                        scheduler        = scheduler,
                        loss_fn          = loss,
                        metric           = metric,
                        tracker          = tracker,
                        step             = step,
                        unroll_steps     = unroll_steps,
                        grad_accum_steps = grad_accum_steps,
                    )
                    _train_loss, train_acc = self._run_validate(
                        model            = model,
                        loader           = train_loader,
                        metric           = metric,
                        loss_fn          = loss,
                        desc             = 'Compute train metrics',
                        unroll_steps     = unroll_steps,
                        grad_accum_steps = grad_accum_steps,
                    )

                    tqdm.write(f"  Train:   loss={format_loss(train_loss)}  acc={format_accuracy(_train_acc)}")
                    tqdm.write(f"  TrainND: loss={format_loss(_train_loss)}  acc={format_accuracy(train_acc)}")
                    tqdm.write(f"  Throughput: {epoch_stats['throughput']:,.0f} tokens/s")
                    tqdm.write(
                        "  Grad norm: mean={:.4g} max={:.4g}   State norm: mean={:.4g} max={:.4g}".format(
                            epoch_stats['grad_norm_mean'], epoch_stats['grad_norm_max'],
                            epoch_stats['state_norm_mean'], epoch_stats['state_norm_max'],
                        )
                    )

                    val_success = False
                    while not val_success:
                        try:
                            val_loss, val_acc = self._run_validate(
                                model            = model,
                                loader           = __val_loader,
                                metric           = metric,
                                loss_fn          = loss,
                                desc             = 'Validate',
                                unroll_steps     = unroll_steps,
                                grad_accum_steps = grad_accum_steps,
                            )
                            val_success = True
                        except OOM_EXCEPTION_TYPES as e:
                            if not is_oom_error(e):
                                raise e
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
                        if self.best_model == 'best_epoch':
                            # A copy: state_dict() tensors alias the live
                            # parameters, which later optimizer steps update in
                            # place.
                            best_model_state_dict = self._copy_state_dict(raw_model)

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
                        'earlystop':  'yes' if stopper.get_stop() else '',
                        **epoch_stats,
                    })

                    if epoch in self.save_checkpoint_epochs:
                        self._save_epoch_checkpoint(raw_model, lr, train_batch_size, optimizer_name,
                                                    scheduler_name, seed, epoch)

                    if stopper.get_stop():
                        tqdm.write('  Training stopped because of no improvement')
                        break

                    if timer.is_over_limit():
                        tqdm.write('  Training stopped because max_runtime has been reached')
                        break

                timer.stop()

                if self.best_model == 'last_epoch':
                    # The weights at the end of training, paired with the best
                    # epoch's validation scores: what the runs before v0.2.0
                    # saved, since their state_dict() was not copied.
                    best_model_state_dict = self._copy_state_dict(raw_model)

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

                if compile_active:
                    self._write_run_env(lr, train_batch_size, optimizer_name, scheduler_name, seed,
                                        compile_active, unroll_steps, grad_accum_steps,
                                        after_training=True)

                return (
                    best_val_acc,
                    best_val_loss,
                    train_acc_for_best_val_acc,
                    best_model_state_dict
                )
            except Exception as e:
                if self.tracker_enabled:
                    tracker.close()
                parent_timer.pause(ok_if_already_paused=True)
                raise e


        model = models.instantiate(self.model, iosize, **model_kwargs)

        if model.supports_unroll_steps():
            unroll_steps     = self._initial_unroll_steps(train_dataset, val_dataset)
            grad_accum_steps = self.grad_accum_steps
            while True:
                try:
                    return __fit(model, unroll_steps=unroll_steps, grad_accum_steps=grad_accum_steps)
                except OOM_EXCEPTION_TYPES as e:
                    if not is_oom_error(e):
                        raise e
                    model.reset()
                    try:
                        new_unroll_steps, new_grad_accum_steps = self._oom_retry(
                            unroll_steps, grad_accum_steps, train_batch_size
                        )
                    except ValueError:
                        raise e
                    if new_grad_accum_steps != grad_accum_steps:
                        print(f"OOM: increasing grad_accum_steps to {new_grad_accum_steps}")
                    else:
                        print(
                            f"OOM: micro-batch size already 1 (grad_accum_steps={grad_accum_steps});",
                            f"decreasing unroll_steps to {new_unroll_steps}"
                        )
                    unroll_steps, grad_accum_steps = new_unroll_steps, new_grad_accum_steps
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
        unroll_steps,
        grad_accum_steps=1
    ):
        model.train(True)
        torch.set_grad_enabled(True)

        metric.reset()

        running_loss, n_samples = 0.0, 0

        per_sample_loss_fn = loss_fn.clone(reduction='none')

        # Throughput bookkeeping. Token counts are taken from tensor shapes
        # (padding included). The wall-clock span relies on the device sync
        # that loss.item() performs on every micro-batch. Time spent waiting
        # for the next batch (data_time) and computing the gradient/state
        # norms (stats_time; the gradient norm counts only when grad_clip is
        # None, since clipping needs it anyway) is subtracted from epoch_time.
        epoch_tokens  = 0
        stats_time    = 0.0
        data_time     = 0.0
        grad_norms    = []
        state_sq_sums = []
        epoch_t0      = time.perf_counter()
        batch_wait_t0 = time.perf_counter()

        with tqdm(loader, desc='Train', leave=False, mininterval=TQDM_MININTERVAL) as pb:
            for batch_idx, batch in enumerate(pb):
                data_time += time.perf_counter() - batch_wait_t0

                x    = batch['inputs']
                y    = batch['outputs']
                mask = batch['mask']

                x    = x.to(device,    non_blocking=True)
                y    = y.to(device,    non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Gradient accumulation: one backward per micro-batch, scaled so
                # that the summed gradient equals the full-batch one, and a
                # single optimizer step per batch.
                micro_batches = self._grad_accum_micro_batches(x, y, mask, grad_accum_steps)
                n_micros      = len(micro_batches)

                for x_mb, y_mb, mask_mb in micro_batches:
                    logits, state = self._forward_checkpointed(model, x_mb, unroll_steps,
                                                               return_state=True)

                    loss = loss_fn(logits, y_mb, mask_mb)

                    if self.tracker_enabled:
                        per_sample_loss = per_sample_loss_fn(logits, y_mb, mask_mb)
                        tracker.log_losses(step, loss, per_sample_losses=per_sample_loss, prefix="train")
                        tracker.log_optimizer(optimizer, step)
                        tracker.before_backward(step)

                    (loss / n_micros).backward()

                    if self.tracker_enabled:
                        tracker.after_backward(step)

                    stats_t0 = time.perf_counter()
                    state_sq = sum_of_squares(state)
                    if state_sq is not None:
                        state_sq_sums.append(state_sq)
                    stats_time += time.perf_counter() - stats_t0

                    # Update loss (unscaled, i.e. the mean loss of this micro-batch)
                    mb_size       = y_mb.size(0)
                    running_loss += loss.item() * mb_size
                    n_samples    += mb_size

                    # Update acc
                    metric.update(logits, y_mb, mask_mb)

                stats_t0 = time.perf_counter()
                grad_norms.append(self._measure_and_clip_grad_norm(model, self.grad_clip))
                if self.grad_clip is None:
                    stats_time += time.perf_counter() - stats_t0

                optimizer.step()

                if self.tracker_enabled:
                    tracker.after_step(step)

                # Batch scheduler step (e.g., LinearCosine)
                if schedulers.get_property(scheduler, 'step') == 'batch':
                    scheduler.step()

                epoch_tokens += x.numel()

                step += 1
                batch_wait_t0 = time.perf_counter()

        epoch_time = (time.perf_counter() - epoch_t0) - stats_time - data_time

        # Compute loss
        epoch_loss = running_loss / max(1, n_samples)

        # Compute acc
        epoch_acc = metric.compute()

        grad_norm_stats = self._grad_norm_epoch_stats(grad_norms)
        state_norm_mean, state_norm_max = self._tensor_list_mean_max(
            [s.sqrt() for s in state_sq_sums]
        )
        epoch_stats = {
            'tokens':          epoch_tokens,
            'epoch_time':      epoch_time,
            'stats_time':      stats_time,
            'data_time':       data_time,
            'throughput':      epoch_tokens / epoch_time if epoch_time > 0 else 0.0,
            'grad_norm_mean':  grad_norm_stats['mean'],
            'grad_norm_min':   grad_norm_stats['min'],
            'grad_norm_max':   grad_norm_stats['max'],
            'state_norm_mean': state_norm_mean,
            'state_norm_max':  state_norm_max,
        }

        return epoch_loss, epoch_acc, step, epoch_stats


    def _compile_enabled(self) -> bool:
        """This trainer's compile_model arg if set, otherwise meta.config.yaml's compile_model."""
        if self.compile_model is not None:
            return bool(self.compile_model)
        return bool(self.global_compile_model)


    @staticmethod
    def _copy_state_dict(model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}


    def _save_epoch_checkpoint(self, raw_model, lr, batch_size, optimizer_name, scheduler_name, seed, epoch):
        path = self._path('checkpoint', f'__epoch_{epoch}.pt', lr, batch_size,
                          optimizer_name, scheduler_name, seed)
        state_dict = {k: v.detach().to('cpu', copy=True) for k, v in raw_model.state_dict().items()}
        torch.save(state_dict, path)
        tqdm.write(f"  Saved epoch-{epoch} checkpoint: {os.path.basename(path)}")


    def _run_env(self, seed, compile_active, unroll_steps, grad_accum_steps):
        """The software/hardware environment and effective settings of one seed."""
        from importlib import metadata
        info = {
            'seed':             seed,
            'model':            self.model,
            'model_args':       models.get_meta(self.model).get('args'),
            'compile_model':    bool(compile_active),
            'compile_source':   'trainer_arg' if self.compile_model is not None else 'meta.config.yaml',
            'unroll_steps':     unroll_steps,
            'grad_accum_steps': grad_accum_steps,
            'max_unroll_steps': self.max_unroll_steps,
            'grad_clip':        self.grad_clip,
            'best_model':       self.best_model,
            'ci_seed':          config.get('ci_seed'),
            'python':           platform.python_version(),
            'platform':         platform.platform(),
            'torch':            torch.__version__,
            'cuda':             torch.version.cuda,
            'device':           str(device),
            'gpu':              torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        for pkg in ('minmaxrnc', 'numpy', 'triton'):
            try:
                info[pkg] = metadata.version(pkg)
            except Exception:
                info[pkg] = None
        return info


    def _write_run_env(self, lr, batch_size, optimizer_name, scheduler_name, seed,
                       compile_active, unroll_steps, grad_accum_steps, after_training=False):
        """Write run_env__<params>__seed_<s>.json. With after_training=True
        (compiled runs only), also record torch._dynamo's counters, i.e.
        evidence that graphs were compiled rather than run eagerly. Never
        fails training."""
        try:
            info = self._run_env(seed, compile_active, unroll_steps, grad_accum_steps)
            suffix = '.json'
            if after_training:
                suffix = '__after_training.json'
                try:
                    from torch._dynamo.utils import counters
                    info['dynamo_counters'] = {k: dict(v) for k, v in counters.items()
                                               if k in ('stats', 'frames', 'graph_break', 'recompiles')}
                except Exception as e:
                    info['dynamo_counters'] = f'unavailable: {e}'
            path = self._path('run_env', suffix, lr, batch_size, optimizer_name, scheduler_name, seed)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"Warning: could not write run_env file: {e}")


    @staticmethod
    def _measure_and_clip_grad_norm(model, grad_clip: Optional[float]) -> float:
        """The global gradient L2 norm before clipping, clipping in place to
        grad_clip if it is not None (max_norm=inf leaves gradients untouched)."""
        max_norm = grad_clip if grad_clip is not None else float('inf')
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return float(total_norm)


    @staticmethod
    def _grad_norm_epoch_stats(grad_norms: List[float]) -> Dict[str, float]:
        if not grad_norms:
            return {'mean': float('nan'), 'min': float('nan'), 'max': float('nan')}
        return {
            'mean': sum(grad_norms) / len(grad_norms),
            'min':  min(grad_norms),
            'max':  max(grad_norms),
        }


    @staticmethod
    def _tensor_list_mean_max(values):
        """One device sync for a list of 0-dim tensors -> (mean, max) as floats."""
        if not values:
            return float('nan'), float('nan')
        stacked = torch.stack(values).float().cpu()
        return stacked.mean().item(), stacked.max().item()


    @staticmethod
    def _oom_retry(unroll_steps: int, grad_accum_steps: int, train_batch_size: int):
        """The (unroll_steps, grad_accum_steps) to retry with after an OOM.
        Doubles grad_accum_steps while the micro-batch can still shrink, then
        halves unroll_steps. Raises ValueError when both are at their floor."""
        micro_batch_size = -(-train_batch_size // grad_accum_steps)  # ceil div
        if micro_batch_size > 1:
            return unroll_steps, grad_accum_steps * 2
        if unroll_steps > 1:
            return unroll_steps // 2, grad_accum_steps
        raise ValueError('Cannot retry further: micro-batch size and unroll_steps are both already 1')


    @staticmethod
    def _is_oom_error(e: BaseException) -> bool:
        return is_oom_error(e)


    def _initial_unroll_steps(self, train_dataset, val_dataset):
        """max_unroll_steps if set, otherwise the longest train/val sequence."""
        if self.max_unroll_steps is not None:
            return self.max_unroll_steps
        return max(train_dataset.max_len, val_dataset.max_len)


    def _grad_accum_micro_batches(self, x, y, mask, grad_accum_steps):
        """Split a batch into grad_accum_steps micro-batches along dim 0 (ceil
        division: the last one may be smaller). Returns (x, y, mask) triples."""
        micro_batch_size = -(-x.size(0) // grad_accum_steps)  # ceil div
        return list(zip(
            x.split(micro_batch_size, dim=0),
            y.split(micro_batch_size, dim=0),
            mask.split(micro_batch_size, dim=0),
        ))


    def _forward_checkpointed(self, model, x, unroll_steps, return_state=False):
        """model(x) one unroll_steps-sized chunk at a time, each chunk under
        torch.utils.checkpoint, so that its activations are recomputed in
        backward instead of being kept for the whole sequence. The state is
        carried across chunks and gradients flow through every chunk boundary,
        so outputs and gradients equal those of a single un-chunked call.

        Sequences no longer than unroll_steps, and models without
        unroll_steps support, get a single plain call (no checkpointing).
        """
        if not model.supports_unroll_steps():
            return model(x, return_state=True) if return_state else model(x)
        if unroll_steps is None or unroll_steps <= 0 or x.size(1) <= unroll_steps:
            return model(x, unroll_steps=unroll_steps, return_state=return_state)

        def _step(x_chunk, state):
            return model(x_chunk, state, unroll_steps=x_chunk.shape[1], return_state=True)

        state = None
        logits_chunks = []
        for x_chunk in x.split(unroll_steps, dim=1):
            logits_chunk, state = checkpoint(_step, x_chunk, state, use_reentrant=False)
            logits_chunks.append(logits_chunk)

        logits = logits_chunks[0] if len(logits_chunks) == 1 else torch.cat(logits_chunks, dim=1)
        if return_state:
            return logits, state
        return logits


    def _run_validate(self, model, loader, metric, loss_fn, desc, unroll_steps, grad_accum_steps=1):
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

                    # Same micro-batches as in training, so that growing
                    # grad_accum_steps on OOM also bounds this forward pass.
                    for x_mb, y_mb, mask_mb in self._grad_accum_micro_batches(x, y, mask, grad_accum_steps):
                        if model.supports_unroll_steps():
                            logits = model(x_mb, unroll_steps=unroll_steps)
                        else:
                            logits = model(x_mb)

                        loss = loss_fn(logits, y_mb, mask_mb)

                        # Update loss
                        mb_size       = y_mb.size(0)
                        running_loss += loss.item() * mb_size
                        n_samples    += mb_size

                        # Update acc
                        metric.update(logits, y_mb, mask_mb)

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

    # Per-epoch columns of train_log__*.csv after the original ones (see
    # _run_train_epoch). Missing values are left empty.
    STATS_COLUMNS = [
        'tokens', 'epoch_time', 'stats_time', 'data_time', 'throughput',
        'grad_norm_mean', 'grad_norm_min', 'grad_norm_max',
        'state_norm_mean', 'state_norm_max',
    ]

    def _csv_write_history(self, history, lr, batch_size, optimizer_name, scheduler_name, seed, n_lr):
        lr_header = ['lr_' + str(i) for i in range(n_lr)]
        header = ['epoch']
        header += lr_header
        header += ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        header += ['runtime', 'timelimit', 'earlystop']
        header += self.STATS_COLUMNS
        path = self._path('train_log', '.csv', lr, batch_size, optimizer_name, scheduler_name, seed)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for h in history:
                row  = [h['epoch']]
                row += h['lr']
                row += [h['train_loss'], h['train_acc'], h['val_loss'], h['val_acc']]
                row += [h['runtime'], h['timelimit'], h['earlystop']]
                row += [h.get(c, '') for c in self.STATS_COLUMNS]
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
