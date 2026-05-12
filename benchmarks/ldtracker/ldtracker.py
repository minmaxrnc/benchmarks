# Copyright (C) 2026 Alessandro Ronca
#
# This file is part of minmaxrnc-benchmarks.
#
# minmaxrnc-benchmarks is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option any later version.

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def _global_l2_norm(tensors: Iterable[torch.Tensor]) -> float:
    total = 0.0
    for t in tensors:
        if t is None:
            continue
        total += t.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    # logs first param group; extend if you use multiple groups
    return float(optimizer.param_groups[0].get("lr", 0.0))


@dataclass
class DynamicsConfig:
    # How often to log scalar metrics (loss, lr, norms)
    scalar_interval: int = 10

    # How often to log per-layer norms (can be noisy/large)
    layerwise_interval: int = 200

    # How often to compute update norm (requires parameter snapshot; can be expensive)
    update_interval: int = 50

    # Activation logging
    act_interval: int = 50                 # mean/std scalars
    act_hist_interval: int = 200           # histograms (more expensive)
    act_hist_max_elems: int = 4096         # cap elements logged per layer histogram

    # Which module types to hook for activations
    activation_module_types: Tuple[type, ...] = (nn.Linear, nn.Parameter)

    # If True, also log gradients histograms (usually too heavy; default off)
    grad_hist_interval: Optional[int] = None
    grad_hist_max_elems: int = 4096


class LearningDynamicsTracker:
    """
    Plug-and-play learning dynamics logger for PyTorch training loops.

    Logs:
      - train loss + (optional) per-sample loss mean/std + histogram
      - learning rate
      - global param norm, global grad norm, grad/param ratio
      - update norm and update/param ratio (every cfg.update_interval steps)
      - layer-wise param/grad norms (every cfg.layerwise_interval steps)
      - activation mean/std and histograms via forward hooks
    """

    def __init__(
        self,
        model: nn.Module,
        log_dir: str = "runs/exp",
        cfg: DynamicsConfig = DynamicsConfig(),
        run_name: Optional[str] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=log_dir) if run_name is None else SummaryWriter(log_dir=f"{log_dir}/{run_name}")

        # Hook state
        self._handles: list = []
        self._act_stats: Dict[str, Tuple[float, float, Optional[torch.Tensor]]] = {}

        # For update-norm
        self._prev_params: Optional[Dict[str, torch.Tensor]] = None

        self._register_activation_hooks()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.writer.close()

    # ----------------------------
    # Hooks
    # ----------------------------
    def _register_activation_hooks(self):
        def make_hook(name: str):
            def hook(_module, _inp, out):
                # Support tuple outputs
                x = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(x):
                    return

                with torch.no_grad():
                    xd = x.detach()
                    mu = float(xd.mean().item())
                    sd = float(xd.std(unbiased=False).item())

                    # Store a small CPU sample for hist logging
                    sample = None
                    if self.cfg.act_hist_interval is not None:
                        flat = xd.reshape(-1)
                        if flat.numel() > 0:
                            n = min(int(flat.numel()), int(self.cfg.act_hist_max_elems))
                            sample = flat[:n].to("cpu", non_blocking=True)

                    self._act_stats[name] = (mu, sd, sample)

            return hook

        for name, m in self.model.named_modules():
            if isinstance(m, self.cfg.activation_module_types):
                self._handles.append(m.register_forward_hook(make_hook(name)))

    # ----------------------------
    # Update snapshot helpers
    # ----------------------------
    @torch.no_grad()
    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in self.model.named_parameters()}

    @torch.no_grad()
    def _update_norm(self, prev: Dict[str, torch.Tensor]) -> float:
        diffs = []
        for n, p in self.model.named_parameters():
            if n in prev:
                diffs.append((p.detach() - prev[n]).float())
        return _global_l2_norm(diffs)

    # ----------------------------
    # Main API (call these in your loop)
    # ----------------------------
    def before_backward(self, step: int):
        # Take snapshot only when we intend to compute update-norm later
        if self.cfg.update_interval and (step % self.cfg.update_interval == 0):
            self._prev_params = self._snapshot_params()
        else:
            self._prev_params = None

    def after_backward(self, step: int):
        # Global norms
        params = [p for p in self.model.parameters()]
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]

        p_norm = _global_l2_norm(params)
        g_norm = _global_l2_norm(grads)
        ratio = g_norm / (p_norm + 1e-12)

        if step % self.cfg.scalar_interval == 0:
            self.writer.add_scalar("norm/param_global", p_norm, step)
            self.writer.add_scalar("norm/grad_global", g_norm, step)
            self.writer.add_scalar("ratio/grad_to_param", ratio, step)

        # Layer-wise norms (optional)
        if self.cfg.layerwise_interval and (step % self.cfg.layerwise_interval == 0):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    self.writer.add_scalar(f"layer_norm/param/{name}", float(p.detach().float().norm().item()), step)
                    if p.grad is not None:
                        self.writer.add_scalar(f"layer_norm/grad/{name}", float(p.grad.detach().float().norm().item()), step)

        # Optional gradient histograms (heavy!)
        if self.cfg.grad_hist_interval and (step % self.cfg.grad_hist_interval == 0):
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.grad is None:
                        continue
                    flat = p.grad.detach().reshape(-1)
                    if flat.numel() == 0:
                        continue
                    n = min(int(flat.numel()), int(self.cfg.grad_hist_max_elems))
                    self.writer.add_histogram(f"grad_hist/{name}", flat[:n].to("cpu", non_blocking=True), step)

    def after_step(self, step: int):
        # Update norm (if we snapped params)
        if self._prev_params is not None:
            u_norm = self._update_norm(self._prev_params)

            # compute param norm for ratio (cheap-ish)
            p_norm = _global_l2_norm([p for p in self.model.parameters()])
            if step % self.cfg.scalar_interval == 0:
                self.writer.add_scalar("norm/update_global", u_norm, step)
                self.writer.add_scalar("ratio/update_to_param", u_norm / (p_norm + 1e-12), step)

        # Activation logging (hooks populated _act_stats on forward pass)
        if self.cfg.act_interval and (step % self.cfg.act_interval == 0):
            for name, (mu, sd, _sample) in self._act_stats.items():
                self.writer.add_scalar(f"act_mean/{name}", mu, step)
                self.writer.add_scalar(f"act_std/{name}", sd, step)

        if self.cfg.act_hist_interval and (step % self.cfg.act_hist_interval == 0):
            for name, (_mu, _sd, sample) in self._act_stats.items():
                if sample is not None and sample.numel() > 0:
                    self.writer.add_histogram(f"act_hist/{name}", sample, step)

        # Clear cache (optional) to avoid holding references
        self._act_stats.clear()

    def log_optimizer(self, optimizer: torch.optim.Optimizer, step: int):
        if step % self.cfg.scalar_interval == 0:
            self.writer.add_scalar("lr", _get_lr(optimizer), step)

    def log_losses(
        self,
        step: int,
        loss: Union[torch.Tensor, float],
        per_sample_losses: Optional[torch.Tensor] = None,
        prefix: str = "train",
        log_hist: bool = True,
        max_hist_elems: int = 4096,
    ):
        loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        if step % self.cfg.scalar_interval == 0:
            self.writer.add_scalar(f"loss/{prefix}", loss_val, step)

        if per_sample_losses is not None:
            with torch.no_grad():
                ls = per_sample_losses.detach().float().reshape(-1)
                if ls.numel() == 0:
                    return
                if step % self.cfg.scalar_interval == 0:
                    self.writer.add_scalar(f"loss_{prefix}/per_sample_mean", float(ls.mean().item()), step)
                    self.writer.add_scalar(f"loss_{prefix}/per_sample_std", float(ls.std(unbiased=False).item()), step)

                if log_hist and (step % max(self.cfg.act_hist_interval or 200, 1) == 0):
                    n = min(int(ls.numel()), int(max_hist_elems))
                    self.writer.add_histogram(f"loss_{prefix}/per_sample_hist", ls[:n].to("cpu", non_blocking=True), step)

