import torch
from typing import Optional, List
from ..utils.device import device
from .metric import Metric
import warnings

class ChunkedTokenAccuracy(Metric):
    """
    Token-level cumulative accuracy across chunked inputs.

    Args:
        num_steps: total number of chunk positions

    Call update(logits, labels, mask, step=k) for k in [0, num_steps-1].
    Call compute() -> List[float] with cumulative accuracies up to each k.
    """
    def __init__(self, name):
        super().__init__(name)
        self.num_steps = None

    @torch.no_grad()
    def set_steps_and_reset(self, num_steps: int):
        if num_steps <= 0:
            raise ValueError("num_steps must be >= 1")
        self.num_steps = num_steps
        self._reset()

    @torch.no_grad()
    def _reset(self):
        if self.num_steps is None:
            raise Exception("Call set_steps_and_reset to reset")
        self.correct_per_step = torch.zeros(self.num_steps, device=device, dtype=torch.long)
        self.total_per_step   = torch.zeros(self.num_steps, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,  # [B, T, C]  (C = classes; last dim = classes)
        labels: torch.Tensor,  # [B, T]     (int indices)
        mask:   torch.Tensor,  # [B, T] bool; True/1 = include
        step:   int
    ):
        if not (0 <= step < self.num_steps):
            raise IndexError(f"step {step} out of range [0, {self.num_steps-1}]")

        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError("Expected logits [B, T, C] and labels [B, T].")

        if mask.dtype is not torch.bool:
            mask = mask.bool()
            warnings.warn(f"mask should have dtype torch.bool, got {mask.dtype}")

        preds   = logits.argmax(dim=-1)  # [B, T]
        correct = (preds == labels) & mask

        # Accumulate into fixed slots
        self.correct_per_step[step] += correct.sum().to(device=self.correct_per_step.device,
                                                        dtype=self.correct_per_step.dtype)

        self.total_per_step[step] += mask.sum().to(device=self.total_per_step.device,
                                                   dtype=self.total_per_step.dtype)


    @torch.no_grad()
    def compute(self) -> List[float]:
        """
        Returns per-step accuracies:
        per_step[k] = accuracy over tokens at step k.
        """
        mask = self.total_per_step > 0
        per_step = torch.full_like(
                self.total_per_step,
                float("nan"),
                dtype=torch.double
                )
        per_step[mask] = (
            self.correct_per_step[mask].to(torch.double) /
            self.total_per_step[mask].to(torch.double)
        )

        return per_step.tolist()


    @torch.no_grad()
    def compute_cumulative(self) -> List[float]:
        """
        Returns cumulative accuracies:
        cumulative[k] = accuracy over tokens from steps 0..k inclusive.
        """
        cum_correct = torch.cumsum(self.correct_per_step, dim=0)
        cum_total   = torch.cumsum(self.total_per_step, dim=0)

        # Produce NaN where cum_total == 0 (no tokens seen yet)
        cumulative = torch.where(
            cum_total > 0,
            cum_correct.to(torch.double) / cum_total.to(torch.double),
            torch.full_like(cum_correct, float("nan"), dtype=torch.double)
        )

        return cumulative.tolist()

