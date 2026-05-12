import torch
from typing import Optional, Iterable
from torcheval import metrics
from ..utils.device import device
from .metric import Metric


class MaskedSeqMultiLabelAccuracy(Metric, metrics.Metric):
    """
    Accuracy over all bits (includes TNs) for multi-label sequences with masking.
    Accepts logits; applies sigmoid+threshold.

    update(logits, targets, mask)
        logits:  (..., C) float tensor
        targets: (..., C) bool/int/float tensor in {0,1}
        mask:    (...)   bool/int/float tensor where True/1 = valid; shape must match logits.shape[:-1]
    compute() -> scalar tensor with accuracy
    """

    def __init__(
        self,
        name,
        *,
        threshold: float = 0.5,
    ) -> None:

        super().__init__(name, device=device)
        self.threshold = threshold

        # register metric state
        self._add_state("correct", torch.tensor(0.0))
        self._add_state("total", torch.tensor(0.0))

        self.to(device)

    @torch.inference_mode()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> None:
        if logits.dim() < 2:
            raise ValueError("Expected logits with shape (..., C).")
        if logits.shape != targets.shape:
            raise ValueError(f"logits and targets must have the same shape; got {logits.shape} vs {targets.shape}.")

        # keep metric states on the same device as inputs
        # self.to(device)

        C = logits.size(-1)

        # mask handling -> shape (...), then expand to (..., C)
        # valid = mask.to(dtype=torch.bool, device=device)
        if mask.shape != logits.shape[:-1]:
            raise ValueError(f"mask must have shape {logits.shape[:-1]}, got {mask.shape}.")

        if mask.dtype is not torch.bool:
            raise TypeError(f"mask must be dtype torch.bool, got {mask.dtype}")

        valid_bits = mask.unsqueeze(-1).expand_as(logits)

        # targets -> bool
        t = (targets.to(device=device).float() > 0.5)

        # predictions from logits
        p = (logits.sigmoid() >= self.threshold)

        eq = (p == t)
        correct = (eq & valid_bits).sum().to(dtype=torch.float)
        total = valid_bits.sum().to(dtype=torch.float)

        self.correct += correct
        self.total += total

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        # return NaN if no valid elements were seen
        if self.total.item() == 0:
            return torch.tensor(float("nan"), device=device)
        return self.correct / self.total

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable["MaskedSeqMultiLabelAccuracy"]) -> None:
        for m in metrics:
            self.correct += m.correct.to(device)
            self.total += m.total.to(device)

    @torch.inference_mode()
    def reset(self) -> None:
        self.correct.zero_()
        self.total.zero_()

