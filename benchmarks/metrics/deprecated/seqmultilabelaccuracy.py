import torch
from torch import Tensor
from typing import Optional, Literal
from torcheval import metrics
from typing import TypeVar, Iterable

from ..utils.device import device
from ..definitions import IGNORE_INDEX
from .metric import Metric

TSelf = TypeVar("TSelf", bound="Metric")


class SeqMultilabelAccuracy(Metric, metrics.MultilabelAccuracy):
    """
    Streaming multilabel accuracy for **sequence** data with shape **[B, T, C]**.

    Parameters
    ----------
    threshold : float, default 0.5
        Probability threshold used to binarize predictions. Applied to probabilities.
        If logits are provided, they are first passed through sigmoid.
    criteria : {"hamming", "exact_match", "overlap", "contain", "belong"}, default "hamming"
        Accuracy definition passed through to `torcheval.metrics.MultilabelAccuracy`.
    ignore_index : float or int, default -1.0
        Sentinel value in `targets` indicating positions to **ignore**. To drop a padded
        time step, set **all C labels** at that step to this value.
    input_type : {"logits", "probs"}, default "logits"
        - "logits": inputs to `update` are logits and will be passed through sigmoid.
        - "probs": inputs are probabilities in [0, 1].

    Expected inputs to `update`
    ---------------------------
    preds : torch.Tensor, shape (B, T, C), dtype float
        Either raw logits or probabilities, depending on `input_type` (global or per-call override).
    targets : torch.Tensor, shape (B, T, C), dtype float/bool/long
        Multi-hot labels per token. Use **0/1** at labeled positions. Use `ignore_index`
        (e.g., -1) for unlabeled/padded positions (ideally all C at a padded step are -1).

    Behavior
    --------
    - Flattens (B, T, C) -> (N, C) where N = B*T.
    - Drops rows where **all C** labels are `ignore_index`.
    - Converts inputs to probabilities (sigmoid if logits; clamp to [0,1] if probs).
    - Clamps targets to {0,1}.
    - Forwards to `torcheval.metrics.MultilabelAccuracy` to update state.
    - `compute()` returns a scalar float accuracy as defined by `criteria`.
    """

    def __init__(
        self,
        name,
        threshold: float = 0.5,
        criteria: str = "hamming",
        input_type: Literal["logits", "probs"] = "logits",
    ):
        super().__init__(
            name,
            threshold=threshold,
            criteria=criteria,
            device = device
        )
        self.ignore_index = IGNORE_INDEX
        self.input_type = input_type

    def reset(self):
        self.base.reset()

    @torch.no_grad()
    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        input_type: Optional[Literal["logits", "probs"]] = None,
    ):
        """
        Update internal state with a batch.

        Parameters
        ----------
        preds : Tensor
            Either logits or probabilities (shape [B, T, C]).
        targets : Tensor
            Multi-hot targets with ignore_index for unlabeled/padded (shape [B, T, C]).
        input_type : {"logits","probs"} or None
            Optional per-call override; defaults to `self.input_type`.
        """
        B, T, C = preds.shape
        x = preds.reshape(-1, C)
        t = targets.reshape(-1, C).to(torch.float32)

        # keep tokens where at least one class is labeled (not ignore_index)
        row_valid = (t != self.ignore_index).any(dim=1)
        if not row_valid.any():
            return

        x = x[row_valid]
        t = t[row_valid].clamp_min_(0.0)  # {-1,0,1} -> {0,1}

        mode = input_type or self.input_type
        probs = x.sigmoid() if mode == "logits" else x.clamp_(0.0, 1.0)

        super().update(probs, t)
        # self.base.update(probs, t)


    @torch.no_grad()
    def compute(self) -> float:
        # out = self.base.compute()
        out = super().compute()
        return float(out.detach().cpu()) if torch.is_tensor(out) else float(out)


    @torch.inference_mode()
    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        raise Exception("Not implemented")

