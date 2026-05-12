import torch
from .loss import Loss
import torch.nn.functional as F

_IGNORE_INDEX = -100

class CrossEntropyLoss(Loss):

    def __init__(self, name, reduction='mean'):
        super().__init__(name)
        self.reduction = reduction

    def clone(self, reduction=None):
        if reduction is None:
            reduction = self.reduction
        return CrossEntropyLoss(self.name, reduction)

    def __call__(self, logits, targets, mask):
        """
        # logits:  [B, T, C]
        # targets: [B, T]     (class indices)
        # mask:    [B, T]     (bool)
        """

        tgt = targets.masked_fill(~mask, _IGNORE_INDEX)
        return F.cross_entropy(
            logits.transpose(1, 2),  # [B, C, T]
            tgt,
            ignore_index=_IGNORE_INDEX,
            reduction=self.reduction
        )

        # B, T, C = logits.shape
        # loss = F.cross_entropy(logits.view(-1, C), targets.view(-1), reduction='none').view(B, T)
        # mask = mask.to(loss.dtype)
        # per_seq = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B]
        # return per_seq.mean()
        #
        # # log_probs = torch.log_softmax(logits, dim=-1)                          # [B, T, C]
        # # nll = -torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
        #
        # # mask = mask.to(nll.dtype)
        # # num = (nll * mask).sum()   # sum over unmasked tokens
        # # denom = mask.sum()
        #
        # # return num / torch.clamp(denom, min=1.0) # returns 0 if everything is masked
        #
        # # # # CrossEntropyLoss expects (N, C, ...), so move C to dim=1
        # # # per_token = super().__call__(
        # # #     logits.transpose(1, 2),  # -> [B, C, T]
        # # #     targets
        # # # )
        #
        # # # mask = mask.to(per_token.dtype)
        # # # num = (per_token * mask).sum()   # sum over unmasked tokens
        # # # denom = mask.sum()
        #
        # # # return num / torch.clamp(denom, min=1.0) # returns 0 if everything is masked

