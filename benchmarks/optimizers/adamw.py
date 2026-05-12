from typing import List, Optional, Tuple
from torch import optim
from .optimizer import Optimizer


class AdamW(Optimizer, optim.AdamW):

    def __init__(self, name, decay, no_decay, lr, weight_decay, **kwargs):
        params = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        super().__init__(name, params=params, lr=lr, _weight_decay=weight_decay, **kwargs)


    @staticmethod
    def get_properties():
        return {
            'requires_decay_groups': True
        }


