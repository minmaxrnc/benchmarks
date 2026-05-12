from torch import optim
from .optimizer import Optimizer


class Adam(Optimizer, optim.Adam):

    def __init__(
        self,
        name,
        **kwargs
    ):
        super().__init__(name, **kwargs)


    @staticmethod
    def get_properties():
        return {
            'requires_decay_groups': False
        }

