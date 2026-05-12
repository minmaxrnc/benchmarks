from abc import ABC, abstractmethod
from torch.optim.lr_scheduler import LRScheduler
from ..utils.properties import Properties


class Scheduler(ABC, Properties, LRScheduler):
    """Abstract class for learning-rate schedulers."""

    def __init__(
        self,
        name,
        *args,
        **kwargs
    ):
        self.name = name
        super().__init__(*args, **kwargs)


    @staticmethod
    def get_required_kwargs():
        return []


