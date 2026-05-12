from torch.optim import lr_scheduler
from .scheduler import Scheduler

class StepLR(Scheduler, lr_scheduler.StepLR):
    """Wrapper of StepLR scheduler."""

    def __init__(
            self,
            name,
            *args,
            **kwargs
            ):
        super().__init__(name, *args, **kwargs)


    @staticmethod
    def get_properties():
        return {
            'step': 'epoch'
        }

