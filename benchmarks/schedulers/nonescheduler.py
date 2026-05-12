from ..optimizers.optimizer import Optimizer
from .scheduler import Scheduler


class NoneScheduler(Scheduler):
    """A scheduler that leaves the lr unchanged."""

    def __init__(self, name, optimizer: Optimizer) -> None:
        super().__init__(name, optimizer=optimizer)


    def get_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


    def step(self):
        if not self._is_initial:
            raise Exception('This should never be called except during init')



    @staticmethod
    def get_properties():
        return {
            'step': 'never'
        }


