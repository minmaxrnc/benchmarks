from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta

from .trainer import Trainer


class Trainers(Factory, metaclass=SingletonMeta):
    @staticmethod
    def _str_as_repr():
        return True


trainers = Trainers(__name__,classes=[
    Trainer
])
