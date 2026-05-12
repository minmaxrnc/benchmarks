from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta

from .crossentropyloss import CrossEntropyLoss

class Losses(Factory, metaclass=SingletonMeta):
    pass

losses = Losses(__name__,classes=[
    CrossEntropyLoss
])

