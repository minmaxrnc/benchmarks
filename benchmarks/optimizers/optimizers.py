from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta


class Optimizers(Factory, metaclass=SingletonMeta):
    pass

optimizers = Optimizers(__name__)

