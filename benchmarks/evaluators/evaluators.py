from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta


class Evaluators(Factory, metaclass=SingletonMeta):
    pass

evaluators = Evaluators(__name__)

