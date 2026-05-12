from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta


class Evaluations(Factory, metaclass=SingletonMeta):
    @staticmethod
    def _str_as_repr():
        return True

evaluations = Evaluations(__name__)

