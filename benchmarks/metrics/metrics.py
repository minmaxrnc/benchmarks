from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta

from .tokenaccuracy import TokenAccuracy
from .chunkedtokenaccuracy import ChunkedTokenAccuracy

class Metrics(Factory, metaclass=SingletonMeta):
    pass

metrics = Metrics(__name__,classes=[
    TokenAccuracy,
    ChunkedTokenAccuracy
])

