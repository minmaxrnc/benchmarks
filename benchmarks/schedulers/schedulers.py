from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta

from .nonescheduler import NoneScheduler
from .stepscheduler import StepLR

class Schedulers(Factory, metaclass=SingletonMeta):
    pass

schedulers = Schedulers(__name__, classes=[
    NoneScheduler,
    StepLR,
])

