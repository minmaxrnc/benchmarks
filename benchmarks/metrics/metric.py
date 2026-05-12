from abc import ABC, abstractmethod

class Metric(ABC):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__()

    @staticmethod
    def get_required_args():
        return []

