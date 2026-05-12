from abc import ABC, abstractmethod

class Loss(ABC):

    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_required_kwargs():
        return []

