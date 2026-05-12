from abc import abstractmethod

class Properties:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_property(self, property_name):
        raise Exception('Not implemented')

