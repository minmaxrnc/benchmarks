import torch
from ..utils.properties import Properties
from ..optimizers import optimizers


class Optimizer(Properties, torch.optim.Optimizer):

    def __init__(self, name, **kwargs):
        self.name = name
        if 'lr' not in kwargs:
            raise ValueError('Optimizer requires `lr` parameter.')
        self.__lr = kwargs.get('lr')
        if '_weight_decay' in kwargs:
            self.__weight_decay = kwargs['_weight_decay']
            del kwargs['_weight_decay']
        else:
            self.__weight_decay = kwargs.get('weight_decay', None)
        super().__init__(**kwargs)


    def get_lr(self):
        return self.__lr


    def get_weight_decay(self):
        if not self.__weight_decay:
            raise Exception(
                'Attempted to get weight decay from optimizer.',
                'This optimizer does not have a weight decay.'
            )
        return self.__weight_decay


    @staticmethod
    def get_required_kwargs():
        return []


    def __str__(self):
        return optimizers.str(self.name)


    def __repr__(self):
        return optimizers.repr(self.name)

