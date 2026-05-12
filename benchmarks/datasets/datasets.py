from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta
from .utils import estimate_bytes

class Datasets(Factory, metaclass=SingletonMeta):

    def str(self, dataset, seed):
        return super().str(dataset) + '(seed=' + str(seed) + ')'

    def repr(self, dataset, seed):
        return super().repr(dataset) + '__seed_' + "{:2d}".format(seed).replace(' ', '0')

    def get_iosize(self, dataset):
        Class = self._get_class(dataset)
        args = self.get_property(dataset, 'args')
        return Class.get_iosize(**args)

    def get_length(self, dataset):
        Class = self._get_class(dataset)
        args = self.get_property(dataset, 'args')
        length = self.get_property(dataset, 'args')['length']
        if type(length) != int:
            raise Exception("This dataset has min_len != max_len")
        return length

    def get_maxlen(self, dataset):
        Class = self._get_class(dataset)
        args = self.get_property(dataset, 'args')
        length = self.get_property(dataset, 'args')['length']
        if type(length) == int:
            return length
        else:
            return length[1]

    def get_minlen(self, dataset):
        Class = self._get_class(dataset)
        args = self.get_property(dataset, 'args')
        length = self.get_property(dataset, 'args')['length']
        if type(length) == int:
            return length
        else:
            return length[0]

    def instantiate(self, dataset_name, *runtime_args, **runtime_kwargs):
        dataset = super().instantiate(dataset_name, *runtime_args, **runtime_kwargs)
        dataset.__postinit__()
        return dataset

    def preinstantiate(self, dataset_name, *runtime_args, **runtime_kwargs):
        return super().instantiate(dataset_name, *runtime_args, **runtime_kwargs)

    def estimate_sample_bytes(self, dataset):
        return estimate_bytes.estimate_sample_bytes(dataset)



datasets = Datasets(__name__)
