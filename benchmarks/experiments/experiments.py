from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta

class Experiments(Factory, metaclass=SingletonMeta):

    @staticmethod
    def _str_as_repr():
        return True

    def get_models(self, experiment):
        models_trainers = self.get_meta(experiment)['args']['models_trainers']
        return [mt['model'] for mt in models_trainers]

    def get_models_trainers(self, experiment):
        return self.get_meta(experiment)['args']['models_trainers']

experiments = Experiments(__name__)

