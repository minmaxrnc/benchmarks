from ..utils.factory import Factory
from ..utils.singleton import SingletonMeta
from .utils import estimate_model_bytes


class Models(Factory, metaclass=SingletonMeta):

    def estimate_bytes(self, model):
        return estimate_model_bytes(model)


models = Models(__name__, classes=[])


def register_model(cls):
    """Register a model class so it can be used in experiments.

    Call this from your model package's __init__.py:

        from benchmarks import register_model
        from .mymodel import MyModel_LM
        register_model(MyModel_LM)

    The class name (cls.__name__) must match the 'class' field in meta/models.yaml.
    """
    models.add_class(cls)
