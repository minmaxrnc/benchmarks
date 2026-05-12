from .datasets import datasets

from .sequences import Sequences
from .latching import Latching
from .inductionheads import InductionHeads

datasets.add_classes(
    Sequences,
    Latching,
    InductionHeads,
)
