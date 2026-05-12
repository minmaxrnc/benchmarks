from .optimizers import optimizers

from .adamw import AdamW
from .adam  import Adam

optimizers.add_classes(
    AdamW,
    Adam,
)
