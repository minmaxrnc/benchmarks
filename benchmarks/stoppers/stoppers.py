
from .emastopper import EmaStopper


def instantiate(name):
    if name != 'emastopper':
        raise ValueError("The only stopper is `emastopper'")
    return EmaStopper()

