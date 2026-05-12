from copy import deepcopy
from . import meta

META = meta.load('config')

def get(key):
    return deepcopy(META[key])

