import sys
from tqdm.auto import tqdm as _tqdm, trange as _trange


class _ForceTTY:
    def __init__(self, f): self._f = f
    def write(self, s): return self._f.write(s)
    def flush(self): return self._f.flush()
    def isatty(self): return True


_file = sys.stderr if sys.stderr.isatty() else _ForceTTY(sys.stderr)


class tqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('file', _file)
        kwargs.setdefault('dynamic_ncols', True)
        super().__init__(*args, **kwargs)


def trange(*args, **kwargs):
    kwargs.setdefault('file', _file)
    kwargs.setdefault('dynamic_ncols', True)
    return _trange(*args, **kwargs)
