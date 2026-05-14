import sys
from tqdm.auto import tqdm as _tqdm, trange as _trange

_IS_COLAB = 'google.colab' in sys.modules
_IS_TTY = sys.stderr.isatty() and not _IS_COLAB


class tqdm(_tqdm):
    def __init__(self, iterable=None, *args, **kwargs):
        if not _IS_TTY:
            desc = kwargs.get('desc')
            if desc:
                print(f"{desc}...", flush=True)
            kwargs['disable'] = True
        else:
            kwargs.setdefault('dynamic_ncols', True)
        super().__init__(iterable, *args, **kwargs)

    @staticmethod
    def write(s, file=None, end='\n', nolock=False):
        if _IS_TTY:
            _tqdm.write(s, file=file, end=end, nolock=nolock)
        else:
            print(s, end=end, flush=True)


def trange(*args, **kwargs):
    if not _IS_TTY:
        desc = kwargs.get('desc')
        if desc:
            print(f"{desc}...", flush=True)
        kwargs['disable'] = True
    else:
        kwargs.setdefault('dynamic_ncols', True)
    return _trange(*args, **kwargs)
