import sys
from tqdm import tqdm as _tqdm, trange as _trange


def _disabled():
    if not sys.stderr.isatty():
        return True
    if 'google.colab' in sys.modules:
        return True
    return False


class tqdm(_tqdm):
    def __init__(self, iterable=None, *args, **kwargs):
        if _disabled():
            desc = kwargs.get('desc')
            if desc:
                print(f"{desc}...", flush=True)
            kwargs['disable'] = True
        else:
            kwargs.setdefault('dynamic_ncols', True)
        super().__init__(iterable, *args, **kwargs)

    @staticmethod
    def write(s, file=None, end='\n', nolock=False):
        if _disabled():
            print(s, end=end, flush=True)
        else:
            _tqdm.write(s, file=file, end=end, nolock=nolock)


def trange(*args, **kwargs):
    if _disabled():
        desc = kwargs.get('desc')
        if desc:
            print(f"{desc}...", flush=True)
        kwargs['disable'] = True
    else:
        kwargs.setdefault('dynamic_ncols', True)
    return _trange(*args, **kwargs)
