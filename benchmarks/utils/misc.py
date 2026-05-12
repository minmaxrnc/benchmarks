from itertools import chain, combinations
from pathlib import Path
import os
import warnings
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Union

def powerset(iterable):
    s = list(iterable)  # keeps original order
    res = []
    for sub in chain.from_iterable(combinations(s, r) for r in range(len(s)+1)):
        res.append(sorted(list(sub)))
    return res

def format_floats_as_strings(obj, fmt="{:.2f}"):
    if isinstance(obj, float):
        return fmt.format(obj)           # -> "2.00"
    if isinstance(obj, dict):
        return {k: format_floats_as_strings(v, fmt) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_floats_as_strings(x, fmt) for x in obj]
    if isinstance(obj, tuple):
        return tuple(format_floats_as_strings(x, fmt) for x in obj)
    return obj

def format_for_filename(obj, fmt="{:.2e}"):
    if isinstance(obj, float):
        return fmt.format(obj)
    if isinstance(obj, dict):
        return {k: format_for_filename(v, fmt) for k, v in obj.items()}
    if isinstance(obj, list):
        return '-'.join([str(format_for_filename(x, fmt)) for x in obj])
    if isinstance(obj, tuple):
        return tuple(format_for_filename(x, fmt) for x in obj)
    return obj


def str_class(class_name, params):
    s_params = []
    params = format_floats_as_strings(params)
    for param in sorted(params.keys()):
        s_params.append(str(param) + '=' + str(params[param]))
    s_params = ', '.join(s_params)
    return class_name + '(' + s_params + ')'

def repr_class(class_name, params):
    s_params = []
    params = format_for_filename(params)
    for param in sorted(params.keys()):
        s_param = str(param).replace('_', '-')
        val = params[param]
        s_val = str(val).replace(' ', '')
        s_params.append(s_param + '_' + s_val)
    s_params = '__'.join(s_params)
    return class_name + '___' + s_params


def bytes_to_mb_or_gb(n_bytes: int, *, style: str = "iec", precision: int = 2) -> str:
    """
    Convert bytes to a readable MB/GB (or MiB/GiB) string, choosing the larger
    convenient unit (GB if >= 1 GB, else MB).

    Parameters
    ----------
    n_bytes : int
        Size in bytes (must be >= 0).
    style : {"iec","si"}
        - "iec": powers of 1024 with labels MiB/GiB.
        - "si" : powers of 1000 with labels MB/GB.
    precision : int
        Decimal places in the formatted number.

    Returns
    -------
    str : e.g., "950.32 MiB", "2.50 GiB", "900.00 MB", "50.00 GB"
    """
    if n_bytes < 0:
        raise ValueError("n_bytes must be >= 0")

    base = 1024 if style == "iec" else 1000
    mb = base ** 2
    gb = base ** 3
    unit_mb, unit_gb = ("MiB", "GiB") if style == "iec" else ("MB", "GB")

    if n_bytes >= gb:
        value, unit = n_bytes / gb, unit_gb
    else:
        value, unit = n_bytes / mb, unit_mb

    return f"{value:.{precision}f} {unit}"



def add_extension(path: Union[str,Path], ext: str, *, replace: bool = False, compound: bool = False) -> str:
    """
    Ensure a filename has a given extension.

    Args:
        path: File path or name.
        ext: Extension to ensure (with or without leading dot), e.g. "pdf" or ".pdf".
        replace:
            - False (default): keep existing extension(s); only add if missing.
            - True: replace the last extension; if compound=True, strip all extensions first.
        compound:
            - False (default): treat only the last extension (e.g., ".gz" in "archive.tar.gz").
            - True: consider all suffixes; if replace=True, remove all before adding `ext`.

    Returns:
        New path as a string (does not touch the filesystem).
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    p = Path(path)

    # Hidden files like ".env" should become ".env.ext" when not replacing
    if not replace and p.suffix == "" and p.name.startswith("."):
        return str(p.with_name(p.name + ext))

    if replace:
        q = p
        if compound:
            # strip all suffixes
            while q.suffix:
                q = q.with_suffix("")
            return str(q.with_suffix(ext))
        # replace only the last suffix (or add if none)
        return str(q.with_suffix(ext))

    # Not replacing: only add if it's not already there
    if compound:
        return str(p if ext in p.suffixes else p.with_name(p.name + ext))
    else:
        return str(p if p.suffix == ext else p.with_name(p.name + ext))

def pjoin(*args, **kwargs):
    if 'ext' in kwargs:
        ext = kwargs['ext']
        if ext not in ['txt', 'json', 'csv']:
            warnings.warn(f"Provided extension '{ext}'")
        return add_extension(os.path.join(*args), kwargs['ext'])
    else:
        return os.path.join(*args)


def zip_dir(src_dir, zip_path):
    src_dir = Path(src_dir)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir))

def unzip(zip_path, dst_dir):
    with ZipFile(zip_path) as zf:
        zf.extractall(dst_dir)
