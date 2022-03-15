import numpy as np


def human_readable(num, binary=True, precision=1, suffix=''):
    """Return human-readable version of bytes
    Args:
        num: number of bytes
        binary: if True, use base 1024, else use base 1000
        precision: precision of floating point number
        suffix: for instance 'b' or 'bytes'
    Inspired by: <https://stackoverflow.com/a/1094933/3075902>
    """
    units = ['B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']
    # units = [f'{u}i' for u in units] if binary else units
    base = 1024 if binary else 1000
    for unit in units:
        if abs(num) < base:
            return f'{num:.{precision}f}{unit}{suffix}'
        num /= base
    yotta = 'Yi' if binary else 'Y'
    return f'{num:.{precision}f}{yotta}{suffix}'


def clean_attrs(attrs):
    for k, v in attrs.items():
        if v is None:
            attrs[k] = 0  # required for NETCDF
        elif type(v) == bool:
            attrs[k] = int(v)  # required for NETCDF
        elif isinstance(v, (np.ndarray, np.generic)):
            attrs[k] = v.tolist()  # required for YAML
    return attrs
