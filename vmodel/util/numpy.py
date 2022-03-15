import numpy as np


def toindices(x: np.ndarray) -> np.ndarray:
    """Transform mask array to array of indices:
    Example:
        [False, True, True, False] -> [1, 2]
    """
    return np.where(x)[0]
