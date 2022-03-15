import numpy as np
from numba import njit


def limit_norm(x, limit):
    """Limits the norm of n-dimensional vector
    Args:
        x (ndarray): Vector
        limit (float): Limit norm
    Returns:
        x (ndarray): Vector with norm limited to limit
    """
    norm = np.linalg.norm(x)
    if norm > limit:
        x /= norm  # make x unit vector
        x *= limit  # scale x by limit
    return x


@njit
def cartesian2polar(x, y):
    """Converts cartesian to polar coordinates

    Args:
        x (float): x-coordinate
        y (float): y-coordinate

    Returns:
        distance (float): range/distance
        bearing (float): bearing angle [rad]
    """
    distance = np.sqrt(x ** 2 + y ** 2)
    bearing = np.arctan2(y, x)
    return distance, bearing


@njit
def polar2cartesian(distance, bearing):
    """Converts polar to cartesian coordinates

    Args:
        distance (float): range/distance
        bearing (float): bearing angle [rad]

    Returns:
        x (float): x-coordinate
        y (float): y-coordinate
    """
    x = distance * np.cos(bearing)
    y = distance * np.sin(bearing)
    return x, y


@njit
def wrap_angle(angle):
    """Wraps angle around to remain in range [-pi, pi]

    Args:
        angle (float): Angle [rad]

    Returns:
        wrapped_angle (float): Angle in [-pi, pi] [rad]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def grid_positions(n: int, distance: float = 1, ndim: int = 2) -> np.ndarray:
    """Generate positions on a grid n-d grid
    Args:
        n: number of positions to generate
        distance: distance between positions
        ndim: number of dimensions
    Returns:
        positions: (n x ndim) positions
    """
    assert n > 0
    assert ndim >= 1
    assert distance > 0

    # Compute side length of hypercube
    sidelen = int(np.ceil(np.sqrt(n)))

    # Get indices of each dimension
    indices = tuple(np.arange(sidelen) for _ in range(ndim))

    # Create n-d cooridinate array from indices
    positions = np.array(np.meshgrid(*indices), dtype=float).T.reshape(-1, ndim)

    # Remove superfluous positions (if any)
    positions = positions[:n]

    # Center positions at zero point
    positions -= ((sidelen - 1) / 2)

    # Scale positions by distance
    positions *= distance

    return positions


def filter_by_angle(positions, angle):
    """Filter positions by spatially balanced topological distance
    Args:
        positions: relative positions
        angle: minimum angle between relative positions
    Return:
        mask: boolean mask of selected positions
    Source: <https://royalsocietypublishing.org/doi/full/10.1098/rsfs.2012.0026>
    """
    pos = positions.copy()
    N = len(pos)
    if N <= 1:
        return np.zeros(N, dtype=bool)
    dist = np.linalg.norm(pos, axis=1)
    idx = dist.argsort()
    idx_orig = idx.argsort()
    pos, dist = pos[idx], dist[idx]
    unit = pos / dist[:, np.newaxis]
    deleted = set()
    for i in range(N):
        for j in range(i + 1, N):
            angle_between = np.arccos(np.dot(unit[i], unit[j]))
            if angle_between < angle:
                deleted.add(j)
    indices = np.array(list(set(range(N)) - deleted), dtype=int)
    mask = np.zeros(N, dtype=bool)
    mask[indices] = True
    return mask[idx_orig]
