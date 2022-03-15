import numpy as np
from scipy.special import gammainc

from vmodel.math import grid_positions


def random_positions(num, method, args):

    if method == 'poisson':
        # Sample initial positions using spherical poisson disk sampling
        return poisson_disk_spherical(args.radius_arena, args.spawn_distance,
                                      num_samples=num, method='volume',
                                      candidate='random')
    elif method == 'uniform':
        def sample_fn(low=0, high=args.radius_arena):
            return np.array(random_uniform_within_circle(low=low, high=high))
        # Sample random non-overlapping initial positions and velocities
        return poisson_disk_dart_throw(num, sample_fn, distance=args.spawn_distance)
    elif method == 'grid':
        # Sample positions on a grid with noise
        positions = grid_positions(num, args.spawn_distance, args.num_dims)
        positions += np.random.normal(scale=0.1, size=positions.shape)
        return positions
    else:
        raise ValueError(f'Unrecognized method: {args.spawn}')


# Uniform sampling in a hyperspere
# Based on Matlab implementation by Roger Stafford
# Can be optimized for Bridson algorithm by excluding all points within the r/2 sphere
def hypersphere_volume_sample(center, radius, k=1):
    ndim = center.size
    x = np.random.normal(size=(k, ndim))
    ssq = np.sum(x**2, axis=1)
    fr = radius * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(k, 1), (1, ndim))
    p = center + np.multiply(x, frtiled)
    return p


# Uniform sampling on the sphere's surface
def hypersphere_surface_sample(center, radius, k=1):
    ndim = center.size
    vec = np.random.standard_normal(size=(k, ndim))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    p = center + np.multiply(vec, radius)
    return p


def random_uniform_within_circle(low=0, high=1, size=None):
    """Sample random point within a unit circle
    Args:
        low (float): Radius of inner circle
        high (float): Radius of outer circle
    Returns:
        x (float): x-coordinate
        y (float): y-coordinate

    """
    distance = np.sqrt(np.random.uniform(low ** 2, high ** 2, size=size))
    bearing = np.random.uniform(0, 2 * np.pi, size=size)
    x = distance * np.cos(bearing)
    y = distance * np.sin(bearing)
    return x, y


def poisson_disk_square2d(radius, distance, num_samples):
    dims = (radius, radius)
    positions = poisson_disk(dims, distance, num_samples, method='surface')
    positions -= positions.mean(axis=0)  # center positions around origin
    return positions


def poisson_disk_spherical(radius: float, distance: float, num_samples=None, k: int = 30,
                           method='volume', candidate='random', ndim=2):
    """Fast spherical Poisson disk sampling in arbitrary dimensions
    Args:
        dims: sizes of the dimensions
        distance: minimum distance between samples
        num_samples: maximum number of samples to draw
        k: limit of sample to choose before rejection
        method: sample points on hypersphere `volume` or `surface`
        candidate: whether the next candidate point is the `first` or chosen at `random`
        ndim: number of dimensions
    Source: <https://dl.acm.org/doi/10.1145/1278780.1278807>
    Inspired by: <https://github.com/diregoblin/poisson_disc_sampling>
    """

    dims = np.array([radius * 2 for _ in range(ndim)], dtype=float)

    # Two distinct sampling methods
    #   volume: samples points within volume of radius to 2 * radius
    #   surface: samples points uniformly on the circle
    # Source: <http://extremelearning.com.au/an-improved-version-of-bridsons-algorithm-n-for-poisson-disc-sampling/>
    if method == 'volume':
        sample_fn = hypersphere_volume_sample
        sample_factor = 2
    elif method == 'surface':
        sample_fn = hypersphere_surface_sample
        eps = 0.001  # add epsilon to avoid rejection
        sample_factor = 1 + eps
    else:
        raise ValueError(f'Unrecognized method: {method}')

    # Two distinct methods of choosing the next candidate
    # random: select next candidate randomly, may lead to very non-circular shapes
    # first: select the first candidate, leads to more circular shapes
    #        can be seen as breadth-first sampling instead of random sampling
    if candidate not in ['random', 'first']:
        raise ValueError(f'Unrecognized candidate method: {candidate}')

    def in_limits(p):
        dist = np.linalg.norm(p - np.full(ndim, radius))
        return np.all(dist < radius)

    # Check if there are samples closer than squared distance to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)

        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True
        a = []
        for i in range(ndim):
            a.append(slice(indmin[i], indmax[i]))
        with np.errstate(invalid='ignore'):
            if np.any(np.sum(np.square(p - P[tuple(a)]), axis=ndim) < distance_squared):
                return True

    def add_point(p):
        points.append(p)
        indices = (p / cellsize).astype(int)
        P[tuple(indices)] = p

    # Cell size bounded by r/sqrt(n), so each grid cell will contain at most one sample
    cellsize = distance / np.sqrt(ndim)
    gridsize = (np.ceil(dims / cellsize)).astype(int)

    # Squared distance because we'll compare squared distance
    distance_squared = distance * distance

    # n-dimensional background grid for storing samples and accelerating spatial searches
    P = np.empty(np.append(gridsize, ndim), dtype=float)
    P.fill(np.nan)  # initialize empty cells with nan

    points = []
    # p = np.random.uniform(np.zeros(ndim), dims)
    p = np.zeros(ndim) + radius
    add_point(p)
    num_points = 1
    keepsampling = True
    while keepsampling and len(points) > 0:
        # Selecting 0th sample is breadth-first sampling, random otherwise
        i = np.random.randint(len(points)) if candidate == 'random' else 0
        p = points[i]
        del points[i]
        Q = sample_fn(np.array(p), distance * sample_factor, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
                if num_samples is not None:
                    num_points += 1
                    if num_points >= num_samples:
                        keepsampling = False  # max nsamples reached
                        break

    # Extract all non-nan samples
    samples = P[~np.isnan(P).any(axis=ndim)]

    # Subtract radius to center samples at the origin
    samples -= np.full((1, ndim), radius)

    return samples


def poisson_disk(dims: tuple, distance: float, num_samples=None, k: int = 30,
                 method='volume'):
    """Fast Poisson disk sampling in arbitrary dimensions
    Args:
        dims: sizes of the dimensions
        distance: minimum distance between samples
        num_samples: maximum number of samples to draw
        k: limit of sample to choose before rejection
        method: sample points on hypersphere `volume` or `surface`
    Source: <https://dl.acm.org/doi/10.1145/1278780.1278807>
    Inspired by: <https://github.com/diregoblin/poisson_disc_sampling>
    """

    dims = np.array(dims, dtype=float)
    ndim = dims.size

    # Two distinct sampling methods
    #   volume: samples points within volume of radius to 2 * radius
    #   surface: samples points uniformly on the circle
    # Source: <http://extremelearning.com.au/an-improved-version-of-bridsons-algorithm-n-for-poisson-disc-sampling/>
    if method == 'volume':
        sample_fn = hypersphere_volume_sample
        sample_factor = 2
    elif method == 'surface':
        sample_fn = hypersphere_surface_sample
        eps = 0.001  # add epsilon to avoid rejection
        sample_factor = 1 + eps
    else:
        raise ValueError(f'Unrecognized method: {method}')

    def in_limits(p):
        return np.all(np.zeros(ndim) <= p) and np.all(p < dims)

    # Check if there are samples closer than squared distance to the candidate "p"
    def in_neighborhood(p, n=2):
        indices = (p / cellsize).astype(int)
        indmin = np.maximum(indices - n, np.zeros(ndim, dtype=int))
        indmax = np.minimum(indices + n + 1, gridsize)

        # Check if the center cell is empty
        if not np.isnan(P[tuple(indices)][0]):
            return True
        a = []
        for i in range(ndim):
            a.append(slice(indmin[i], indmax[i]))
        with np.errstate(invalid='ignore'):
            if np.any(np.sum(np.square(p - P[tuple(a)]), axis=ndim) < distance_squared):
                return True

    def add_point(p):
        points.append(p)
        indices = (p / cellsize).astype(int)
        P[tuple(indices)] = p

    # Cell size bounded by r/sqrt(n), so each grid cell will contain at most one sample
    cellsize = distance / np.sqrt(ndim)
    gridsize = (np.ceil(dims / cellsize)).astype(int)

    # Squared distance because we'll compare squared distance
    distance_squared = distance * distance

    # n-dimensional background grid for storing samples and accelerating spatial searches
    P = np.empty(np.append(gridsize, ndim), dtype=float)
    P.fill(np.nan)  # initialize empty cells with nan

    points = []
    add_point(np.random.uniform(np.zeros(ndim), dims))
    num_points = 1
    keepsampling = True
    while keepsampling and len(points) > 0:
        i = np.random.randint(len(points))
        p = points[i]
        del points[i]
        Q = sample_fn(np.array(p), distance * sample_factor, k)
        for q in Q:
            if in_limits(q) and not in_neighborhood(q):
                add_point(q)
                if num_samples is not None:
                    num_points += 1
                    if num_points >= num_samples:
                        keepsampling = False  # max nsamples reached
                        break
    return P[~np.isnan(P).any(axis=ndim)]


def poisson_disk_dart_throw(n, sample_fn, distance):
    """Generate n random non-overlapping positions with a minimum distance.

    Args:
        n: Number of positions to generate
        sample_fn: Function that samples a random position
        distance: Minimum distance between positions

    Returns:
        positions (ndarray): Positions (n x ndim)
    """

    # First position is always valid
    positions = []
    positions.append(sample_fn())

    # Stop when we're sampling 100x as many positions desired
    maxiter = n * 1000

    iterations = 1
    while len(positions) < n:

        position = sample_fn()

        # Compute distance between sampled position and existing ones
        distances = np.linalg.norm(np.array(positions) - position, axis=1)

        # If the position is too close to any other, sample again
        if distances.min() > distance:
            positions.append(position)

        iterations += 1
        if iterations > maxiter:
            raise RuntimeError(f'Maximum number of iterations reached: {maxiter}')

    return np.array(positions)
