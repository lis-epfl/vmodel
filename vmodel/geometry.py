from collections import defaultdict
from itertools import combinations

import numpy as np
from numba import njit
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import distance_matrix as compute_distance_matrix


def distance_matrix(positions: np.ndarray):
    return compute_distance_matrix(positions, positions)


def convex_hull(positions: np.ndarray):
    """Convenience function for convex hull
    Args:
        positions (agent x space): Positions
    Return:
        hull (indices): Indices of positions belonging to the convex hull
    """
    hull = np.zeros(len(positions), dtype=np.bool_)
    indices = ConvexHull(positions).vertices
    hull[indices] = True
    return hull


def voronoi_neighbors(positions: np.ndarray) -> list:
    """Compute a list of adjacent neighbors in a Voronoi sense
    Args:
        positions (agent x space): positions
    Return:
        voronoi: agent-indexed list of neighbor index list
    """
    tri = Delaunay(positions)
    neighbors = defaultdict(set)
    for p in tri.vertices:
        for i, j in combinations(p, 2):
            neighbors[i].add(j)
            neighbors[j].add(i)
    return [list(neighbors[k]) for k in range(len(positions))]


def subtended_angle_circle(distance, radius):
    """Find subtended angle of a circle with given distance and radius
    Args:
        distance: Relative distance to the center
        radius: Radius of the circle
    Return:
        subtended_angle: Subtended angle [rad]
    """
    radius_unit = radius / distance  # project radius onto unit circle
    subtended_angle = np.arcsin(radius_unit) * 2.
    return subtended_angle


def distance_for_subtended_angle(angle, radius):
    """Find distance to a circle with a given radius that produces a given subtened angle
    Args:
        angle: Subtended angle [rad]
        radius: radius of the circle
    Return:
        distance: distance to center of the circle
    """
    return radius / np.sin(angle / 2.)


@njit
def is_angle_in_interval(angle, interval):
    """Check if angle is in the interval
    Args:
        angle (float): Angle [rad]
        interval tuple(float, float): Angle interval (low, high) [rad]
    Returns:
        True, if the angle is in the given interval
    """
    lo, hi = interval
    if lo < hi:
        return angle >= lo and angle <= hi  # normal case
    else:
        return angle >= lo or angle <= hi  # interval spans accros quadrant 1 and 4


@njit
def tangent_points_to_circle(center, radius):
    """Computes the tangent intersection points of a circle (as seen from the origin)

    Args:
        center (x, y): Target point (center of circle)
        radius (float): Radius of cicle

    Returns:
        points ((x, y), (x, y)): Tangent points, or None if no such points exist
    Source: https://en.wikipedia.org/wiki/Tangent_lines_to_circles#With_analytic_geometry
    """
    x, y = center
    d2 = x ** 2 + y ** 2
    r = radius
    r2 = radius ** 2
    if d2 < r2:
        return None  # no tangent points exist if point within circle radius
    r2_d2 = r2 / d2
    first_x = r2_d2 * x
    first_y = r2_d2 * y
    r_d2_sqrt_d2_r2 = r / d2 * np.sqrt(d2 - r2)
    second_x = r_d2_sqrt_d2_r2 * -y
    second_y = r_d2_sqrt_d2_r2 * x
    pt1_x = first_x + second_x
    pt1_y = first_y + second_y
    pt2_x = first_x - second_x
    pt2_y = first_y - second_y
    return (x - pt1_x, y - pt1_y), (x - pt2_x, y - pt2_y)


def lattice2d_rect(shape, basis):
    """Generate rectangular lattice with shape and basis vectors
    Args:
        shape (tuple): (width, height)
        basis (list): list of basis vectors v1 and v2
    Source: <https://stackoverflow.com/a/6145068/3075902>
    """
    # Get the lower limit on the cell size.
    dx_cell = max(abs(basis[0][0]), abs(basis[1][0]))
    dy_cell = max(abs(basis[0][1]), abs(basis[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = shape[0] // dx_cell
    ny = shape[1] // dy_cell
    # Generate a square lattice, with too many points.
    # Here I generate a factor of 4 more points than I need, which ensures
    # coverage for highly sheared lattices.  If your lattice is not highly
    # sheared, than you can generate fewer points.
    x_sq = np.arange(-nx, nx, dtype=float)
    y_sq = np.arange(-ny, nx, dtype=float)
    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    x_lattice = basis[0][0] * x_sq + basis[1][0] * y_sq
    y_lattice = basis[0][1] * x_sq + basis[1][1] * y_sq
    # Trim to fit in box.
    mask = ((x_lattice < shape[0] / 2.0) & (x_lattice > -shape[0] / 2.0))
    mask = mask & ((y_lattice < shape[1] / 2.0) & (y_lattice > -shape[1] / 2.0))
    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]
    # Make output compatible with original version.
    out = np.empty((len(x_lattice), 2), dtype=float)
    out[:, 0] = x_lattice
    out[:, 1] = y_lattice
    return out


def lattice2d_circ(radius, basis):
    """Generate circular lattice with radius and basis vectors
    Args:
        radius (float): radius of the lattice
        basis (list): list of basis vectors v1 and v2
    """
    shape = (radius * 2, radius * 2)
    points = lattice2d_rect(shape, basis)
    distances = np.linalg.norm(points, axis=1)
    mask = distances < radius
    points = points[mask]
    return points
