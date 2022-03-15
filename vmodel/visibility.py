import numpy as np
from numba import njit

from vmodel.metrics import distance_matrix


def visibility_set(relative: np.ndarray, radius: float, distances: np.ndarray = None):
    """Check whether there are any occlusions
    Args:
        relative: relative positions
        radius: radius of agents
    Returns:
        visibility_set: boolean mask of visible relative positions
    """
    distances = np.linalg.norm(relative, axis=1) if distances is None else distances
    units = relative / distances[:, np.newaxis]  # project onto unit circle
    radii = np.full(len(relative), radius) if type(radius) is float else radius
    scaled = radii / distances  # shrink radii depending on distance
    rs = scaled[np.newaxis, :] + scaled[:, np.newaxis]  # add pairwise radii
    dmat = distance_matrix(units)  # pairwise distances between points on unit circle
    overlapmat = dmat < rs  # check if overlap...
    closermat = distances[np.newaxis, :] < distances[:, np.newaxis]  # ... and distance
    occmat = overlapmat & closermat
    occluded = occmat.sum(axis=1, dtype=bool)  # all values > 0 will be True!
    return ~occluded


def visibility_graph(positions: np.ndarray, radius: float):
    """Computes complete visibility graph (each agent to all others)
    Args:
        positions: absolute positions
        radius: radius of agents
    Returns:
        adj: boolean adjacency matrix of visibility graph (generally directed!)
    """
    N = len(positions)
    adj = np.zeros((N, N), dtype=bool)
    for a in range(N):
        pos_self = positions[a]
        pos_others = np.delete(positions, a, axis=0)
        pos_rel = pos_others - pos_self
        vis_others = visibility_set(pos_rel, radius)
        adj[a] = np.insert(vis_others, a, False)  # cannot see self, i.e. zero diag
    return adj


@njit
def is_occluding(p1: np.ndarray, r1: float, p2: np.ndarray, r2: float):
    """Check whether p1 occludes p2 with radius"""
    d1, d2 = np.linalg.norm(p1), np.linalg.norm(p2)  # compute distances
    u1, u2 = p1 / d1, p2 / d2  # project to unit circle
    rs1, rs2 = r1 / d1, r2 / d2  # scale radii by distance
    d = np.linalg.norm(u1 - u2)  # compute distance between projected points
    return d < rs1 + rs2 and (d1 - r1) <= (d2 - r2)
