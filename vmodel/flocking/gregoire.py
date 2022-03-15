"""
Moving and staying together without a leader
Source: <https://www.sciencedirect.com/science/article/pii/S0167278903001027>
"""
import numpy as np


def flock(positions: np.ndarray, distance: float, perception_radius: float, d_c: float):
    d_e, d_a = distance, perception_radius
    distances = np.linalg.norm(positions, axis=1)
    units = positions / distances[:, np.newaxis]
    commands = []
    for i in range(len(positions)):
        command = units[i] * potential(distances[i], d_c, d_e, d_a)
        commands.append(command)
    return np.sum(commands, axis=0)


def potential(r, r_c, r_e, r_a):
    """Potential (Eq. 4)
    Args:
        r: distance between agent i and j
        r_c: hard-core repulsion distance
        r_e: equilibrium (preferred) distance
        r_a: maximum interaction range
    Returns:
        potential: attractive/repulsive potential
    """
    if r < r_c:
        return -1  # should be -inf
    elif r < r_a:
        return 0.25 * (r - r_e) / (r_a - r_c)
    else:
        return 1
