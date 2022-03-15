"""
Reynolds flocking algorithm (aka Boids)
Source: <https://dl.acm.org/doi/10.1145/37402.37406>
"""
import numpy as np


def flock(positions: np.ndarray, distances: np.ndarray = None, separation_gain=1.0,
          cohesion_gain=1.0) -> np.ndarray:
    """Computes Reynolds command (separation + cohesion)
    Args:
        positions: relative positions of agents (N x D)
        distances: precomputed relative distances (N x 1) (optional)
        separation_gain: separation gain
        cohesion_gain: cohesion gain
    Returns:
        command: Reynolds command (D)
    """
    # Compute separation commmand (inversely proportional to distance)
    distances = np.linalg.norm(positions, axis=1) if distances is None else distances
    dist_inv = positions / distances[:, np.newaxis] ** 2
    separation = -separation_gain * dist_inv.sum(axis=0)

    # Compute cohesion command (proportional to distance)
    cohesion = cohesion_gain * positions.mean(axis=0)

    return separation + cohesion


def infer_separation_gain3(distance: float) -> float:
    """Compute separation gain from desired distance
    Args:
        distance: Desired minimum inter-agent distance
    Returns:
        separation_gain: Separation gain to produce (at least) the desired distance
    Note: This is the analytical equilibrium distance of the Reynolds equations for N = 3
    """
    return distance ** 2 / 2
