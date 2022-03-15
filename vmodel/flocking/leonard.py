"""
Virtual Leaders, Artificial Potentials and Coordinated Control of Groups
Paper: <https://ieeexplore.ieee.org/document/980728>
"""
import numpy as np


def flock(positions: np.ndarray, distance: float, perception_radius: float, a=1):
    ds = np.linalg.norm(positions, axis=1)
    us = positions / ds[:, np.newaxis]
    d_0, d_1 = distance, perception_radius
    return np.sum([action(ds[i], d_0, d_1, a) * us[i] for i in range(len(ds))], axis=0)


def action(r_ij, d_0, d_1, a=1.0) -> float:
    return a * (- (d_0 / r_ij ** 2) + (1 / r_ij)) if r_ij < d_1 else 0.0


def potential(r_ij, d_0, d_1, alpha=1.0):
    if r_ij < d_1:
        return alpha * (np.log(r_ij) + (d_0 / r_ij))
    else:
        return alpha * (np.log(d_1) + (d_0 / d_1))
