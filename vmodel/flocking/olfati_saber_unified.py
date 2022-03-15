"""
Flocking with obstacle avoidance: cooperation with limited communication in mobile networks
Conference paper (CDC): <https://ieeexplore.ieee.org/document/1272912>
Technical report: <https://authors.library.caltech.edu/28025/>
"""
import numpy as np
from numba import njit


def flock(positions: np.ndarray, distance: float, perception_radius: float,
          a=1.0, b=5.0):
    """Olfati saber algorithm (only gradient based term) (Eq. 23, first term)
    Args:
        positions: Relative positions of other agents
        distance: Desired inter-agent distance
        perception_radius: Cutoff at perception radius
    """
    d, r = distance, perception_radius
    assert d > 0
    assert r > d
    assert a > 0
    assert b > a
    ds = np.linalg.norm(positions, axis=1)
    us = positions / ds[..., np.newaxis]
    return np.sum([phi(ds[i] - d, d, r, a, b) * us[i] for i in range(len(us))], axis=0)


@njit
def phi(z, d, r, a, b, delta=0.2):
    x = (z + d)
    term1 = rho_grad(x, r, delta=delta) * psi(z, a, b)
    term2 = rho(x, r, delta=delta) * psi_grad(z, a, b)
    return term1 + term2


@njit
def psi_hat(z, d, r, a, b, delta=0.0):
    return rho((z + d), r, delta) * psi(z, a, b)


@njit
def psi(z: float, a: float, b: float) -> float:
    """Penalty function with uniformly bounded derivative (Eq. 20)
    Args:
        z: Relative distance
        a: Cohesion strength
        b: Separation strength
    """
    c = np.abs(a - b) / (2 * np.sqrt(a * b))
    return ((a + b) / 2) * (np.sqrt(1 + (z + c) ** 2) - np.sqrt(1 + c ** 2)) + ((a - b) / 2) * z


@njit
def psi_grad(z: float, a: float, b: float) -> float:
    """Derivative of the penalty function is an asymmetric sigmoidal function (Eq. 21)
    Args:
        z: Relative distance
        a: Cohesion strength
        b: Separation strength
    """
    c = np.abs(a - b) / (2 * np.sqrt(a * b))
    return ((a + b) / 2) * ((z + c) / np.sqrt(1 + (z + c) ** 2)) + ((a - b) / 2)


@njit
def rho(z: float, r: float = 1.0, delta: float = 0.0) -> float:
    """Influence map that varies smoothly betwenn 0 and 1 (Eq. 4)
    Args:
        z: Relative distance
        r: perception radius
        delta: Cutoff after which the function starts decreasing
    """
    if z < delta * r:
        return 1
    elif z < r:
        return 0.5 * (1 + np.cos(np.pi * (((z / r) - delta) / (1 - delta))))
    else:
        return 0


@njit
def rho_grad(z: float, r: float = 1.0, delta: float = 0.0) -> float:
    if z < delta * r:
        return 0
    elif z < r:
        return (1 / r) * -0.5 * np.pi * np.sin(np.pi * (((z / r) - delta) / (1 - delta))) / (1 - delta)
    else:
        return 0
