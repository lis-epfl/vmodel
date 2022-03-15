"""
Olfati-Saber algorithm (Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory)
Source: <https://ieeexplore.ieee.org/document/1605401>
This version is simplified in the sense that it does not use the sigma norm.
"""
import numpy as np
from numba import njit
from scipy import integrate


def flock(positions: np.ndarray, distance: float, perception_radius: float,
          a=5.0, b=5.0, eps=0.1, h=0.2):
    """Olfati saber algorithm (only gradient based term) (Eq. 23, first term)
    Args:
        positions: relative positions of other agents
        distance: desired inter-agent distance
        perception_radius: cutoff at perception radius
        h: parameter of the bump function
    """
    d, r = distance, perception_radius
    assert d > 0
    assert r >= d
    assert eps > 0
    assert h >= 0 and h <= 1
    ds = np.linalg.norm(positions, axis=1)
    us = positions / ds[:, np.newaxis]
    terms = [phi_alpha(z, d, r, a, b, h) * n for z, n in zip(ds, us)]
    return np.sum(terms, axis=0)


@njit
def phi_alpha(z, d, r, a, b, h=0.2):
    """Action function (Eq. 15, first line)"""
    return rho_h(z / r, h) * phi(z - d, a, b)


@njit
def rho_h(z, h=0.2):
    """Bump function that varies smoothly from 1 to 0 over the (0, 1) interval (Eq. 10)
    Args:
        z (float): Relative distance
        h (float): Offset from which rho_h starts decreasing
    Returns:
        (float): value that varies smoothly on the interval (1 to 0)
    """
    if z < h:
        return 1
    elif z < 1:
        return 0.5 * (1 + np.cos(np.pi * ((z - h) / (1 - h))))
    else:
        return 0


@njit
def phi(z, a, b):
    """Action function helper (Eq. 15, second line)
    Args:
        z (float): relative distance
        a (float): a parameter (a > 0)
        b (float): b parameter (b >= a)
    """
    c = np.abs(a - b) / np.sqrt(4 * a * b)
    return 0.5 * ((a + b) * sigma_1(z + c) + (a - b))


@njit
def sigma_1(z):
    """Action function helper (Eq. 15, text below)
    Args:
        z: relative distance
    """
    return z / np.sqrt(1 + z ** 2)


def psi_alpha(z, d, r, a, b, h):
    """Pairwise attractive/repulsive potential (Eq. 16)"""
    return integrate.quad(lambda z: phi_alpha(z, d, r, a, b, h), d, z)[0]
