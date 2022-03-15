"""
Olfati-Saber algorithm (Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory)
Source: <https://ieeexplore.ieee.org/document/1605401>
"""
import numpy as np
from numba import njit
from scipy import integrate


def flock(positions: np.ndarray, distance: float, perception_radius: float,
          a=1.0, b=5.0, eps=0.1, h=0.2):
    """Olfati saber algorithm (only gradient based term) (Eq. 23, first term)
    Args:
        positions: relative positions of other agents
        distance: desired inter-agent distance
        perception_radius: cutoff at perception radius
        eps: parameter of the sigma-norm
        h: parameter of the bump function
    """
    d, r = distance, perception_radius
    assert d > 0
    assert r >= d
    assert eps > 0
    assert h >= 0 and h <= 1
    return np.sum([gradient_based_term(p, d, r, a, b, eps, h) for p in positions], axis=0)


def gradient_based_term(x, d, r, a, b, eps, h):
    """Gradient-based term (Eq. 23, first term) """
    # n = sigma_norm_grad(x)  # This does not seem to work
    n = x / np.linalg.norm(x)  # TODO: check why this works better than the above!
    return phi_alpha(sigma_norm(x, eps), d, r, a, b, eps, h) * n


@njit
def sigma_norm(z, eps=0.1):
    """Sigma-norm (Eq. 8)"""
    return (1 / eps) * (np.sqrt(1 + eps * np.linalg.norm(z) ** 2) - 1)


@njit
def sigma_norm_scalar(z, eps=0.1):
    """Scalar version of the sigma-norm (otherwise Numba complains about norm of float)"""
    return (1 / eps) * (np.sqrt(1 + eps * z ** 2) - 1)


@njit
def sigma_norm_grad(z, eps=0.1):
    """Gradient of sigma-norm (Eq. 9)"""
    return z / np.sqrt(1 + eps * np.linalg.norm(z) ** 2)


@njit
def sigma_norm_grad_scalar(z, eps=0.1):
    """Scalar version of the sigma-norm gradient"""
    return z / np.sqrt(1 + eps * z ** 2)


@njit
def phi_alpha(z, d, r, a, b, eps=0.1, h=0.2):
    """Action function (Eq. 15, first line)"""
    d_alpha = sigma_norm_scalar(d, eps)
    r_alpha = sigma_norm_scalar(r, eps)
    return rho_h(z / r_alpha, h) * phi(z - d_alpha, a, b)


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


def psi_alpha(z, d, r, a, b, e, h):
    """Pairwise attractive/repulsive potential (Eq. 16)"""
    d_alpha = sigma_norm_scalar(d, e)
    return integrate.quad(lambda z: phi_alpha(z, d, r, a, b, e, h), d_alpha, z)[0]
