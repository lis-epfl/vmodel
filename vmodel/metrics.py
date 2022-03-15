import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA


def connectivity(adjacency: np.ndarray) -> float:
    """Compute algebraic connectivity from adjacency matrix
    Args:
        adjacency: adjacency matrix (N x N)
    Returns:
        connectivity: second smallest eigenvalue of the Laplacian adjusted for graph size
    Note: The algebraic connectivity is only well-defined for undirected graphs!
    """
    adj = adjacency.copy()
    if (adj != adj.T).any():
        adj[adj != adj.T] = True
    degree = np.diag(adj.sum(axis=1))
    laplacian = degree - adj
    eigvals = np.linalg.eigvalsh(laplacian)
    connectivity = np.sort(eigvals)[1] / len(eigvals)
    return np.maximum(connectivity, 0.0)


def aspect_ratio(pos: np.ndarray, use_pca=False) -> float:
    """Computes the aspect ratio of the smallest bounding rectangle
    Args:
        pas: positions (N x D)
    Return
        ratio: aspect ratio of the positions
    """
    pos = PCA().fit_transform(pos) if use_pca else pos
    xs, ys = pos.T
    width, height = (xs.max() - xs.min()), (ys.max() - ys.min())
    return width / height


def area_convex_hull(pos: np.ndarray) -> float:
    """Computes the area of the agent positions based on their convex hull
    Args:
        pos: absolute positions (N x D)
    Return:
        area: area of the convex hull
    Note: Meaning of area/volume of the ConvexHull object are dimension-dependent
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    """
    # Note: When input points are 2-dim, hull.volume is the area of the convex hull!
    ndim = pos.shape[1]
    hull = ConvexHull(pos)
    return hull.volume if ndim == 2 else hull.area


def agent_density(pos: np.ndarray) -> float:
    """Computes the agent density
    Args:
        pos: absolute positions (N x D)
    Return:
        density: agent density
    """
    N = pos.shape[0]
    return N / area_convex_hull(pos)


def order(vel: np.ndarray) -> float:
    """Compute order parameter from velocity matrix
    Args:
        vel: velocity matrix (N x D)
    Returns:
        order: velocity correlation
    """
    N, _ = vel.shape
    speed = np.linalg.norm(vel, axis=1, keepdims=True)  # N x 1
    speed_prod = speed.dot(speed.T)  # N x N
    mask = (speed_prod != 0)  # avoid division by zero!
    dot_prod = vel.dot(vel.T)  # N x N
    np.fill_diagonal(dot_prod, 0)  # i != j
    return (dot_prod[mask] / speed_prod[mask]).sum() / (N * (N - 1))


def distance_matrix(pos: np.ndarray) -> np.ndarray:
    """Compute distance matrix
    Args:
        pos: positions (N x D)
    Return:
        dmat: distance matrix (N x N)
    """
    return np.sqrt(np.sum((pos[:, np.newaxis, :] - pos[np.newaxis, :, :]) ** 2, axis=-1))


def union(adj: np.ndarray) -> float:
    """Compute the union metric
    Args:
        adj: adjacency matrix (N x N)
    Returns:
        union: union metric
    """
    N = len(adj)
    ncomps = connected_components(adj, return_labels=False)
    return 1 - ((ncomps - 1) / (N - 1))
