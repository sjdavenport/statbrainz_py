"""Cluster TDP (true discovery proportion) bounds ported from StatBrainz.

Faithful ports of: rkval (+ fdk), clustertp_lowerbound, cluster_tp2tdp, clustertdp.

The ``lowerbound`` method implements the All-Resolutions-Inference style
combinatorial lower bound on the number of true positives in a cluster. The
``heuristic``/``greedy`` method in MATLAB dispatches external jobs (fgreedy) and
is NOT ported here.
"""

import numpy as np

__all__ = ["rkval", "clustertp_lowerbound", "cluster_tp2tdp", "clustertdp"]


def _fdk(k, d):
    if d == 0 or k == 0:
        return 0
    floork1overd = int(np.floor(k ** (1 / d)))
    ldk = int(
        np.floor(
            (np.log(k) - d * np.log(floork1overd))
            / (np.log(floork1overd + 1) - np.log(floork1overd))
        )
    )
    bdkplus = (floork1overd + 1) ** (d - ldk) * (floork1overd + 2) ** ldk
    bdk = floork1overd ** (d - ldk) * (floork1overd + 1) ** ldk
    return bdkplus + _fdk(k - bdk, d - 1)


def rkval(k, d=3):
    """Compute the ``r_k`` constant used in the cluster TP lower bound.

    Parameters
    ----------
    k : int
        Cluster (size) threshold.
    d : int, optional
        Dimension. Default ``3``.

    Returns
    -------
    float
        ``min_i (fdk(i, d) - i) / fdk(i, d)`` over ``i = 1..k``.
    """
    min_value = np.inf
    for i in range(1, k + 1):
        fdki = _fdk(i, d)
        if fdki != 0:
            min_value = min(min_value, (fdki - i) / fdki)
    return min_value


def clustertp_lowerbound(CL, cluster_threshold, D=3):
    """Combinatorial lower bound on the number of true positives in a cluster.

    Parameters
    ----------
    CL : dict or numpy.ndarray
        Cluster voxel coordinates. Either a dict with keys ``'x'``, ``'y'``,
        ``'z'`` (1-based, as in MATLAB), or an ``n x 3`` array of coordinates.
    cluster_threshold : int
        The cluster-forming size threshold.
    D : int, optional
        Dimension (only 3 is supported, matching MATLAB). Default ``3``.

    Returns
    -------
    int
        The lower bound on the number of true positives.
    """
    if isinstance(CL, dict):
        x = np.asarray(CL["x"], dtype=float).ravel()
        y = np.asarray(CL["y"], dtype=float).ravel()
        z = np.asarray(CL["z"], dtype=float).ravel()
    else:
        arr = np.asarray(CL, dtype=float)
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    rk = rkval(cluster_threshold, D)
    range_x = (x.max() - x.min()) + 3
    range_y = (y.max() - y.min()) + 3

    CL_lin = x + range_x * y + range_x * range_y * z

    nx = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    ny = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    nz = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    neighbors = nx + range_x * ny + range_x * range_y * nz

    def CLplus(cl):
        if cl.size == 0:
            return cl
        return np.unique((cl[:, None] + neighbors[None, :]).ravel())

    def CLmin(cl):
        clplus = CLplus(cl)
        clrest = np.setdiff1d(clplus, cl)
        return np.setdiff1d(cl, np.unique((clrest[:, None] - neighbors[None, :]).ravel()))

    def CLprune(cl, i):
        for _ in range(i):
            cl = CLmin(cl)
        for _ in range(i):
            cl = CLplus(cl)
        return cl

    CL_lin = np.unique(CL_lin)
    max_i = int(np.floor(len(CL_lin) ** (1 / 3)))
    sV = float(len(CL_lin) > cluster_threshold)

    for i in range(0, max_i + 1):
        pruned = CLprune(CL_lin, i)
        pruned_plus = CLplus(pruned)
        pruned_rest = np.setdiff1d(pruned_plus, pruned)
        sV = max(sV, rk * len(pruned_plus) - len(pruned_rest))

    return int(np.ceil(sV))


def cluster_tp2tdp(tp_bounds, clusters):
    """Convert true-positive bounds to true-discovery-proportion bounds.

    Parameters
    ----------
    tp_bounds : array_like
        Per-cluster TP bounds.
    clusters : sequence
        Per-cluster voxel collections (length gives the cluster size).

    Returns
    -------
    numpy.ndarray
        ``tp_bounds / cluster_sizes``.
    """
    tp_bounds = np.asarray(tp_bounds, dtype=float)
    cluster_sizes = np.array([len(c) for c in clusters], dtype=float)
    return tp_bounds / cluster_sizes


def clustertdp(clusters, cluster_threshold, method="lowerbound"):
    """TDP and TP bounds for each cluster (``lowerbound`` method only).

    Parameters
    ----------
    clusters : sequence of numpy.ndarray
        Each cluster is an ``n x 3`` array of (1-based) voxel coordinates.
    cluster_threshold : int
        Cluster-forming size threshold.
    method : str, optional
        Only ``'lowerbound'`` is supported here. The MATLAB
        ``'heuristic'``/``'greedy'`` paths dispatch external jobs and are not
        ported.

    Returns
    -------
    tdp_bounds : numpy.ndarray
        Per-cluster TDP lower bounds.
    tp_bounds : numpy.ndarray
        Per-cluster TP lower bounds.
    """
    if method != "lowerbound":
        raise NotImplementedError(
            "Only method='lowerbound' is ported; the heuristic/greedy paths "
            "dispatch external jobs in MATLAB."
        )
    n = len(clusters)
    tp_bounds = np.zeros(n)
    tdp_bounds = np.zeros(n)
    for I in range(n):
        coords = np.asarray(clusters[I], dtype=float)
        CL = {"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]}
        tp_bounds[I] = clustertp_lowerbound(CL, cluster_threshold, 3)
        tdp_bounds[I] = tp_bounds[I] / coords.shape[0]
    return tdp_bounds, tp_bounds
