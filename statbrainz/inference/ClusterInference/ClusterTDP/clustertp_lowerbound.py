"""clustertp_lowerbound (mirrors StatBrainz/Inference/ClusterInference/ClusterTDP/clustertp_lowerbound.m)."""

import numpy as np

from statbrainz.Inference.ClusterInference.AuxFiles.rkval import rkval

__all__ = ['clustertp_lowerbound']


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
