"""clustertdp (mirrors StatBrainz/Inference/ClusterInference/ClusterTDP/clustertdp.m)."""

import numpy as np

from .clustertp_lowerbound import clustertp_lowerbound

__all__ = ['clustertdp']


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
