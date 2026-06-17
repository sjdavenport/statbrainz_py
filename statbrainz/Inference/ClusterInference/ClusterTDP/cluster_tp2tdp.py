"""cluster_tp2tdp (mirrors StatBrainz/Inference/ClusterInference/ClusterTDP/cluster_tp2tdp.m)."""

import numpy as np

__all__ = ['cluster_tp2tdp']


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
