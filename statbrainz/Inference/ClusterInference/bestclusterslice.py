"""bestclusterslice (mirrors StatBrainz/Inference/ClusterInference/bestclusterslice.m)."""

import numpy as np

__all__ = ['bestclusterslice']


def bestclusterslice(slice_no, surviving_cluster_im):
    """Find the slice(s) along axis ``slice_no`` containing the most cluster mass.

    Parameters
    ----------
    slice_no : int
        Axis to scan (1-based, matching MATLAB: 1, 2, or 3). ``0`` returns the
        best slice along all three axes.
    surviving_cluster_im : numpy.ndarray
        3D binary cluster image.

    Returns
    -------
    maxsumlocs : numpy.ndarray
        1-based slice index/indices with the maximum in-slice sum.
    totalinslice : numpy.ndarray
        Per-slice sums along the chosen axis.
    """
    im = np.asarray(surviving_cluster_im)
    if slice_no == 0:
        bestx, _ = bestclusterslice(1, im)
        besty, _ = bestclusterslice(2, im)
        bestz, _ = bestclusterslice(3, im)
        return np.array([bestx[0], besty[0], bestz[0]]), None

    axis = slice_no - 1  # 1-based -> 0-based
    other_axes = tuple(a for a in range(im.ndim) if a != axis)
    totalinslice = im.sum(axis=other_axes)
    maxsumlocs = np.nonzero(totalinslice == totalinslice.max())[0] + 1  # 1-based
    return maxsumlocs, totalinslice
