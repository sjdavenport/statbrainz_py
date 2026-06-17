"""cluster_im (mirrors StatBrainz/Inference/ClusterInference/cluster_im.m)."""

import numpy as np

from statbrainz.Statistics_Functions.convindall import convindall

__all__ = ['cluster_im']


def cluster_im(dim, index_locations, threshold):
    """Build an image of clusters whose size exceeds ``threshold``.

    Parameters
    ----------
    dim : sequence of int
        Output image shape.
    index_locations : sequence of array_like
        Per-cluster 1-based column-major flat indices (as from
        :func:`numOfConComps`).
    threshold : float
        Minimum cluster size (strictly greater) to keep.

    Returns
    -------
    surviving_cluster_im : numpy.ndarray
        Binary image with surviving clusters set to 1.
    surviving_clusters : numpy.ndarray
        Coordinates (1-based) of the surviving cluster voxels (via convindall),
        stacked; empty array if none survive.
    surviving_clusters_vec : list of numpy.ndarray
        The surviving clusters' 1-based flat index lists.
    """
    dim = tuple(int(d) for d in dim)
    flat = np.zeros(int(np.prod(dim)))
    surviving_clusters_vec = []
    for idx in index_locations:
        idx = np.atleast_1d(np.asarray(idx, dtype=int))
        if idx.size > threshold:
            flat[idx - 1] = 1  # 1-based -> 0-based
            surviving_clusters_vec.append(idx)
    surviving_cluster_im = flat.reshape(dim, order="F")

    if surviving_clusters_vec:
        all_idx = np.concatenate(surviving_clusters_vec)
        surviving_clusters = convindall(all_idx, dim)
    else:
        surviving_clusters = np.array([])
    return surviving_cluster_im, surviving_clusters, surviving_clusters_vec
