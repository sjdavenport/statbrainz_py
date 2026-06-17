"""numOfConComps (mirrors StatBrainz/Inference/ClusterInference/numOfConComps.m)."""

import numpy as np
from scipy import ndimage

from ._shared import _structure, _pixel_idx_lists

__all__ = ['numOfConComps']


def numOfConComps(data, thresh, connectivity_criterion=None):
    """Connected components of ``data > thresh``.

    Parameters
    ----------
    data : numpy.ndarray
        2D or 3D image.
    thresh : float
        Threshold; voxels with ``data > thresh`` are foreground.
    connectivity_criterion : int, optional
        2D: 4 or 8 (default 4). 3D: 6, 18, or 26 (default 18).

    Returns
    -------
    number_of_clusters : int
        Number of connected components.
    occurences : numpy.ndarray
        Counts of each distinct cluster size.
    cluster_sizes : numpy.ndarray
        The distinct cluster sizes (sorted ascending).
    index_locations : list of numpy.ndarray
        1-based column-major flat indices per cluster.
    """
    data = np.asarray(data)
    s = data.shape
    D = 1 if s[0] == 1 else data.ndim
    if connectivity_criterion is None:
        connectivity_criterion = 4 if D == 2 else 18

    ones_and_zeros = (data > thresh).astype(int)
    structure = _structure(D, connectivity_criterion)
    labeled, number_of_clusters = ndimage.label(ones_and_zeros, structure=structure)

    index_locations = _pixel_idx_lists(labeled, number_of_clusters)
    size_array = np.array([len(idx) for idx in index_locations])
    if size_array.size:
        cluster_sizes = np.unique(size_array)
        occurences = np.array([np.sum(size_array == s_) for s_ in cluster_sizes])
    else:
        cluster_sizes = np.array([], dtype=int)
        occurences = np.array([], dtype=int)
    return number_of_clusters, occurences, cluster_sizes, index_locations
