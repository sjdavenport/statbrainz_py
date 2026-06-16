"""Connected-component / cluster utilities ported from StatBrainz.

Faithful ports of: numOfConComps, getlargestcluster, cluster_im, index2mask,
bestclusterslice.

MATLAB ``bwconncomp`` returns ``PixelIdxList`` as **1-based column-major** linear
indices. To interoperate with the rest of the package (``convindall`` etc.) we
keep that convention: ``index_locations`` here are lists of 1-based column-major
linear indices.

Connectivity mapping (MATLAB ``bwconncomp`` -> scipy structuring element):
* 2D: 4 -> cross (rank-1), 8 -> full 3x3
* 3D: 6 -> faces (rank-1), 18 -> faces+edges (rank-2), 26 -> full 3x3x3
"""

import numpy as np
from scipy import ndimage

from ..statistics.indexing import convindall

__all__ = [
    "numOfConComps",
    "getlargestcluster",
    "cluster_im",
    "index2mask",
    "bestclusterslice",
]


def _structure(D, connectivity_criterion):
    if D == 2:
        if connectivity_criterion == 4:
            return ndimage.generate_binary_structure(2, 1)
        if connectivity_criterion == 8:
            return ndimage.generate_binary_structure(2, 2)
        raise ValueError("In 2D the connectivity criterion must be 4 or 8")
    if D == 3:
        if connectivity_criterion == 6:
            return ndimage.generate_binary_structure(3, 1)
        if connectivity_criterion == 18:
            return ndimage.generate_binary_structure(3, 2)
        if connectivity_criterion == 26:
            return ndimage.generate_binary_structure(3, 3)
        raise ValueError("In 3D the connectivity criterion must be 6, 18 or 26")
    raise ValueError("The dimension must be 2 or 3.")


def _pixel_idx_lists(labeled, num):
    """Per-label 1-based column-major flat indices (like bwconncomp PixelIdxList)."""
    flat = labeled.ravel(order="F")
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    out = []
    for lab in range(1, num + 1):
        lo = np.searchsorted(sorted_labels, lab, side="left")
        hi = np.searchsorted(sorted_labels, lab, side="right")
        out.append(order[lo:hi] + 1)  # 1-based
    return out


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


def index2mask(indices, dim=(91, 109, 91)):
    """Set 1-based column-major flat ``indices`` to 1 in a zero image of ``dim``."""
    dim = tuple(int(d) for d in dim)
    flat = np.zeros(int(np.prod(dim)))
    indices = np.atleast_1d(np.asarray(indices, dtype=int))
    flat[indices - 1] = 1
    return flat.reshape(dim, order="F")


def getlargestcluster(mask):
    """Return a binary mask of the largest connected component of ``mask``."""
    mask = np.asarray(mask) > 0
    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    labeled, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return np.zeros(mask.shape, dtype=float)
    idx_lists = _pixel_idx_lists(labeled, num)
    lengths = [len(i) for i in idx_lists]
    largest = int(np.argmax(lengths))
    cluster_im_out, _, _ = cluster_im(mask.shape, [idx_lists[largest]], 0.5)
    return cluster_im_out


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
