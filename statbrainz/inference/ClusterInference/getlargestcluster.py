"""getlargestcluster (mirrors StatBrainz/Inference/ClusterInference/getlargestcluster.m)."""

import numpy as np
from scipy import ndimage

from ._shared import _pixel_idx_lists
from .cluster_im import cluster_im

__all__ = ['getlargestcluster']


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
