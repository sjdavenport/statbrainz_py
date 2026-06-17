"""mask_bndry (mirrors StatBrainz/Statistics_Functions/Mask_functions/mask_bndry.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.dilate_mask import dilate_mask

__all__ = ['mask_bndry']


def mask_bndry(mask, nonboundary_mask=None):
    """Outer and inner one-voxel boundaries of a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    nonboundary_mask : numpy.ndarray, optional
        If given, boundary voxels inside this mask are removed.

    Returns
    -------
    outer_bndry, inner_bndry : numpy.ndarray
        Boundary masks (as float arrays, matching MATLAB subtraction output).
    """
    mask_f = (np.asarray(mask) > 0).astype(float)
    dilated = dilate_mask(mask, 1).astype(float)
    shrunk = dilate_mask(mask, -1).astype(float)
    inner_bndry = mask_f - shrunk
    outer_bndry = dilated - mask_f
    if nonboundary_mask is not None:
        nb = np.asarray(nonboundary_mask, dtype=float)
        inner_bndry = inner_bndry * (1 - nb)
        outer_bndry = outer_bndry * (1 - nb)
    return outer_bndry, inner_bndry
