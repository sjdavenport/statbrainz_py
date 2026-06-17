"""srf_dilate_mask (mirrors StatBrainz/Surface/srf_dilate_mask.m)."""

import numpy as np

from .smooth_surface import smooth_surface

__all__ = ['srf_dilate_mask']


def srf_dilate_mask(srf, mask, dilation):
    """Dilate (``dilation>0``) or erode (``dilation<0``) a mask on a surface.

    Dilation by ``k`` is ``k`` rounds of edge-averaging thresholded at ``>0``;
    erosion dilates the complement.

    Parameters
    ----------
    srf : dict
        Surface.
    mask : numpy.ndarray
        Per-vertex binary mask.
    dilation : int
        Amount to dilate (positive) or erode (negative).

    Returns
    -------
    numpy.ndarray
        Dilated/eroded mask (boolean for dilation; matches MATLAB for erosion).
    """
    if dilation == 0:
        return mask
    if dilation > 0:
        m = np.asarray(mask, dtype=float)
        return smooth_surface(srf, m, 0, "ones", dilation) > 0
    outside_mask = 1 - np.asarray(mask, dtype=float)
    outside_dilated = srf_dilate_mask(srf, outside_mask, -dilation)
    return 1 - outside_dilated
