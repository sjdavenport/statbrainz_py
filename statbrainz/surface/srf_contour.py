"""srf_contour (mirrors StatBrainz/Surface/srf_contour.m)."""

import numpy as np

from .srf_dilate_mask import srf_dilate_mask

__all__ = ['srf_contour']


def srf_contour(srf, mask, adjacent_set_mask=None):
    """Inner and outer one-ring contours of a mask on a surface.

    Parameters
    ----------
    srf : dict
        Surface (or bilateral dict).
    mask : numpy.ndarray or dict
        Per-vertex binary mask.
    adjacent_set_mask : numpy.ndarray, optional
        If given, restrict the contours to the adjacent set.

    Returns
    -------
    inner_contour, outer_contour : numpy.ndarray or dict
    """
    if "lh" in srf and "rh" in srf:
        inner = {}
        outer = {}
        for hemi in ("lh", "rh"):
            inner[hemi], outer[hemi] = srf_contour(srf[hemi], mask[hemi])
        return inner, outer

    mask_f = np.asarray(mask, dtype=float)
    dilated_mask = srf_dilate_mask(srf, mask, 1).astype(float)
    shrunk_mask = np.asarray(srf_dilate_mask(srf, mask, -1), dtype=float)
    inner_contour = mask_f - shrunk_mask
    outer_contour = dilated_mask - mask_f
    if adjacent_set_mask is not None:
        adj_dilated = srf_dilate_mask(srf, adjacent_set_mask, 1).astype(float)
        inner_contour = inner_contour * adj_dilated
        outer_contour = outer_contour * np.asarray(adjacent_set_mask, dtype=float)
    return inner_contour, outer_contour
