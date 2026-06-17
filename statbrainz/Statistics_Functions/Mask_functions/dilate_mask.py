"""dilate_mask (mirrors StatBrainz/Statistics_Functions/Mask_functions/dilate_mask.m)."""

import numpy as np
from scipy import ndimage

__all__ = ['dilate_mask']


def dilate_mask(mask, dilation=1):
    """Dilate (``dilation>0``) or erode (``dilation<0``) a binary mask.

    Uses a full box structuring element of side ``2*abs(dilation)+1``.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    dilation : int, optional
        Amount to dilate (positive) or erode (negative). Default ``1``.

    Returns
    -------
    numpy.ndarray
        Boolean dilated/eroded mask.
    """
    mask = np.asarray(mask) > 0
    D = mask.ndim
    if dilation == 0:
        return mask.copy()
    k = abs(int(dilation))
    structure = np.ones((2 * k + 1,) * D, dtype=bool)
    if dilation > 0:
        out = ndimage.binary_dilation(mask, structure=structure)
    else:
        out = ~ndimage.binary_dilation(~mask, structure=structure)
    return out.astype(bool)
