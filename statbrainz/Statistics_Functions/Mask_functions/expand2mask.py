"""expand2mask (mirrors StatBrainz/Statistics_Functions/Mask_functions/expand2mask.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.mask_bounds import mask_bounds

__all__ = ['expand2mask']


def expand2mask(im, mask, padding=0):
    """Place a cropped sub-image ``im`` back into a full 2D MNI-sized image.

    Mirrors the MATLAB ``expand2mask`` which is hard-coded to a ``91 x 109``
    output for 2D input.

    Parameters
    ----------
    im : numpy.ndarray
        Cropped image to place back.
    mask : numpy.ndarray
        Mask defining the crop bounds.
    padding : int, optional
        Padding used when the bounds were computed. Default ``0``.

    Returns
    -------
    numpy.ndarray
        ``91 x 109`` array with ``im`` placed at the mask bounds.
    """
    bounds, _ = mask_bounds(mask, padding)
    D = np.asarray(im).ndim
    if len(bounds) == 3 and D == 2:
        bounds = bounds[:2]
    out = np.zeros((91, 109))
    out[tuple(bounds)] = im
    return out
