"""getMNImask (mirrors StatBrainz/Brain_Functions/getMNImask.m)."""

import numpy as np

from statbrainz.ImageViewing.imgload import imgload
from statbrainz.Statistics_Functions.Mask_functions.mask_bounds import mask_bounds

__all__ = ['getMNImask']


def getMNImask():
    """Return the cropped 2mm MNI brain mask and its bounds.

    Returns
    -------
    MNImask : numpy.ndarray
        Boolean mask cropped to its tight bounding box (shape ``72 x 90 x 77``).
    bounds : list of slice
        The 0-based crop bounds used (per dimension).
    """
    MNImask = imgload("MNImask")
    bounds, _ = mask_bounds(MNImask)
    MNImask = MNImask[tuple(bounds)].astype(bool)
    return MNImask, bounds
