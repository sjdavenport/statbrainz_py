"""voxinMNI (mirrors StatBrainz/Brain_Functions/voxinMNI.m)."""

import numpy as np

from statbrainz.ImageViewing.imgload import imgload
from statbrainz.Statistics_Functions.convind import convind

__all__ = ['voxinMNI']


def voxinMNI(loc):
    """Test whether a voxel is inside the MNI brain mask.

    Parameters
    ----------
    loc : int or sequence of int
        A 1-based column-major linear index, or a 1-based coordinate vector
        (length 3) which is converted via :func:`convind`.

    Returns
    -------
    bool
        ``True`` if the voxel is in the mask.
    """
    loc = np.atleast_1d(np.asarray(loc, dtype=int))
    if loc.size > 1:
        loc = int(convind(loc))
    else:
        loc = int(loc[0])
    MNI = imgload("MNImask")
    return bool(MNI.ravel(order="F")[loc - 1])
