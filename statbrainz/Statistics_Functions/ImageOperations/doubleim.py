"""doubleim (mirrors StatBrainz/Statistics_Functions/ImageOperations/doubleim.m)."""

import numpy as np

__all__ = ['doubleim']


def doubleim(orig_data):
    """Double the resolution of a ``91 x 109 x 91`` image to ``182 x 218 x 182``.

    Each original voxel is replicated into a ``2 x 2 x 2`` block, matching the
    MATLAB ``doubleim``.

    Parameters
    ----------
    orig_data : numpy.ndarray
        ``91 x 109 x 91`` array.

    Returns
    -------
    numpy.ndarray
        ``182 x 218 x 182`` array.
    """
    orig_data = np.asarray(orig_data, dtype=float)
    new_data = np.zeros((182, 218, 182))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_data[i:182:2, j:218:2, k:182:2] = orig_data
    return new_data
