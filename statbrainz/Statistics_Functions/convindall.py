"""convindall (mirrors StatBrainz/Statistics_Functions/convindall.m)."""

import numpy as np

from statbrainz.Statistics_Functions.convind import convind

_STDSIZE = (91, 109, 91)

__all__ = ['convindall']


def convindall(indices2convert, dim=_STDSIZE):
    """Apply :func:`convind` to each entry of ``indices2convert``.

    Parameters
    ----------
    indices2convert : sequence of int
        1-based linear indices to convert to coordinates.
    dim : sequence of int, optional
        Array shape. Default 2mm MNI size.

    Returns
    -------
    numpy.ndarray
        ``len(indices2convert) x len(dim)`` array of 1-based coordinates.
    """
    indices2convert = np.atleast_1d(np.asarray(indices2convert))
    return np.array([convind(i, dim) for i in indices2convert])
