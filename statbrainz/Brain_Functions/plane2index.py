"""plane2index (mirrors StatBrainz/Brain_Functions/plane2index.m)."""

import numpy as np

__all__ = ['plane2index']


def plane2index(plane, extra=False):
    """Convert a 2D ``plane`` specifier into a numpy indexing tuple.

    The MATLAB version returns a cell index like ``{':',':',fixed}``. Here we
    return a Python tuple suitable for ``array[idx]`` with **0-based** numpy
    semantics: the fixed axis is the one with a non-zero entry in ``plane``, and
    the fixed voxel is ``plane[axis] - 1`` (converting MATLAB's 1-based voxel
    number to a 0-based index).

    Parameters
    ----------
    plane : array_like
        A vector/matrix with a single non-zero entry indicating which axis is
        fixed and at which (1-based) voxel.
    extra : bool, optional
        If ``True`` return a 4D index (3 spatial ``slice(None)`` + value),
        else a 3D index. Default ``False``.

    Returns
    -------
    tuple
        Indexing tuple, e.g. ``(slice(None), slice(None), 49)``.
    """
    plane = np.asarray(plane).ravel()
    (nz,) = np.nonzero(plane)
    axis = int(nz[0])
    fixed_voxel = int(plane[axis]) - 1  # MATLAB 1-based -> 0-based
    ndim = 4 if extra else 3
    idx = [slice(None)] * ndim
    idx[axis] = fixed_voxel
    return tuple(idx)
