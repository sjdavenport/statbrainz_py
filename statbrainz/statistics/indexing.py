"""Index conversion utilities ported from StatBrainz.

Faithful ports of: convind, convindall.

IMPORTANT — indexing convention
--------------------------------
These mirror MATLAB exactly, so that linear indices match cross-referenced
MATLAB code:

* Linear indices are **1-based** and use **column-major (Fortran)** ordering
  (the same as MATLAB ``ind2sub``/``sub2ind`` and ``A(:)``).
* Coordinate vectors are **1-based**.

If you want 0-based numpy linear indices, use
``numpy.unravel_index(idx, shape, order='F')`` directly instead.
"""

import numpy as np

__all__ = ["convind", "convindall"]

_STDSIZE = (91, 109, 91)  # 2mm MNI brain image size in voxels


def convind(ind, size_of_array=_STDSIZE, conv2what=0):
    """Switch between 1-based linear indexing and 1-based coordinate indexing.

    Parameters
    ----------
    ind : int or sequence of int
        If a scalar (length 1) it is a 1-based linear index in column-major
        order and the coordinate vector is returned. If a sequence with length
        equal to ``len(size_of_array)`` it is a 1-based coordinate vector and
        the 1-based linear index is returned.
    size_of_array : sequence of int, optional
        Shape ``[x_1, ..., x_n]`` of the array. Default is the 2mm MNI size
        ``(91, 109, 91)``.
    conv2what : {0, 1, 2}, optional
        Only applies to the linear->coordinate direction.
        ``0`` (default) returns the 1-based coordinates unchanged;
        ``1`` returns fsleyes (0-based) coordinates;
        ``2`` returns aligned anatomical (MNI mm) coordinates.

    Returns
    -------
    numpy.ndarray or int
        A coordinate vector if ``ind`` is scalar, otherwise the linear index.
    """
    ind_arr = np.atleast_1d(np.asarray(ind))
    len_ind = ind_arr.size
    size_of_array = tuple(int(s) for s in np.atleast_1d(size_of_array))
    D = len(size_of_array)

    if len_ind > 1 and len_ind != D:
        raise ValueError(
            "The length of ind must be 1 (linear) or match the array size "
            "(coordinate representation)."
        )

    # 1D / row-vector degenerate case: nothing to change.
    if D == 2 and size_of_array[0] == 1:
        return ind

    if len_ind == 1:
        # Linear (1-based) -> coordinate (1-based), column-major order.
        linear0 = int(ind_arr[0]) - 1
        coords0 = np.unravel_index(linear0, size_of_array, order="F")
        out = np.array(coords0, dtype=float) + 1  # back to 1-based
        if conv2what == 1:
            out = out - 1
        elif conv2what == 2:
            out = out - 1
            out[0] = 90 - 2 * out[0]
            out[1] = 2 * out[1] - 126
            out[2] = 2 * out[2] - 72
        return np.squeeze(out)

    # Coordinate (1-based) -> linear (1-based), column-major order.
    coords0 = (ind_arr.astype(int) - 1)
    linear0 = np.ravel_multi_index(tuple(coords0), size_of_array, order="F")
    return int(linear0) + 1


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
