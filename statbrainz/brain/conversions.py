"""Misc brain conversion helpers ported from StatBrainz Brain Functions.

Faithful ports of: sigma2FWHM (alias of statistics.sigma2fwhm), plane2index,
nifti_type.
"""

import numpy as np

from ..statistics.transforms import sigma2fwhm

__all__ = ["sigma2fwhm", "plane2index", "nifti_type"]


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


def nifti_type(files):
    """Classify file names as ``.nii.gz`` (1), ``.nii`` (2), or other (0).

    Parameters
    ----------
    files : str or sequence of str
        File name(s) to classify.

    Returns
    -------
    numpy.ndarray
        Integer code per file: 0 = neither, 1 = ``.nii.gz``, 2 = ``.nii``.
    """
    if isinstance(files, str):
        files = [files]
    out = np.zeros(len(files), dtype=int)
    for i, f in enumerate(files):
        if ".nii.gz" in f:
            out[i] = 1
        elif ".nii" in f:
            out[i] = 2
        else:
            out[i] = 0
    return out
