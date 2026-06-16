"""MNI mask helpers ported from StatBrainz Brain Functions.

Faithful ports of: getMNImask, voxinMNI, gen_mask, mvtstat_dep.
These depend on the bundled ``MNImask`` volume loaded via :func:`imgload`.
"""

import numpy as np

from .io import imgload
from ..statistics.mask_bounds import mask_bounds
from ..statistics.indexing import convind
from ..statistics.mask_functions import nan2zero

__all__ = ["getMNImask", "voxinMNI", "gen_mask", "mvtstat_dep"]

_STDSIZE = (91, 109, 91)


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
    return bool(MNI.ravel(order="F")[loc - 1])  # 1-based -> 0-based


def gen_mask(data, use_MNI=True, make3D=True, stdsize=_STDSIZE):
    """Intersection mask of per-subject masks (optionally with the MNI mask).

    Parameters
    ----------
    data : numpy.ndarray
        ``nsubj x nvox`` matrix of per-subject masks (rows).
    use_MNI : bool, optional
        Start from the MNI mask. Default ``True``.
    make3D : bool, optional
        Reshape the result to ``(91, 109, 91)``. Default ``True``.
    stdsize : sequence of int, optional
        Standard volume size used when ``use_MNI`` is ``False`` and for the 3D
        reshape. Default ``(91, 109, 91)``.

    Returns
    -------
    numpy.ndarray
        The intersection mask (3D if ``make3D``, else a vector).
    """
    data = np.asarray(data, dtype=float)
    if use_MNI:
        mask = imgload("MNImask").astype(float).ravel(order="F")
    else:
        mask = np.ones(int(np.prod(stdsize)))

    nsubj = data.shape[0]
    for subj in range(nsubj):
        mask = mask * data[subj, :]
    if make3D:
        mask = mask.reshape((91, 109, 91), order="F")
    return mask


def mvtstat_dep(data, Dim=None, nansaszeros=False):
    """Deprecated voxelwise t-statistic with subjects on the FIRST axis.

    Unlike :func:`statbrainz.statistics.mvtstat`, here ``data`` is
    ``nsubj x nvox`` (subjects index the first axis).

    Parameters
    ----------
    data : numpy.ndarray
        ``nsubj x nvox`` matrix.
    Dim : int or sequence of int, optional
        If ``1``, reshape to ``(91, 109, 91)``. If it matches ``nvox``, reshape
        to ``Dim``. Otherwise return vectors.
    nansaszeros : bool, optional
        Replace NaNs in ``tstat``/``cohensd`` with zeros.

    Returns
    -------
    tstat, xbar, std_dev, cohensd : numpy.ndarray
    """
    data = np.asarray(data, dtype=float)
    nsubj = data.shape[0]
    xbar = np.mean(data, axis=0)
    sq_xbar = np.mean(data**2, axis=0)
    est_var = (nsubj / (nsubj - 1)) * (sq_xbar - xbar**2)
    std_dev = np.real(np.sqrt(est_var.astype(complex)))

    nvox = data.shape[1]
    if Dim is not None and np.ndim(Dim) == 0 and Dim == 1:
        xbar = xbar.reshape(_STDSIZE, order="F")
        std_dev = std_dev.reshape(_STDSIZE, order="F")
    elif Dim is not None and np.prod(np.atleast_1d(Dim)) == nvox:
        Dim = tuple(int(d) for d in np.atleast_1d(Dim))
        xbar = xbar.reshape(Dim, order="F")
        std_dev = std_dev.reshape(Dim, order="F")

    tstat = np.sqrt(nsubj) * xbar / std_dev
    cohensd = xbar / std_dev
    if nansaszeros:
        tstat = nan2zero(tstat)
        cohensd = nan2zero(cohensd)
    return tstat, xbar, std_dev, cohensd
