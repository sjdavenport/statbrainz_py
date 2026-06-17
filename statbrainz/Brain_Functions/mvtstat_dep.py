"""mvtstat_dep (mirrors StatBrainz/Brain_Functions/mvtstat_dep.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero

_STDSIZE = (91, 109, 91)

__all__ = ['mvtstat_dep']


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
