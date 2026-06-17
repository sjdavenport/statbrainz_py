"""mvtstat (mirrors StatBrainz/Statistics_Functions/Stats_functions/mvtstat.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero

__all__ = ['mvtstat']


def mvtstat(data, Dim=None, nansaszeros=False):
    """Voxelwise one-sample t-statistic over the last axis (subjects).

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array; the last axis indexes subjects.
    Dim : sequence of int, optional
        Output spatial shape. Defaults to ``data.shape[:-1]``.
    nansaszeros : bool, optional
        Replace NaNs in ``tstat``/``cohensd`` with zeros. Default ``False``.

    Returns
    -------
    tstat : numpy.ndarray
        t-statistic image.
    xbar : numpy.ndarray
        Mean image.
    std_dev : numpy.ndarray
        Standard deviation image (population estimate, ddof=1 scaling).
    cohensd : numpy.ndarray
        Cohen's d image (``xbar / std_dev``).
    """
    data = np.asarray(data, dtype=float)
    sD = data.shape
    if Dim is None:
        Dim = sD[:-1]
    Dim = tuple(int(d) for d in np.atleast_1d(Dim))
    nsubj = sD[-1]

    xbar = np.mean(data, axis=-1)
    sq_xbar = np.mean(data**2, axis=-1)
    est_var = (nsubj / (nsubj - 1)) * (sq_xbar - xbar**2)
    std_dev = np.real(np.sqrt(est_var.astype(complex)))

    if np.prod(Dim) == np.prod(sD[:-1]) and len(Dim) > 1:
        xbar = xbar.reshape(Dim, order="F")
        std_dev = std_dev.reshape(Dim, order="F")
    else:
        xbar = xbar.ravel(order="F")
        std_dev = std_dev.ravel(order="F")

    tstat = np.sqrt(nsubj) * xbar / std_dev
    cohensd = xbar / std_dev
    if nansaszeros:
        tstat = nan2zero(tstat)
        cohensd = nan2zero(cohensd)
    return tstat, xbar, std_dev, cohensd
