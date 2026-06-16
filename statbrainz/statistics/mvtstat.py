"""Voxelwise t-statistic and noise generation ported from StatBrainz.

Faithful ports of: mvtstat, gen_noise.
"""

import numpy as np

from .mask_functions import nan2zero
from .convolution import fast_conv

__all__ = ["mvtstat", "gen_noise"]


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


def gen_noise(mask, FWHM, nsubj=1, mean_img=None, std_img=None, rng=None):
    """Generate smoothed, variance-normalised noise within a mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Spatial mask; noise is multiplied by it.
    FWHM : float or sequence of float
        Smoothing FWHM.
    nsubj : int, optional
        Number of subjects (trailing axis). Default ``1``.
    mean_img : numpy.ndarray, optional
        Mean image added after scaling. Default zeros.
    std_img : numpy.ndarray, optional
        Per-voxel std multiplier. Default ones.
    rng : numpy.random.Generator, optional
        Random generator. Default ``np.random.default_rng()``.

    Returns
    -------
    numpy.ndarray
        Noise of shape ``(*mask.shape, nsubj)`` (or ``mask.shape`` if nsubj==1
        and the trailing singleton is squeezed by the caller's expectations —
        here it is kept as ``(*mask.shape, nsubj)`` to match fast_conv).
    """
    mask = np.asarray(mask, dtype=float)
    if mean_img is None:
        mean_img = np.zeros(mask.shape)
    if std_img is None:
        std_img = np.ones(mask.shape)
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.standard_normal((*mask.shape, nsubj))
    smooth_noise, ss = fast_conv(noise, FWHM, mask.ndim)
    smooth_noise = smooth_noise / np.sqrt(ss)
    smooth_noise = smooth_noise * mask[..., None] * std_img[..., None] + mean_img[..., None]
    return smooth_noise
