"""Gker (mirrors StatBrainz/Statistics_Functions/ImageOperations/Kernels/Gker.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.FWHM2sigma import fwhm2sigma

__all__ = ['Gker']


def Gker(x, sigma2_or_fwhm, use_fwhm=True):
    """1D Gaussian kernel value and its first two derivatives.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the kernel.
    sigma2_or_fwhm : float
        FWHM (if ``use_fwhm``) or variance ``sigma**2`` otherwise.
    use_fwhm : bool, optional
        Interpret ``sigma2_or_fwhm`` as a FWHM. Default ``True``.

    Returns
    -------
    val, deriv, deriv2 : numpy.ndarray
        Kernel value and first/second derivatives at ``x``.
    """
    x = np.asarray(x, dtype=float)
    if use_fwhm:
        sigma2 = fwhm2sigma(sigma2_or_fwhm) ** 2
    else:
        sigma2 = sigma2_or_fwhm
    base = np.exp(-(x**2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    val = base
    deriv = (-x / sigma2) * base
    deriv2 = (-1 / sigma2 + x**2 / sigma2**2) * base
    return val, deriv, deriv2
