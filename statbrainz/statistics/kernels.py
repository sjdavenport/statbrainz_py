"""Gaussian kernels ported from StatBrainz ImageOperations/Kernels.

Faithful ports of: Gker, GkerMV, GkerMV2, Gkerderiv, Gkerderiv2.
"""

import numpy as np

from .transforms import fwhm2sigma

__all__ = ["Gker", "GkerMV", "GkerMV2", "Gkerderiv", "Gkerderiv2"]


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


def GkerMV(x, sigma2_or_fwhm, use_fwhm=True):
    """Isotropic multivariate Gaussian kernel.

    Parameters
    ----------
    x : array_like
        ``D x n`` array of points (columns are points).
    sigma2_or_fwhm : float
        FWHM (if ``use_fwhm``) or variance otherwise.
    use_fwhm : bool, optional
        Default ``True``.

    Returns
    -------
    numpy.ndarray
        Length-``n`` kernel values.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if use_fwhm:
        sigma2 = fwhm2sigma(sigma2_or_fwhm) ** 2
    else:
        sigma2 = sigma2_or_fwhm
    D = x.shape[0]
    return np.exp(-np.sum(x**2, axis=0) / (2 * sigma2)) / (
        np.sqrt(2 * np.pi * sigma2) ** D
    )


def GkerMV2(x, sigma2_or_fwhm, use_fwhm=True):
    """Anisotropic (diagonal-covariance, 2D) multivariate Gaussian kernel.

    Mirrors the MATLAB implementation which is specialised to 2 dimensions.

    Parameters
    ----------
    x : array_like
        ``2 x n`` array of points.
    sigma2_or_fwhm : array_like
        Per-dimension FWHM (if ``use_fwhm``) or variances otherwise.
    use_fwhm : bool, optional
        Default ``True``.

    Returns
    -------
    numpy.ndarray
        Length-``n`` kernel values.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if use_fwhm:
        sigma2 = fwhm2sigma(np.asarray(sigma2_or_fwhm, dtype=float)) ** 2
    else:
        sigma2 = np.asarray(sigma2_or_fwhm, dtype=float)
    D = x.shape[0]
    det_sigma = np.prod(sigma2)
    norm = det_sigma ** (-0.5) * (2 * np.pi) ** (-D / 2)
    return norm * np.exp(
        -x[0] ** 2 / (2 * sigma2[0]) - x[1] ** 2 / (2 * sigma2[1])
    )


def Gkerderiv(x, fwhm):
    """First derivative of the 1D Gaussian kernel (FWHM parameterisation)."""
    x = np.asarray(x, dtype=float)
    sigma2 = fwhm2sigma(fwhm) ** 2
    return (-x / sigma2) * np.exp(-(x**2) / (2 * sigma2)) / np.sqrt(
        2 * np.pi * sigma2
    )


def Gkerderiv2(x, fwhm):
    """Second derivative of the 1D Gaussian kernel (FWHM parameterisation)."""
    x = np.asarray(x, dtype=float)
    sigma2 = fwhm2sigma(fwhm) ** 2
    return (-1 / sigma2 + x**2 / sigma2**2) * np.exp(
        -(x**2) / (2 * sigma2)
    ) / np.sqrt(2 * np.pi * sigma2)
