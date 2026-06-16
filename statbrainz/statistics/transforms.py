"""Numeric transforms ported from StatBrainz Statistics_Functions.

Faithful ports of: apower, asinh_trans, asinh_data_trans, FWHM2sigma, sigma2FWHM.
"""

import numpy as np

__all__ = ["apower", "asinh_trans", "asinh_data_trans", "fwhm2sigma", "sigma2fwhm"]


def apower(x, power=0.5):
    """Raise ``x`` to ``power`` preserving sign: ``sign(x) * |x| ** power``.

    Parameters
    ----------
    x : array_like
        A numeric array.
    power : float, optional
        The exponent to apply. Default ``0.5``.

    Returns
    -------
    numpy.ndarray
        Array the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    pos = x >= 0
    neg = x < 0
    y[pos] = x[pos] ** power
    y[neg] = -((-x[neg]) ** power)
    return y


def asinh_trans(data, param):
    """Inverse hyperbolic sinh transform: ``asinh(data * param) / param``.

    Parameters
    ----------
    data : array_like
        Data to transform.
    param : float
        Scaling parameter.

    Returns
    -------
    numpy.ndarray
        Transformed data, same shape as ``data``.
    """
    data = np.asarray(data, dtype=float)
    return np.arcsinh(data * param) / param


def asinh_data_trans(data, param=1.0):
    """Standardize ``data`` by its std (no demeaning), then apply :func:`asinh_trans`.

    The standard deviation is computed along the first axis, matching MATLAB's
    ``std(data)`` for matrix input (column-wise) and the whole vector for a
    vector input.

    Parameters
    ----------
    data : array_like
        Data to transform.
    param : float, optional
        Scaling parameter. Default ``1``.

    Returns
    -------
    numpy.ndarray
        Transformed data, same shape as ``data``.
    """
    data = np.asarray(data, dtype=float)
    # MATLAB std default normalises by N-1 (ddof=1).
    if data.ndim <= 1:
        std_dev = np.std(data, ddof=1)
    else:
        std_dev = np.std(data, axis=0, ddof=1)
    standard_data = data / std_dev
    return np.arcsinh(standard_data * param) / param


def fwhm2sigma(fwhm):
    """Convert a Gaussian FWHM to its standard deviation."""
    return np.asarray(fwhm, dtype=float) / np.sqrt(8 * np.log(2))


def sigma2fwhm(sigma):
    """Convert a Gaussian standard deviation to its FWHM."""
    return np.sqrt(8 * np.log(2)) * np.asarray(sigma, dtype=float)
