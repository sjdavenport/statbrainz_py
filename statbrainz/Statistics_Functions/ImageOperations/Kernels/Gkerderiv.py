"""Gkerderiv (mirrors StatBrainz/Statistics_Functions/ImageOperations/Kernels/Gkerderiv.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.FWHM2sigma import fwhm2sigma

__all__ = ['Gkerderiv']


def Gkerderiv(x, fwhm):
    """First derivative of the 1D Gaussian kernel (FWHM parameterisation)."""
    x = np.asarray(x, dtype=float)
    sigma2 = fwhm2sigma(fwhm) ** 2
    return (-x / sigma2) * np.exp(-(x**2) / (2 * sigma2)) / np.sqrt(
        2 * np.pi * sigma2
    )
