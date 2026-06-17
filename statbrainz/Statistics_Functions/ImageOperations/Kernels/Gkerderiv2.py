"""Gkerderiv2 (mirrors StatBrainz/Statistics_Functions/ImageOperations/Kernels/Gkerderiv2.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.FWHM2sigma import fwhm2sigma

__all__ = ['Gkerderiv2']


def Gkerderiv2(x, fwhm):
    """Second derivative of the 1D Gaussian kernel (FWHM parameterisation)."""
    x = np.asarray(x, dtype=float)
    sigma2 = fwhm2sigma(fwhm) ** 2
    return (-1 / sigma2 + x**2 / sigma2**2) * np.exp(
        -(x**2) / (2 * sigma2)
    ) / np.sqrt(2 * np.pi * sigma2)
