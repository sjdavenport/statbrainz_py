"""sigma2FWHM (mirrors StatBrainz/Brain_Functions/sigma2FWHM.m)."""

import numpy as np

__all__ = ['sigma2fwhm']


def sigma2fwhm(sigma):
    """Convert a Gaussian standard deviation to its FWHM."""
    return np.sqrt(8 * np.log(2)) * np.asarray(sigma, dtype=float)
