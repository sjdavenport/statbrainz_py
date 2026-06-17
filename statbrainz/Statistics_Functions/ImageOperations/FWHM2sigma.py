"""FWHM2sigma (mirrors StatBrainz/Statistics_Functions/ImageOperations/FWHM2sigma.m)."""

import numpy as np

__all__ = ['fwhm2sigma']


def fwhm2sigma(fwhm):
    """Convert a Gaussian FWHM to its standard deviation."""
    return np.asarray(fwhm, dtype=float) / np.sqrt(8 * np.log(2))
