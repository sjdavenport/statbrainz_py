"""GkerMV (mirrors StatBrainz/Statistics_Functions/ImageOperations/Kernels/GkerMV.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.FWHM2sigma import fwhm2sigma

__all__ = ['GkerMV']


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
