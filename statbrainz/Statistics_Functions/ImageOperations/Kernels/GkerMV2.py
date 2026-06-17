"""GkerMV2 (mirrors StatBrainz/Statistics_Functions/ImageOperations/Kernels/GkerMV2.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.FWHM2sigma import fwhm2sigma

__all__ = ['GkerMV2']


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
