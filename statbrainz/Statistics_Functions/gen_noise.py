"""gen_noise (mirrors StatBrainz/Statistics_Functions/gen_noise.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.fast_conv import fast_conv

__all__ = ['gen_noise']


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
