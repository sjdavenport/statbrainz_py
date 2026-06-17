"""srf_noise (mirrors StatBrainz/Surface/srf_noise.m)."""

import numpy as np

from .smooth_surface import smooth_surface

__all__ = ['srf_noise']


def srf_noise(srf, FWHM=0, nsubj=1, metric="ones", mask=None, rng=None):
    """Generate (optionally smoothed, masked) Gaussian noise on a surface.

    Parameters
    ----------
    srf : dict
        Surface (or bilateral dict).
    FWHM : float, optional
        Smoothing FWHM. ``0`` (default) leaves the noise unsmoothed.
    nsubj : int, optional
        Number of subjects (columns). Default ``1``.
    metric : {'ones', 'dist'}, optional
        Smoothing metric.
    mask : numpy.ndarray or dict, optional
        Per-vertex mask; if given, noise is masked (and re-normalised after
        smoothing).
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    numpy.ndarray or dict
        ``(nvertices, nsubj)`` noise (or bilateral dict).
    """
    if rng is None:
        rng = np.random.default_rng()

    if "lh" in srf or "rh" in srf:
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in srf:
                hemi_mask = mask[hemi] if mask is not None else None
                out[hemi] = srf_noise(srf[hemi], FWHM, nsubj, metric, hemi_mask, rng)
        return out

    data = rng.standard_normal((srf["nvertices"], nsubj))
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=float)
        data = data * mask_arr[:, None] if data.ndim == 2 else data * mask_arr
    if FWHM > 0:
        data = smooth_surface(srf, data.T, FWHM, metric).T if data.ndim == 2 else \
            smooth_surface(srf, data, FWHM, metric)
        if mask is not None:
            smoothed_mask = smooth_surface(srf, np.asarray(mask, dtype=float), FWHM, metric)
            data = data / (smoothed_mask[:, None] if data.ndim == 2 else smoothed_mask)
    return data
