"""smooth_surface (mirrors StatBrainz/Surface/smooth_surface.m)."""

import numpy as np

from statbrainz.Surface.SurfStat.SurfStatSmooth import SurfStatSmooth
from .srf_fwhm2niters import srf_fwhm2niters

__all__ = ['smooth_surface']


def smooth_surface(srf, data, FWHM, metric="ones", niters=None):
    """Smooth per-vertex ``data`` on ``srf`` (recurses over hemispheres).

    Parameters
    ----------
    srf : dict
        Surface, or a bilateral ``{'lh':..., 'rh':...}`` dict.
    data : numpy.ndarray or dict
        Per-vertex data (or ``{'lh':..., 'rh':...}``).
    FWHM : float
        Smoothing FWHM.
    metric : {'ones', 'dist'}, optional
        Edge weighting.
    niters : int, optional
        Iterations; defaults to :func:`srf_fwhm2niters`.

    Returns
    -------
    numpy.ndarray or dict
        Smoothed data.
    """
    if "lh" in srf or "rh" in srf:
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in srf:
                out[hemi] = smooth_surface(srf[hemi], data[hemi], FWHM, metric)
        return out
    if niters is None:
        niters = srf_fwhm2niters(FWHM, srf)
    return SurfStatSmooth(srf, data, FWHM, metric, niters)
