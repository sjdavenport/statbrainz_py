"""srf_scb2cope (mirrors StatBrainz/Inference/CopeSets/srf_scb2cope.m)."""

import numpy as np

from statbrainz.Surface.srf_contour import srf_contour

__all__ = ['srf_scb2cope']


def srf_scb2cope(srf, lower_band_im, upper_band_im, muhat, c):
    """Convert simultaneous confidence bands to cope sets on a surface.

    Parameters
    ----------
    srf : dict
        Surface structure (or bilateral dict with ``lh``/``rh``).
    lower_band_im, upper_band_im : numpy.ndarray or dict
        Lower/upper confidence bands over vertices.
    muhat : numpy.ndarray or dict
        Estimated mean field over vertices.
    c : float
        Threshold at which to generate the cope set.

    Returns
    -------
    lower_set, upper_set, contour, yellow_set : numpy.ndarray or dict
    """
    if "lh" in srf and "rh" in srf:
        lower_set, upper_set, yellow_set = {}, {}, {}
        for h in ("lh", "rh"):
            lower_set[h] = np.asarray(upper_band_im[h]) > c
            upper_set[h] = np.asarray(lower_band_im[h]) > c
            yellow_set[h] = np.asarray(muhat[h]) > c
        # srf_contour returns (inner, outer); the MATLAB code uses the inner
        # one-ring contour (its first output) thresholded at > 0.
        contour = {}
        for h in ("lh", "rh"):
            inner, _ = srf_contour(srf[h], np.asarray(muhat[h]) > c)
            contour[h] = inner > 0
        return lower_set, upper_set, contour, yellow_set

    lower_set = np.asarray(upper_band_im) > c
    upper_set = np.asarray(lower_band_im) > c
    yellow_set = np.asarray(muhat) > c
    inner, _ = srf_contour(srf, np.asarray(muhat) > c)
    contour = inner > 0
    return lower_set, upper_set, contour, yellow_set
