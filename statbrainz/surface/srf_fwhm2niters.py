"""srf_fwhm2niters (mirrors StatBrainz/Surface/srf_fwhm2niters.m)."""

import numpy as np

from .srf_face_area import srf_face_area

__all__ = ['srf_fwhm2niters']


def srf_fwhm2niters(FWHM, srf, fudge_factor=None):
    """Convert a smoothing FWHM to a number of edge-averaging iterations.

    Parameters
    ----------
    FWHM : float
        Target FWHM.
    srf : dict
        Surface.
    fudge_factor : float, optional
        Defaults to ``1.478 * (69/40)`` (matches FreeSurfer smoothing).

    Returns
    -------
    int
        Number of iterations.
    """
    if fudge_factor is None:
        fudge_factor = 1.478 * (69 / 40)
    face_area, _ = srf_face_area(srf)
    surface_area = face_area.sum()
    area_per_vertex = surface_area / srf["nvertices"]
    numerator = 1.14 * 4 * np.pi * FWHM**2
    denominator = 8 * np.log(2) * 7 * area_per_vertex
    return int(np.floor(fudge_factor * numerator / denominator + 0.5))
