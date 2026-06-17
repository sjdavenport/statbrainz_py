"""resample_srf (mirrors StatBrainz/Surface/resample_srf.m)."""

import numpy as np

from .resample_srf_nn import resample_srf_nn

__all__ = ['resample_srf']


def resample_srf(surface_data, srfin, srfout, intertype="nn"):
    """Resample ``surface_data`` from ``srfin`` onto ``srfout`` (nearest neighbour).

    Parameters
    ----------
    surface_data : numpy.ndarray or dict
        Per-vertex data on ``srfin`` (or bilateral dict).
    srfin, srfout : dict
        Source and target surfaces.
    intertype : {'nn'}, optional
        Interpolation type. Only nearest-neighbour is implemented (as in MATLAB).

    Returns
    -------
    numpy.ndarray or dict
        Resampled data on ``srfout``.
    """
    if isinstance(surface_data, dict) and ("lh" in surface_data or "rh" in surface_data):
        nn = resample_srf_nn(srfin, srfout)
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in surface_data:
                out[hemi] = np.asarray(surface_data[hemi])[nn[hemi]]
        return out
    if intertype in ("nn", "nearestneighbour"):
        nn_idx = resample_srf_nn(srfin, srfout)
        return np.asarray(surface_data)[nn_idx]
    raise NotImplementedError(f"intertype {intertype!r} not implemented")
