"""srf_fdr_crs (mirrors StatBrainz/Inference/CopeSets/srf_fdr_crs.m)."""

import numpy as np

from .fdr_crs import fdr_crs

__all__ = ['srf_fdr_crs']


def srf_fdr_crs(data, mask, thresh, alpha_quant=0.05):
    """FDR-controlled confidence regions for a bilateral surface.

    Parameters
    ----------
    data : dict
        Fields ``lh`` and ``rh``, each ``[nvertices, nsubj]``.
    mask : dict
        Fields ``lh`` and ``rh``, each a length-``nvertices`` boolean array
        indicating the region of interest.
    thresh : float
        Excursion threshold (a single number).
    alpha_quant : float, optional
        FDR level. Default ``0.05``.

    Returns
    -------
    lower_band, upper_band : dict
        Lower/upper cope sets with ``lh``/``rh`` 0/1 arrays over vertices.
    """
    if np.ndim(thresh) != 0:
        raise ValueError("Thresh must be a single number")

    mask_lh = np.asarray(mask["lh"]).astype(bool).ravel()
    mask_rh = np.asarray(mask["rh"]).astype(bool).ravel()

    masked_lh = np.asarray(data["lh"])[mask_lh, :]
    masked_rh = np.asarray(data["rh"])[mask_rh, :]

    lower_band = {"lh": np.zeros(mask_lh.shape), "rh": np.zeros(mask_rh.shape)}
    upper_band = {"lh": np.zeros(mask_lh.shape), "rh": np.zeros(mask_rh.shape)}

    stacked = np.vstack([masked_lh, masked_rh])
    lower_out, upper_out = fdr_crs(stacked, thresh, alpha_quant)
    lower_out = lower_out.ravel()
    upper_out = upper_out.ravel()

    n_lh = int(mask_lh.sum())
    lower_band["lh"][mask_lh] = lower_out[:n_lh]
    lower_band["rh"][mask_rh] = lower_out[n_lh:]
    upper_band["lh"][mask_lh] = upper_out[:n_lh]
    upper_band["rh"][mask_rh] = upper_out[n_lh:]

    return lower_band, upper_band
