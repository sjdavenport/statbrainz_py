"""spatialBH (mirrors StatBrainz/Inference/MHT/spatialBH.m)."""

import numpy as np
from scipy import stats

from statbrainz.Statistics_Functions.ImageOperations.fast_conv import fast_conv
from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz.Inference.MHT.fdrBH import fdrBH

__all__ = ['spatialBH']


def spatialBH(data, FWHM=np.nan):
    """Spatial BH: smooth, compute a t-image, and run BH on its p-values.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array.
    FWHM : float, optional
        Smoothing FWHM applied before the t-statistic. ``nan`` (default) skips
        smoothing.

    Returns
    -------
    rej_locs : numpy.ndarray
        Boolean rejection image.
    nrejections : int
        Number of rejections.
    """
    data = np.asarray(data, dtype=float)
    nsubj = data.shape[-1]
    D = data.ndim - 1
    if not np.isnan(FWHM):
        data, _ = fast_conv(data, FWHM, D)
    tstateval, _, _, _ = mvtstat(data)
    pvals = 1 - stats.t.cdf(tstateval, nsubj - 1)
    rej_locs, nrejections, _, _ = fdrBH(pvals)
    return rej_locs, nrejections
