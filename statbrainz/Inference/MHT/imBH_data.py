"""imBH_data (mirrors StatBrainz/Inference/MHT/imBH_data.m)."""

import numpy as np
from scipy import stats

from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz.Inference.MHT.fdrBH import fdrBH

__all__ = ['imBH_data']


def imBH_data(data, mask):
    """BH FDR control on a t-statistic image computed from subject ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array.
    mask : numpy.ndarray
        Binary spatial mask.

    Returns
    -------
    rejection_ind : numpy.ndarray
        Binary image of rejected voxels.
    n_rejections : int
        Number of rejected voxels.
    """
    data = np.asarray(data, dtype=float)
    nsubj = data.shape[-1]
    data_tstat, _, _, _ = mvtstat(data)
    pvals = 2 * (1 - stats.t.cdf(np.abs(data_tstat), nsubj - 1))
    mask = np.asarray(mask)
    pvals2use = pvals[mask.astype(bool)]
    pval_rejection_ind, _, _, _ = fdrBH(pvals2use)
    if not pval_rejection_ind.any():
        rejection_ind = np.zeros(data.shape[:-1])
    else:
        maxp = np.max(pvals2use[pval_rejection_ind])
        rejection_ind = (pvals <= maxp + np.finfo(float).eps).astype(float)
    return rejection_ind, int(rejection_ind.sum())
