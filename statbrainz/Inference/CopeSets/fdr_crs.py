"""fdr_crs (mirrors StatBrainz/Inference/CopeSets/fdr_crs.m)."""

import numpy as np
from scipy import stats

from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz.Inference.MHT.fdrBH import fdrBH

__all__ = ['fdr_crs']


def fdr_crs(data, thresh, alpha_quant=0.05):
    """FDR-controlled confidence regions for an excursion set ``{mu > thresh}``.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array.
    thresh : float
        Excursion threshold (a single number).
    alpha_quant : float, optional
        FDR level. Default ``0.05``.

    Returns
    -------
    lower_set : numpy.ndarray
        Boolean inner confidence set.
    upper_set : numpy.ndarray
        Boolean outer confidence set.
    """
    if np.ndim(thresh) != 0:
        raise ValueError("Thresh must be a single number")
    data = np.asarray(data, dtype=float)
    nsubj = data.shape[-1]

    data_tstat, _, data_std, _ = mvtstat(data - thresh)
    pvals = 1 - stats.t.cdf(data_tstat, nsubj - 1)

    pvals2use = pvals[data_std > np.finfo(float).eps].ravel(order="F")
    allpvals = np.column_stack([pvals2use, 1 - pvals2use])
    pval_rejection_ind, _, _, _ = fdrBH(allpvals, 2 * alpha_quant)
    if pval_rejection_ind.any():
        maxp = np.max(allpvals[pval_rejection_ind])
    else:
        maxp = -np.inf

    eps = np.finfo(float).eps
    upper_set = pvals <= maxp + eps
    lower_set = ~((1 - pvals) <= maxp + eps)
    return lower_set, upper_set
