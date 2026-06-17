"""Coverage-of-probability (CoPE) confidence sets ported from StatBrainz.

Faithful port of: fdr_crs.

NOT ported (broken / missing upstream dependencies in the MATLAB source):
* scopes, scopes_lm  -> call fastperm / fastperm_mean / fastlmperm /
  lmthresh2scb, none of which exist anywhere in the StatBrainz MATLAB package.
* fdr_simul_cs       -> the MATLAB source is incomplete (an empty ``for`` loop
  and references to undefined ``thresh`` / ``data_tstat``).
The display functions (cope_display, srf_cope_*) are Wave 5 (viewing).
"""

import numpy as np
from scipy import stats

from ..statistics.mvtstat import mvtstat
from .mht import fdrBH

__all__ = ["fdr_crs"]


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
