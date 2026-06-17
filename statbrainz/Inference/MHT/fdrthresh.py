"""fdrthresh (mirrors StatBrainz/Inference/MHT/fdrthresh.m)."""

import numpy as np
from scipy import stats

__all__ = ['fdrthresh']


def fdrthresh(pvals, pval_rejection_ind, df):
    """t-statistic threshold for the largest rejected p-value of a BH procedure.

    Parameters
    ----------
    pvals : numpy.ndarray
        Vector of p-values.
    pval_rejection_ind : numpy.ndarray
        Boolean array indicating which p-values are rejected.
    df : float
        Degrees of freedom.

    Returns
    -------
    float
        ``tinv(maxp, df)`` for the largest rejected p-value.
    """
    pvals = np.asarray(pvals, dtype=float)
    maxp = np.max(pvals[np.asarray(pval_rejection_ind, dtype=bool)])
    return stats.t.ppf(maxp, df)
