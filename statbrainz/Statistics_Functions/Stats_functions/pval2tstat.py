"""pval2tstat (mirrors StatBrainz/Statistics_Functions/Stats_functions/pval2tstat.m)."""

import numpy as np
from scipy import stats

__all__ = ['pval2tstat']


def pval2tstat(pval, df):
    """Convert p-values to t-statistics via the inverse t-distribution.

    Mirrors MATLAB ``-tinv(pval, df)``.

    Parameters
    ----------
    pval : array_like
        p-value array.
    df : float
        Degrees of freedom.

    Returns
    -------
    numpy.ndarray
        t-statistics, same shape as ``pval``.
    """
    pval = np.asarray(pval, dtype=float)
    return -stats.t.ppf(pval, df)
