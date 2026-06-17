"""tstat2pval (mirrors StatBrainz/Statistics_Functions/Stats_functions/tstat2pval.m)."""

import numpy as np
from scipy import stats

__all__ = ['tstat2pval']


def tstat2pval(tstat, df, do2sample=True):
    """Convert t-statistics to p-values.

    Parameters
    ----------
    tstat : array_like
        t-statistic array.
    df : float
        Degrees of freedom.
    do2sample : bool, optional
        If ``True`` (default) two-sided p-values, else one-sided.

    Returns
    -------
    numpy.ndarray
        p-values, same shape as ``tstat``.
    """
    tstat = np.asarray(tstat, dtype=float)
    if do2sample:
        return 2 * (1 - stats.t.cdf(np.abs(tstat), df))
    return 1 - stats.t.cdf(tstat, df)
