"""bernstd (mirrors StatBrainz/Statistics_Functions/Stats_functions/bernstd.m)."""

import numpy as np
from scipy import stats

__all__ = ['bernstd']


def bernstd(p, N, level=0.95):
    """Bernoulli (CLT) standard-error confidence intervals.

    Parameters
    ----------
    p : array_like
        Proportion(s) in ``[0, 1]``.
    N : int
        Sample size.
    level : float, optional
        Confidence level. Default ``0.95``.

    Returns
    -------
    interval : numpy.ndarray
        ``2 x len(p)`` array; row 0 lower bound, row 1 upper bound.
    std_error : numpy.ndarray
        ``len(p)`` vector of standard errors.
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    z = stats.norm.ppf(1 - (1 - level) / 2)
    std_error = np.sqrt(p * (1 - p)) * z / np.sqrt(N)
    interval = np.vstack([p - std_error, p + std_error])
    return interval, std_error
