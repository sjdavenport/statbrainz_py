"""distbn2pval (mirrors StatBrainz/Statistics_Functions/Stats_functions/distbn2pval.m)."""

import numpy as np

__all__ = ['distbn2pval']


def distbn2pval(distbn, value_vec):
    """Empirical (right-tail) p-values of ``value_vec`` against ``distbn``.

    Each p-value is ``mean(distbn >= value)``.

    Parameters
    ----------
    distbn : array_like
        Reference distribution.
    value_vec : array_like
        Values to evaluate.

    Returns
    -------
    numpy.ndarray
        Empirical p-values, same length as ``value_vec``.
    """
    distbn = np.asarray(distbn, dtype=float)
    value_vec = np.atleast_1d(np.asarray(value_vec, dtype=float))
    nindist = distbn.size
    return np.array([np.sum(distbn >= v) / nindist for v in value_vec])
