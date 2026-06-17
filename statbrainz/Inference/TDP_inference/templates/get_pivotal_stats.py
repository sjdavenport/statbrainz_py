"""get_pivotal_stats (mirrors StatBrainz/Inference/TDP_inference/templates/get_pivotal_stats.m)."""

import numpy as np

from .inverse_linear_template import inverse_linear_template

__all__ = ['get_pivotal_stats']


def get_pivotal_stats(p0, inverse_template=inverse_linear_template, K=-1):
    """Pivotal statistics from a matrix of permuted null p-values.

    Parameters
    ----------
    p0 : numpy.ndarray
        ``B x p`` matrix of null p-values from ``B`` permutations/bootstraps
        for ``p`` hypotheses.
    inverse_template : callable, optional
        Inverse template function ``f(values, k, m)``. Default
        :func:`inverse_linear_template`.
    K : int, optional
        For JER control over ``1:K``. If ``< 0`` it is set to ``p``.

    Returns
    -------
    numpy.ndarray
        Length-``B`` vector of pivotal statistics (row-wise minima).
    """
    p0 = np.sort(np.asarray(p0, dtype=float), axis=1)
    B, p = p0.shape

    # MATLAB column index i (1-based) -> inverse_template(col, i+1, p).
    # In 0-based Python, column j corresponds to MATLAB i = j+1, so k = j+2.
    cols = [inverse_template(p0[:, j], j + 2, p) for j in range(p)]
    tk_inv_all = np.column_stack(cols)

    if K < 0:
        K = tk_inv_all.shape[1]

    return np.min(tk_inv_all[:, :K], axis=1)
