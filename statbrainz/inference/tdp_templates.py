"""TDP reference-family templates ported from StatBrainz Inference/TDP_inference.

Faithful ports of: linear_template, inverse_linear_template, get_pivotal_stats.

Reference: Blanchard, Neuvial & Roquain (2020), Annals of Statistics 48(3).
"""

import numpy as np

__all__ = ["linear_template", "inverse_linear_template", "get_pivotal_stats"]


def linear_template(alpha, k, m):
    """Linear template ``alpha * (1:(k+1)) / m``.

    Parameters
    ----------
    alpha : float
        Confidence level in ``[0, 1]``.
    k : int
        Rejection set index.
    m : int
        Total number of hypotheses.

    Returns
    -------
    numpy.ndarray
        Vector of length ``k + 1``.
    """
    return alpha * np.arange(1, k + 2) / m


def inverse_linear_template(y, k, m):
    """Inverse of the linear template: ``y * m / k``.

    Parameters
    ----------
    y : array_like
        Values to apply the inverse template to.
    k : int
        Rejection set index.
    m : int
        Total number of hypotheses.

    Returns
    -------
    numpy.ndarray
        Same shape as ``y``.
    """
    return np.asarray(y, dtype=float) * m / k


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
