"""gmcdf (mirrors StatBrainz/Statistics_Functions/Stats_functions/gmcdf.m)."""

import numpy as np
from scipy import stats

__all__ = ['gmcdf']


def _check_mixture(weights, means, std_devs):
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    std_devs = np.asarray(std_devs, dtype=float)
    if abs(weights.sum() - 1) > 1e-6:
        raise ValueError("Mixture weights must sum to 1.")
    if not (len(weights) == len(means) == len(std_devs)):
        raise ValueError(
            "Weights, means, and standard deviations must have the same length."
        )
    return weights, means, std_devs


def gmcdf(x, weights, means, std_devs):
    """CDF of a Gaussian mixture model.

    Parameters
    ----------
    x : array_like
        Evaluation point(s).
    weights : array_like
        Mixture weights (must sum to 1).
    means : array_like
        Component means.
    std_devs : array_like
        Component standard deviations.

    Returns
    -------
    numpy.ndarray
        Mixture CDF at ``x``.
    """
    weights, means, std_devs = _check_mixture(weights, means, std_devs)
    x = np.asarray(x, dtype=float)
    cdf = np.zeros_like(x, dtype=float)
    for w, mu, sd in zip(weights, means, std_devs):
        cdf = cdf + w * stats.norm.cdf(x, loc=mu, scale=sd)
    return cdf
