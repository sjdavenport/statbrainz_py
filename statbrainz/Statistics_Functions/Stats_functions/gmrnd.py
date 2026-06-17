"""gmrnd (mirrors StatBrainz/Statistics_Functions/Stats_functions/gmrnd.m)."""

import numpy as np
from scipy import stats

__all__ = ['gmrnd']


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


def gmrnd(num_samples, weights, means, std_devs, rng=None):
    """Draw random samples from a Gaussian mixture model.

    Parameters
    ----------
    num_samples : int
        Number of samples.
    weights : array_like
        Mixture weights (must sum to 1).
    means : array_like
        Component means.
    std_devs : array_like
        Component standard deviations.
    rng : numpy.random.Generator, optional
        Random generator (for reproducibility). Default ``np.random.default_rng()``.

    Returns
    -------
    numpy.ndarray
        ``num_samples`` vector of samples.
    """
    weights, means, std_devs = _check_mixture(weights, means, std_devs)
    if rng is None:
        rng = np.random.default_rng()
    num_components = len(weights)
    data = np.zeros(num_samples)
    component_indices = rng.choice(num_components, size=num_samples, p=weights)
    for i in range(num_components):
        sel = component_indices == i
        n_i = int(sel.sum())
        if n_i:
            data[sel] = rng.normal(means[i], std_devs[i], size=n_i)
    return data
