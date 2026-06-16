"""Statistical distribution helpers ported from StatBrainz Stats_functions.

Faithful ports of: lcdf, bernstd, distbn2pval, tstat2pval, pval2tstat.
MATLAB's norminv/tcdf/tinv are provided by ``scipy.stats``.
"""

import numpy as np
from scipy import stats

__all__ = [
    "lcdf",
    "bernstd",
    "distbn2pval",
    "tstat2pval",
    "pval2tstat",
    "gmcdf",
    "gmrnd",
]


def lcdf(x, mu, b):
    """CDF of the Laplace distribution with location ``mu`` and scale ``b``.

    Parameters
    ----------
    x : array_like
        Value(s) at which to evaluate the CDF.
    mu : float
        Location parameter (mean).
    b : float
        Scale parameter, must be positive.

    Returns
    -------
    numpy.ndarray
        CDF value(s) at ``x``.
    """
    if b <= 0:
        raise ValueError("Scale parameter b must be positive")
    x = np.asarray(x, dtype=float)
    cdf = np.where(
        x <= mu,
        0.5 * np.exp((x - mu) / b),
        1 - 0.5 * np.exp(-(x - mu) / b),
    )
    return cdf


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
