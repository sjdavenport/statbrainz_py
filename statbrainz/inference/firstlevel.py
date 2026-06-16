"""First-level GLM helpers ported from StatBrainz Inference/FirstLevel.

Faithful ports of: prewhiten, Xgen2.

NOTE: the MATLAB ``prewhiten`` has a known quirk — its loop overwrites
``data_pw`` each iteration rather than accumulating, and ``cov`` of a single
time series row is a scalar. This port reproduces the literal final-iteration
behaviour.
"""

import numpy as np

from ..statistics.convolution import fast_conv

__all__ = ["prewhiten", "Xgen2"]


def prewhiten(data):
    """Reproduce MATLAB ``prewhiten`` (literal behaviour).

    Returns the last row divided by its standard deviation (ddof=1).

    Parameters
    ----------
    data : numpy.ndarray
        ``n x T`` array of time series (rows).

    Returns
    -------
    numpy.ndarray
        The last row divided by its standard deviation.
    """
    data = np.asarray(data, dtype=float)
    ts = data[-1, :]
    ts_cov = np.cov(ts)
    return ts / np.sqrt(ts_cov)


def Xgen2(n, m, rho, method="Gaussian", rng=None):
    """Generate an ``n x m`` correlated design matrix.

    Parameters
    ----------
    n : int
        Number of rows (observations).
    m : int
        Number of columns (variables).
    rho : float or sequence of float
        Correlation parameter (per-block for ``'ar1mix'``).
    method : str, optional
        One of ``'Gaussian'``, ``'ar1'``, ``'ar1mix'``, ``'equi'``, ``'012'``,
        ``'012equi'``. Default ``'Gaussian'``.
    rng : numpy.random.Generator, optional
        Random generator. Default ``np.random.default_rng()``.

    Returns
    -------
    numpy.ndarray
        The ``n x m`` design matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    rho_arr = np.atleast_1d(np.asarray(rho, dtype=float))
    if rho_arr.size == 1 and rho_arr[0] == 0:
        return rng.standard_normal((n, m))

    if method == "Gaussian":
        noise = rng.standard_normal((n, m))
        X, ss = fast_conv(noise, float(rho_arr[0]), 1)
        return X / np.sqrt(ss)

    if method == "ar1":
        r = float(rho_arr[0])
        X = np.zeros((n, m))
        X[:, 0] = rng.standard_normal(n)
        for i in range(1, m):
            X[:, i] = r * X[:, i - 1] + rng.standard_normal(n)
        return X * np.sqrt(1 - r**2)

    if method == "ar1mix":
        half = m // 2
        blocks = []
        for r in (float(rho_arr[0]), float(rho_arr[1])):
            Xb = np.zeros((n, half))
            Xb[:, 0] = rng.standard_normal(n)
            for i in range(1, half):
                Xb[:, i] = r * Xb[:, i - 1] + rng.standard_normal(n)
            Xb = Xb / np.sqrt(1 / (1 - r**2))
            blocks.append(Xb)
        return np.hstack(blocks)

    if method == "equi":
        r = float(rho_arr[0])
        X = np.zeros((n, m))
        w = rng.standard_normal(n)
        for i in range(n):
            X[i, :] = np.sqrt(r) * w[i] + rng.standard_normal(m) * np.sqrt(1 - r)
        return X

    if method == "012":
        r = float(rho_arr[0])
        pbin = 0.5
        p1 = pbin + r * (1 - pbin)
        p0 = (1 - p1) * pbin / (1 - pbin)
        X1 = np.zeros((n, m))
        X2 = np.zeros((n, m))
        X1[:, 0] = rng.binomial(1, pbin, n)
        for j in range(1, m):
            pvec = (p1 - p0) * X1[:, j - 1] + p0
            X1[:, j] = rng.binomial(1, pvec)
        X2[:, 0] = rng.binomial(1, pbin, n)
        for j in range(1, m):
            pvec = (p1 - p0) * X2[:, j - 1] + p0
            X2[:, j] = rng.binomial(1, pvec)
        return (X1 + X2 - 2 * pbin) / np.sqrt(2 * pbin * (1 - pbin))

    if method == "012equi":
        Z1 = Xgen2(n, m, rho, "equi", rng)
        Z2 = Xgen2(n, m, rho, "equi", rng)
        return (Z1 > 0).astype(float) + (Z2 > 0).astype(float)

    raise ValueError(f"Unknown method: {method!r}")
