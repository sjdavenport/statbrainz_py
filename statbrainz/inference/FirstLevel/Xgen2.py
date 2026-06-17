"""Xgen2 (mirrors StatBrainz/Inference/FirstLevel/Xgen2.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.fast_conv import fast_conv

__all__ = ['Xgen2']


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
