"""lcdf (mirrors StatBrainz/Statistics_Functions/Stats_functions/lcdf.m)."""

import numpy as np

__all__ = ['lcdf']


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
