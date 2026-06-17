"""apower (mirrors StatBrainz/Statistics_Functions/apower.m)."""

import numpy as np

__all__ = ['apower']


def apower(x, power=0.5):
    """Raise ``x`` to ``power`` preserving sign: ``sign(x) * |x| ** power``.

    Parameters
    ----------
    x : array_like
        A numeric array.
    power : float, optional
        The exponent to apply. Default ``0.5``.

    Returns
    -------
    numpy.ndarray
        Array the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    pos = x >= 0
    neg = x < 0
    y[pos] = x[pos] ** power
    y[neg] = -((-x[neg]) ** power)
    return y
