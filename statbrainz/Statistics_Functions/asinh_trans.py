"""asinh_trans (mirrors StatBrainz/Statistics_Functions/asinh_trans.m)."""

import numpy as np

__all__ = ['asinh_trans']


def asinh_trans(data, param):
    """Inverse hyperbolic sinh transform: ``asinh(data * param) / param``.

    Parameters
    ----------
    data : array_like
        Data to transform.
    param : float
        Scaling parameter.

    Returns
    -------
    numpy.ndarray
        Transformed data, same shape as ``data``.
    """
    data = np.asarray(data, dtype=float)
    return np.arcsinh(data * param) / param
