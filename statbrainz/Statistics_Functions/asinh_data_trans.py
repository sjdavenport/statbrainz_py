"""asinh_data_trans (mirrors StatBrainz/Statistics_Functions/asinh_data_trans.m)."""

import numpy as np

__all__ = ['asinh_data_trans']


def asinh_data_trans(data, param=1.0):
    """Standardize ``data`` by its std (no demeaning), then apply :func:`asinh_trans`.

    The standard deviation is computed along the first axis, matching MATLAB's
    ``std(data)`` for matrix input (column-wise) and the whole vector for a
    vector input.

    Parameters
    ----------
    data : array_like
        Data to transform.
    param : float, optional
        Scaling parameter. Default ``1``.

    Returns
    -------
    numpy.ndarray
        Transformed data, same shape as ``data``.
    """
    data = np.asarray(data, dtype=float)
    # MATLAB std default normalises by N-1 (ddof=1).
    if data.ndim <= 1:
        std_dev = np.std(data, ddof=1)
    else:
        std_dev = np.std(data, axis=0, ddof=1)
    standard_data = data / std_dev
    return np.arcsinh(standard_data * param) / param
