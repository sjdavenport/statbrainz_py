"""prewhiten (mirrors StatBrainz/Inference/FirstLevel/prewhiten.m)."""

import numpy as np

__all__ = ['prewhiten']


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
