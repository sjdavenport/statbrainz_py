"""linear_template (mirrors StatBrainz/Inference/TDP_inference/templates/linear_template.m)."""

import numpy as np

__all__ = ['linear_template']


def linear_template(alpha, k, m):
    """Linear template ``alpha * (1:(k+1)) / m``.

    Parameters
    ----------
    alpha : float
        Confidence level in ``[0, 1]``.
    k : int
        Rejection set index.
    m : int
        Total number of hypotheses.

    Returns
    -------
    numpy.ndarray
        Vector of length ``k + 1``.
    """
    return alpha * np.arange(1, k + 2) / m
