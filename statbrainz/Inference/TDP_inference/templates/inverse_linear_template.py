"""inverse_linear_template (mirrors StatBrainz/Inference/TDP_inference/templates/inverse_linear_template.m)."""

import numpy as np

__all__ = ['inverse_linear_template']


def inverse_linear_template(y, k, m):
    """Inverse of the linear template: ``y * m / k``.

    Parameters
    ----------
    y : array_like
        Values to apply the inverse template to.
    k : int
        Rejection set index.
    m : int
        Total number of hypotheses.

    Returns
    -------
    numpy.ndarray
        Same shape as ``y``.
    """
    return np.asarray(y, dtype=float) * m / k
