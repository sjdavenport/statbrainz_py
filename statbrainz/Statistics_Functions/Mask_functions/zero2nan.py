"""zero2nan (mirrors StatBrainz/Statistics_Functions/Mask_functions/zero2nan.m)."""

import numpy as np

__all__ = ['zero2nan']


def zero2nan(vec):
    """Set (near-)zero entries of ``vec`` to NaN.

    Matches MATLAB: entries with ``abs(vec) <= 2*eps`` become NaN. Logical
    input is converted to float first.
    """
    vec = np.array(vec, dtype=float)
    vec[np.abs(vec) <= 2 * np.finfo(float).eps] = np.nan
    return vec
