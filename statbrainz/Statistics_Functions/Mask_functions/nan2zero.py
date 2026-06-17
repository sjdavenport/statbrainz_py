"""nan2zero (mirrors StatBrainz/Statistics_Functions/Mask_functions/nan2zero.m)."""

import numpy as np

__all__ = ['nan2zero']


def nan2zero(vec, val=0.0):
    """Replace NaN entries of ``vec`` with ``val`` (returns a new array)."""
    vec = np.array(vec, dtype=float)
    vec[np.isnan(vec)] = val
    return vec
