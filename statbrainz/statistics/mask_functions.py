"""Mask / array value utilities ported from StatBrainz Mask_functions.

Faithful ports of: nan2zero, zero2nan.
"""

import numpy as np

__all__ = ["nan2zero", "zero2nan"]


def nan2zero(vec, val=0.0):
    """Replace NaN entries of ``vec`` with ``val`` (returns a new array)."""
    vec = np.array(vec, dtype=float)
    vec[np.isnan(vec)] = val
    return vec


def zero2nan(vec):
    """Set (near-)zero entries of ``vec`` to NaN.

    Matches MATLAB: entries with ``abs(vec) <= 2*eps`` become NaN. Logical
    input is converted to float first.
    """
    vec = np.array(vec, dtype=float)
    vec[np.abs(vec) <= 2 * np.finfo(float).eps] = np.nan
    return vec
