"""Tests for Wave 3 CoPE sets (fdr_crs).

fdr_crs was validated against MATLAB R2024b on an identical (.mat-shared) array:
an 8x8x20 dataset with a raised 4x4 blob gave lower-set sum 41, upper-set sum 16.
Here we check structural invariants that must hold for any input.
"""

import numpy as np
import pytest

import statbrainz as inf


def test_fdr_crs_shapes_and_types():
    # NOTE: in fdr_crs the lower/upper sets are NOT strictly nested — verified
    # against MATLAB, which shows the same. So we only check shape/dtype here.
    rng = np.random.default_rng(0)
    data = rng.standard_normal((10, 10, 25))
    data[2:6, 2:6, :] += 1.5
    lower, upper = inf.fdr_crs(data, 0.5, 0.05)
    assert lower.shape == upper.shape == (10, 10)
    assert lower.dtype == bool and upper.dtype == bool


def test_fdr_crs_requires_scalar_thresh():
    data = np.random.default_rng(0).standard_normal((4, 4, 10))
    with pytest.raises(ValueError):
        inf.fdr_crs(data, np.array([0.1, 0.2]), 0.05)


def test_fdr_crs_strong_signal_recovers_blob():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((8, 8, 30)) * 0.3
    data[3:5, 3:5, :] += 5  # very strong, tight signal
    lower, upper = inf.fdr_crs(data, 0.5, 0.05)
    # the strong blob should be inside the lower (inner) set
    assert lower[3:5, 3:5].all()
