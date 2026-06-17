"""Tests for Wave 3 MHT additions: spatialBH, localized_vi."""

import numpy as np

from statbrainz import inference as inf


def test_spatialBH_recovers_signal():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 8, 20))
    data[2:5, 2:5, :] += 3
    rej, nrej = inf.spatialBH(data, FWHM=np.nan)
    assert rej.shape == (8, 8)
    assert nrej >= 9  # the 3x3 signal block


def test_spatialBH_with_smoothing_runs():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((8, 8, 20))
    data[3:6, 3:6, :] += 3
    rej, nrej = inf.spatialBH(data, FWHM=2)
    assert rej.shape == (8, 8)
    assert nrej >= 1


def test_localized_vi():
    tstat = np.zeros((5, 5))
    tstat[1, 1] = 4.0  # region A peak
    tstat[3, 3] = 1.0  # region B peak
    regionA = np.zeros((5, 5)); regionA[0:2, 0:2] = 1
    regionB = np.zeros((5, 5)); regionB[3:5, 3:5] = 1
    null = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # 6 null maxima
    pvals, maxes = inf.localized_vi(tstat, [regionA, regionB], null)
    np.testing.assert_array_equal(maxes, [4.0, 1.0])
    # region A max (4.0) exceeds all nulls -> p = 0; region B (1.0) -> 5/6
    assert pvals[0] == 0.0
    assert pvals[1] == 5 / 6
