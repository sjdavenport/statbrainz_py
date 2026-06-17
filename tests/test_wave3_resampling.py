"""Tests for Wave 3 perm_thresh (global-maximum path)."""

import numpy as np
import pytest

import statbrainz as inf


def test_perm_thresh_runs_and_thresholds():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 6, 15))
    data[2:4, 2:4, :] += 2
    im, thr, maxima = inf.perm_thresh(
        data, niters=200, show_loader=False, rng=rng
    )
    assert im.shape == (6, 6)
    assert maxima.shape == (200,)
    assert np.isfinite(thr)


def test_perm_thresh_twosample():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 5, 12))
    im, thr, maxima = inf.perm_thresh(
        data, twosample=True, niters=100, show_loader=False, rng=rng
    )
    assert im.dtype == bool


def test_perm_thresh_mean_stat():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((5, 5, 12))
    im, thr, maxima = inf.perm_thresh(
        data, stat="Z", niters=100, show_loader=False, rng=rng
    )
    assert np.isfinite(thr)


def test_perm_thresh_mask_unsupported():
    data = np.random.default_rng(0).standard_normal((4, 4, 10))
    with pytest.raises(NotImplementedError):
        inf.perm_thresh(data, mask=np.ones((4, 4)), show_loader=False)
