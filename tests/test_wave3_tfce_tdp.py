"""Tests for Wave 3 TFCE and cluster-TDP ports.

Numeric expectations cross-checked against MATLAB R2024b.
"""

import numpy as np
import pytest

import statbrainz as inf


# ----- tfce ----------------------------------------------------------------

def test_tfce_matches_matlab():
    img = np.zeros((5, 5))
    img[1:4, 1:4] = 1
    img[2, 2] = 3
    t = inf.tfce(img, 2, 0.5, 8, 0.5, 0)
    assert t[2, 2] == pytest.approx(12.625, abs=1e-3)
    assert t.sum() == pytest.approx(27.625, abs=1e-3)


def test_tfce_zero_image():
    img = np.zeros((4, 4))
    t = inf.tfce(img)
    assert np.all(t == 0)


def test_voxLCE_matches_matlab():
    assert inf.voxLCE(10, 4, 2, 0) == np.True_ or inf.voxLCE(10, 4, 2, 0)
    # array input
    out = inf.voxLCE(np.array([0.0, 10.0]), 4, 2, 0)
    assert out.tolist() == [False, True]


# ----- rkval ---------------------------------------------------------------

def test_rkval_matches_matlab():
    assert inf.rkval(10, 3) == pytest.approx(0.6970, abs=1e-4)
    assert inf.rkval(50, 3) == pytest.approx(0.5200, abs=1e-4)
    assert inf.rkval(20, 2) == pytest.approx(0.3333, abs=1e-4)


# ----- clustertp_lowerbound ------------------------------------------------

def test_clustertp_lowerbound_matches_matlab():
    xx, yy, zz = np.meshgrid(
        np.arange(1, 5), np.arange(1, 5), np.arange(1, 5), indexing="ij"
    )
    CL = {"x": xx.ravel(), "y": yy.ravel(), "z": zz.ravel()}
    assert inf.clustertp_lowerbound(CL, 5, 3) == 36


def test_clustertp_lowerbound_array_input():
    coords = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1]], dtype=float)
    lb = inf.clustertp_lowerbound(coords, 2, 3)
    assert isinstance(lb, int)


# ----- cluster_tp2tdp / clustertdp ----------------------------------------

def test_cluster_tp2tdp():
    tp = [10, 6]
    clusters = [np.zeros((20, 3)), np.zeros((12, 3))]
    tdp = inf.cluster_tp2tdp(tp, clusters)
    np.testing.assert_allclose(tdp, [0.5, 0.5])


def test_clustertdp_lowerbound():
    xx, yy, zz = np.meshgrid(
        np.arange(1, 5), np.arange(1, 5), np.arange(1, 5), indexing="ij"
    )
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(float)
    tdp, tp = inf.clustertdp([coords], 5)
    assert tp[0] == 36
    assert tdp[0] == pytest.approx(36 / 64)


def test_clustertdp_rejects_heuristic():
    with pytest.raises(NotImplementedError):
        inf.clustertdp([np.zeros((5, 3))], 3, method="heuristic")
