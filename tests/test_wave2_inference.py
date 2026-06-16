"""Tests for Wave 2 inference: cluster components, imBH, Xgen2.

Cluster/imBH expectations cross-checked against MATLAB R2024b.
"""

import numpy as np
import pytest

from statbrainz import inference as inf


def _three_cluster_image():
    data = np.zeros((6, 6))
    data[0:2, 0:2] = 5  # 4 voxels
    data[3:6, 4:6] = 5  # 6 voxels
    data[5, 0] = 5      # 1 voxel
    return data


def test_numOfConComps_matches_matlab():
    data = _three_cluster_image()
    nc, occ, sizes, idx = inf.numOfConComps(data, 0, 4)
    assert nc == 3
    np.testing.assert_array_equal(sizes, [1, 4, 6])
    np.testing.assert_array_equal(occ, [1, 1, 1])
    sized4 = [np.sort(i) for i in idx if len(i) == 4][0]
    np.testing.assert_array_equal(sized4, [1, 2, 7, 8])


def test_cluster_im_matches_matlab():
    data = _three_cluster_image()
    _, _, _, idx = inf.numOfConComps(data, 0, 4)
    sci, sc, scv = inf.cluster_im(data.shape, idx, 3)
    assert int(sci.sum()) == 10
    found = np.sort(np.nonzero(sci.ravel(order="F"))[0] + 1)
    np.testing.assert_array_equal(found, [1, 2, 7, 8, 28, 29, 30, 34, 35, 36])


def test_cluster_im_no_survivors():
    data = _three_cluster_image()
    _, _, _, idx = inf.numOfConComps(data, 0, 4)
    sci, sc, scv = inf.cluster_im(data.shape, idx, 100)
    assert sci.sum() == 0
    assert scv == []
    assert sc.size == 0


def test_getlargestcluster_matches_matlab():
    data = _three_cluster_image()
    lc = inf.getlargestcluster(data > 0)
    assert int(lc.sum()) == 6


def test_index2mask_roundtrip():
    mask = inf.index2mask([1, 2, 7, 8], (6, 6))
    assert int(mask.sum()) == 4
    np.testing.assert_array_equal(
        np.nonzero(mask.ravel(order="F"))[0] + 1, [1, 2, 7, 8]
    )


def test_bestclusterslice():
    im = np.zeros((4, 4, 4))
    im[:, :, 2] = 1  # all mass on z-slice index 3 (1-based)
    locs, totals = inf.bestclusterslice(3, im)
    np.testing.assert_array_equal(locs, [3])
    assert totals[2] == 16


def test_imBH_matches_matlab():
    pvals = np.ones((4, 4)) * 0.5
    pvals[0, 0] = 0.001
    pvals[1, 1] = 0.009
    pvals[2, 2] = 0.02
    ri, nr = inf.imBH(pvals, np.ones((4, 4)))
    assert nr == 1


def test_imBH_data_runs():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 6, 10))
    data[0:3, 0:3, :] += 3  # strong signal in a corner
    mask = np.ones((6, 6))
    ri, nr = inf.imBH_data(data, mask)
    assert ri.shape == (6, 6)
    assert nr >= 9  # the 3x3 signal block should be rejected


# ----- Xgen2 ---------------------------------------------------------------

def test_Xgen2_zero_rho_is_iid():
    rng = np.random.default_rng(0)
    X = inf.Xgen2(100, 5, 0, rng=rng)
    assert X.shape == (100, 5)


def test_Xgen2_ar1_shape_and_correlation():
    rng = np.random.default_rng(1)
    X = inf.Xgen2(2000, 8, 0.8, method="ar1", rng=rng)
    assert X.shape == (2000, 8)
    # adjacent columns correlated
    c = np.corrcoef(X[:, 0], X[:, 1])[0, 1]
    assert c > 0.4


def test_Xgen2_unknown_method():
    with pytest.raises(ValueError):
        inf.Xgen2(10, 4, 0.5, method="nope")
