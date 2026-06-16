"""Tests for the rest of Wave 1: brain vectorization, conversions, MHT, TDP.

Expected values cross-checked against MATLAB R2024b output.
"""

import numpy as np
import pytest

from statbrainz import brain, statistics as st, inference as inf


# ----- brain.vec_data / unwrap (column-major faithful) ---------------------

def test_vec_data_matches_matlab():
    a = np.arange(1, 19).reshape((3, 3, 2), order="F").astype(float)
    mask = np.zeros((3, 3))
    mask[0, 0] = 1
    mask[1, 2] = 1
    mask[2, 1] = 1
    out = brain.vec_data(a, mask)
    np.testing.assert_array_equal(out, [[1, 10], [6, 15], [8, 17]])


def test_vec_data_unwrap_roundtrip():
    rng = np.random.default_rng(1)
    mask = rng.random((8, 9, 7)) > 0.5
    data = rng.standard_normal((8, 9, 7, 4))
    v = brain.vec_data(data, mask)
    back = brain.unwrap(v, mask)
    # in-mask voxels recovered exactly, out-of-mask are zero
    np.testing.assert_allclose(back[mask], data[mask])
    assert np.all(back[~mask] == 0)


def test_unwrap_cell_list():
    mask = np.array([[1, 0], [0, 1]])
    data = [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])]
    out = brain.unwrap(data, mask)
    assert isinstance(out, list) and len(out) == 2
    np.testing.assert_array_equal(out[0][..., 0], [[1, 0], [0, 2]])


# ----- brain.conversions ---------------------------------------------------

def test_plane2index():
    idx = brain.plane2index([0, 0, 50])  # fix 3rd axis at voxel 50 (1-based)
    assert idx == (slice(None), slice(None), 49)
    idx4 = brain.plane2index([0, 12, 0], extra=True)
    assert idx4 == (slice(None), 11, slice(None), slice(None))


def test_nifti_type():
    out = brain.nifti_type(["nii", ".nii", ".nii.gz.asdf"])
    np.testing.assert_array_equal(out, [0, 2, 1])


# ----- statistics.square_signal / mask_bounds ------------------------------

def test_square_signal_matches_matlab():
    s = st.square_signal([5, 5], 1, [[3, 3]])
    assert int(s.sum()) == 9
    flat = s.ravel(order="F")
    np.testing.assert_array_equal(
        np.nonzero(flat)[0] + 1, [7, 8, 9, 12, 13, 14, 17, 18, 19]
    )


def test_mask_bounds_matches_matlab():
    m2 = np.zeros((6, 6))
    m2[1:4, 2:5] = 1
    b, bm = st.mask_bounds(m2)
    assert (b[0].start, b[0].stop) == (1, 4)
    assert (b[1].start, b[1].stop) == (2, 5)
    assert bm.shape == (3, 3)
    assert bm.all()


def test_mask_bounds_padding():
    m = np.zeros((10,))
    m[4:6] = 1
    b, _ = st.mask_bounds(m, padding=2)
    assert (b[0].start, b[0].stop) == (2, 8)


def test_pad_im_pad_and_crop():
    im = np.ones((3, 3))
    padded = st.pad_im(im, [1, 1])
    assert padded.shape == (5, 5)
    assert padded[0, 0] == 0 and padded[2, 2] == 1
    cropped = st.pad_im(np.arange(27).reshape(3, 3, 3), [-1, -1, -1])
    assert cropped.shape == (1, 1, 1)


# ----- inference.mht -------------------------------------------------------

def test_fdrBH_matches_matlab():
    pv = np.array([0.001, 0.7, 0.02, 0.5, 0.009, 0.8, 0.04, 0.2, 0.6, 0.95])
    ri, nr, rl, mp = inf.fdrBH(pv, 0.1)
    assert nr == 4
    np.testing.assert_array_equal(rl + 1, [1, 3, 5, 7])
    assert mp == pytest.approx(0.04)


def test_fdrBH_no_rejections():
    pv = np.array([0.9, 0.8, 0.95])
    ri, nr, rl, mp = inf.fdrBH(pv, 0.05)
    assert nr == 0
    assert not ri.any()
    assert np.isnan(mp)


def test_fdp_calc():
    rej = np.array([[1, 1], [0, 1]])
    sig = np.array([[1, 0], [0, 1]])
    # rejections at 3 locs, 1 of them outside signal -> fdp 1/3
    assert inf.fdp_calc(rej, sig) == pytest.approx(1 / 3)
    assert inf.fdp_calc(np.zeros((2, 2)), sig) == 0.0


# ----- inference.tdp_templates ---------------------------------------------

def test_linear_template():
    np.testing.assert_allclose(
        inf.linear_template(0.4, 5, 1000), 0.4 * np.arange(1, 7) / 1000
    )


def test_inverse_linear_template_inverts():
    alpha, k, m = 0.3, 4, 500
    t = inf.linear_template(alpha, k, m)  # length k+1
    # element at position k (1-based) maps back to alpha
    back = inf.inverse_linear_template(t[k - 1], k, m)
    assert back == pytest.approx(alpha)


def test_get_pivotal_stats_shape_and_value():
    p0 = np.array([[0.2, 0.5, 0.9], [0.1, 0.4, 0.8]])
    out = inf.get_pivotal_stats(p0)
    assert out.shape == (2,)
    # row 0 sorted = [0.2,0.5,0.9]; inverse uses k=j+2, m=3
    expected_row0 = min(0.2 * 3 / 2, 0.5 * 3 / 3, 0.9 * 3 / 4)
    assert out[0] == pytest.approx(expected_row0)
