"""Tests for Wave 2: kernels, fast_conv, morphology, mvtstat, gen_noise.

Numeric expectations cross-checked against MATLAB R2024b.
"""

import numpy as np
import pytest

import statbrainz as st


# ----- kernels -------------------------------------------------------------

def test_Gker_value_integrates_to_one():
    x = np.arange(-30, 30, 0.01)
    val, _, _ = st.Gker(x, 5.0)
    assert np.trapezoid(val, x) == pytest.approx(1.0, abs=1e-3)


def test_Gker_derivatives_match_finite_diff():
    x = np.array([1.3])
    h = 1e-6
    v0, d1, d2 = st.Gker(x, 4.0)
    vp, _, _ = st.Gker(x + h, 4.0)
    vm, _, _ = st.Gker(x - h, 4.0)
    assert d1[0] == pytest.approx((vp[0] - vm[0]) / (2 * h), abs=1e-5)
    assert d2[0] == pytest.approx((vp[0] - 2 * v0[0] + vm[0]) / h**2, abs=1e-3)


def test_GkerMV_isotropic_matches_product():
    x = np.array([[0.5], [0.5]])
    iso = st.GkerMV(x, 3.0)
    val1d, _, _ = st.Gker(np.array([0.5]), 3.0)
    assert iso[0] == pytest.approx(val1d[0] ** 2)


# ----- fast_conv (vs MATLAB) ----------------------------------------------

def test_fast_conv_1d_matches_matlab():
    x = np.zeros(11)
    x[5] = 1
    sm, ss = st.fast_conv(x, 3, 1)
    expected = [
        0.0001, 0.0023, 0.0196, 0.0913, 0.2301, 0.3131,
        0.2301, 0.0913, 0.0196, 0.0023, 0.0001,
    ]
    np.testing.assert_allclose(sm, expected, atol=1e-4)
    assert ss == pytest.approx(0.2214, abs=1e-4)


def test_fast_conv_2d_matches_matlab():
    img = np.zeros((7, 7))
    img[3, 3] = 1
    sm, ss = st.fast_conv(img, 2, 2)
    np.testing.assert_allclose(
        sm[3, :],
        [0.0004, 0.0138, 0.1103, 0.2206, 0.1103, 0.0138, 0.0004],
        atol=1e-4,
    )
    assert ss == pytest.approx(0.1107, abs=1e-4)


def test_fast_conv_zero_fwhm_passthrough():
    x = np.arange(5.0)
    sm, ss = st.fast_conv(x, 0)
    np.testing.assert_array_equal(sm, x)
    assert ss == 1.0


def test_fast_conv_subject_loop():
    data = np.zeros((7, 7, 3))
    data[3, 3, :] = 1
    sm, ss = st.fast_conv(data, 2, 2)
    assert sm.shape == (7, 7, 3)
    # each subject smoothed identically
    np.testing.assert_allclose(sm[:, :, 0], sm[:, :, 1])


# ----- morphology (vs MATLAB) ---------------------------------------------

def test_dilate_mask_matches_matlab():
    m = np.zeros((5, 5))
    m[2, 2] = 1
    assert int(st.dilate_mask(m, 1).sum()) == 9
    back = st.dilate_mask(st.dilate_mask(m, 1), -1)
    assert int(back.sum()) == 1


def test_dilate_mask_zero_is_identity():
    m = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(st.dilate_mask(m, 0), m > 0)


def test_mask_bndry():
    m = np.zeros((7, 7))
    m[2:5, 2:5] = 1
    outer, inner = st.mask_bndry(m)
    # outer boundary is one ring outside, disjoint from mask
    assert np.all(outer[m > 0] == 0)
    assert outer.sum() > 0


def test_doubleim_shape_and_replication():
    orig = np.random.default_rng(0).standard_normal((91, 109, 91))
    out = st.doubleim(orig)
    assert out.shape == (182, 218, 182)
    # the 2x2x2 block all equals the original voxel
    np.testing.assert_allclose(out[0:2, 0:2, 0:2], orig[0, 0, 0])


# ----- mvtstat (vs MATLAB) -------------------------------------------------

def test_mvtstat_matches_matlab():
    data = np.arange(1, 25).reshape((2, 3, 4), order="F").astype(float)
    t, xb, sd, cd = st.mvtstat(data)
    np.testing.assert_allclose(
        t,
        [[2.5820, 3.0984, 3.6148], [2.8402, 3.3566, 3.8730]],
        atol=1e-4,
    )
    np.testing.assert_array_equal(xb, [[10, 12, 14], [11, 13, 15]])


def test_mvtstat_nansaszeros():
    # All-zero data: xbar=0, std=0 -> 0/0 = NaN -> zeroed (matches MATLAB).
    data = np.zeros((2, 2, 5))
    with np.errstate(invalid="ignore"):
        t, _, _, _ = st.mvtstat(data, nansaszeros=True)
    assert np.all(t == 0)
    # Constant non-zero data: xbar=1, std=0 -> 1/0 = Inf (NOT zeroed, matches MATLAB).
    data2 = np.ones((2, 2, 5))
    with np.errstate(divide="ignore"):
        t2, _, _, _ = st.mvtstat(data2, nansaszeros=True)
    assert np.all(np.isinf(t2))


# ----- gen_noise -----------------------------------------------------------

def test_gen_noise_shape_and_mask():
    mask = np.zeros((10, 10))
    mask[2:8, 2:8] = 1
    rng = np.random.default_rng(0)
    noise = st.gen_noise(mask, 3, nsubj=4, rng=rng)
    assert noise.shape == (10, 10, 4)
    # outside the mask is zero
    assert np.all(noise[0, 0, :] == 0)
