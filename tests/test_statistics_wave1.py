"""Tests for Wave 1 statistics ports (leaf functions)."""

import numpy as np
import pytest
from scipy import stats

import statbrainz as sb


def test_apower_default_sqrt():
    assert sb.apower(4.0) == pytest.approx(2.0)
    # sign preserving
    assert sb.apower(-4.0) == pytest.approx(-2.0)
    x = np.array([-9.0, 0.0, 16.0])
    np.testing.assert_allclose(sb.apower(x), [-3.0, 0.0, 4.0])


def test_apower_custom_power():
    np.testing.assert_allclose(sb.apower(np.array([-2.0, 3.0]), 3), [-8.0, 27.0])


def test_asinh_trans():
    data = np.array([0.0, 1.0, -1.0])
    np.testing.assert_allclose(sb.asinh_trans(data, 2.0), np.arcsinh(data * 2.0) / 2.0)


def test_asinh_data_trans_standardizes():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(1000)
    out = sb.asinh_data_trans(data, 1.0)
    expected = np.arcsinh(data / np.std(data, ddof=1)) / 1.0
    np.testing.assert_allclose(out, expected)


def test_fwhm_sigma_roundtrip():
    assert sb.sigma2fwhm(sb.fwhm2sigma(5.0)) == pytest.approx(5.0)
    assert sb.fwhm2sigma(np.sqrt(8 * np.log(2))) == pytest.approx(1.0)


def test_lcdf_at_location_is_half():
    assert sb.lcdf(2.0, 2.0, 1.0) == pytest.approx(0.5)
    # symmetric points
    lo = sb.lcdf(1.0, 2.0, 1.0)
    hi = sb.lcdf(3.0, 2.0, 1.0)
    assert lo + hi == pytest.approx(1.0)


def test_lcdf_bad_scale():
    with pytest.raises(ValueError):
        sb.lcdf(0.0, 0.0, -1.0)


def test_bernstd():
    interval, se = sb.bernstd(0.05, 1000, 0.95)
    z = stats.norm.ppf(0.975)
    expected_se = np.sqrt(0.05 * 0.95) * z / np.sqrt(1000)
    assert se[0] == pytest.approx(expected_se)
    assert interval[0, 0] == pytest.approx(0.05 - expected_se)
    assert interval[1, 0] == pytest.approx(0.05 + expected_se)


def test_distbn2pval():
    distbn = np.arange(1, 101)  # 1..100
    # values >= 91 : 10 of them (91..100) -> 0.10
    pv = sb.distbn2pval(distbn, [90.5])
    assert pv[0] == pytest.approx(0.10)


def test_tstat_pval_roundtrip_twosided():
    pv = sb.tstat2pval(2.0, 30, do2sample=True)
    expected = 2 * (1 - stats.t.cdf(2.0, 30))
    assert pv == pytest.approx(expected)


def test_pval2tstat_inverts_onesided():
    df = 25
    t = 1.7
    pv = sb.tstat2pval(t, df, do2sample=False)
    np.testing.assert_allclose(sb.pval2tstat(pv, df), t, atol=1e-9)


def test_nan2zero():
    out = sb.nan2zero(np.array([1.0, np.nan, 3.0]))
    np.testing.assert_array_equal(out, [1.0, 0.0, 3.0])
    out = sb.nan2zero(np.array([np.nan]), val=7.0)
    assert out[0] == 7.0


def test_zero2nan():
    out = sb.zero2nan(np.array([1.0, 0.0, 2.0]))
    assert np.isnan(out[1])
    assert out[0] == 1.0


def test_convind_roundtrip_3d():
    # MATLAB docstring example: convind([32,15,37]) <-> convind(358390)
    linear = sb.convind([32, 15, 37])  # coords -> linear
    assert linear == 358390
    coords = sb.convind(358390)  # linear -> coords
    np.testing.assert_array_equal(coords, [32, 15, 37])


def test_convind_2d_roundtrip():
    linear = sb.convind([13, 15], [250, 250])
    coords = sb.convind(linear, [250, 250])
    np.testing.assert_array_equal(coords, [13, 15])


def test_convind_fsleyes_offset():
    coords1 = sb.convind(358390, conv2what=1)
    coords0 = sb.convind(358390, conv2what=0)
    np.testing.assert_array_equal(coords1, np.array(coords0) - 1)


def test_convindall():
    out = sb.convindall([358390], [91, 109, 91])
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out[0], [32, 15, 37])
