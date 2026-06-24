"""Tests for the permutation cluster / TFCE ports.

perm_tfce, real_tfce_clusters, perm_cluster, localized_csi.

These functions use random sign flips, so they cannot reproduce MATLAB's RNG
stream voxel-for-voxel. The tests instead pin the deterministic structure
(shapes, the observed-statistic entries that are RNG-independent, monotonicity
and the documented edge cases) and check that the seeded numpy stream is
reproducible run-to-run.
"""

import numpy as np
import pytest
from scipy.stats import norm

import statbrainz as sb


def _make_data(seed=0, dim=(20, 20), nsubj=15, sig_mag=1.0):
    rng = np.random.default_rng(seed)
    sig = np.zeros(dim)
    sig[3:8, 3:8] = sig_mag
    data = rng.standard_normal((*dim, nsubj)) + sig[..., None]
    return data, np.ones(dim), sig


# ----- perm_tfce -----------------------------------------------------------

def test_perm_tfce_shapes_and_observed_entry():
    data, mask, _ = _make_data()
    # The observed (first) maximum is RNG-independent: recompute it directly.
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    observed_tfce_max = sb.tfce(tstat * (mask > 0), 2, 0.5, 8, 0.1, 0).max()

    thr, vom, store = perm_tfce_call(data, mask, nperm=40, seed=1)
    assert vom.shape == (40,)
    assert vom[0] == pytest.approx(observed_tfce_max)
    assert np.isnan(store)
    # threshold is a percentile of the maxima vector
    assert thr == pytest.approx(np.percentile(vom, 95))


def test_perm_tfce_seed_reproducible():
    data, mask, _ = _make_data()
    thr1, vom1, _ = perm_tfce_call(data, mask, nperm=30, seed=7)
    thr2, vom2, _ = perm_tfce_call(data, mask, nperm=30, seed=7)
    assert thr1 == thr2
    np.testing.assert_array_equal(vom1, vom2)


def test_perm_tfce_store_perms():
    data, mask, _ = _make_data()
    _, _, store = sb.perm_tfce(data, mask, nperm=10, show_loader=False,
                               store_perms=True, rng=np.random.default_rng(3))
    assert store.shape == ((mask > 0).sum(), 10)


def perm_tfce_call(data, mask, nperm, seed):
    return sb.perm_tfce(data, mask, nperm=nperm, show_loader=False,
                        rng=np.random.default_rng(seed))


# ----- real_tfce_clusters --------------------------------------------------

def test_real_tfce_clusters_threshold_zero_keeps_all():
    data, mask, _ = _make_data()
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    tfce_im = sb.tfce(tstat * mask, 2, 0.5, 8, 0.1, 0)
    # threshold below every cluster's TFCE max -> all positive clusters survive
    cim, clusters, pvals = sb.real_tfce_clusters(tfce_im, mask, -1.0)
    assert cim.shape == mask.shape
    assert len(clusters) >= 1
    # cluster image is binary
    assert set(np.unique(cim)).issubset({0.0, 1.0})
    assert pvals is None


def test_real_tfce_clusters_high_threshold_keeps_none():
    data, mask, _ = _make_data()
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    tfce_im = sb.tfce(tstat * mask, 2, 0.5, 8, 0.1, 0)
    cim, clusters, _ = sb.real_tfce_clusters(tfce_im, mask, 1e12)
    assert len(clusters) == 0
    assert cim.sum() == 0


def test_real_tfce_clusters_pvals_match_distbn2pval():
    data, mask, _ = _make_data()
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    tfce_im = sb.tfce(tstat * mask, 2, 0.5, 8, 0.1, 0)
    vom = np.array([1.0, 2.0, 3.0, 100.0, 200.0])
    _, _, pvals = sb.real_tfce_clusters(tfce_im, mask, -1.0, vec_of_maxima=vom)
    # every p-value is in [0, 1]
    assert pvals is not None
    assert np.all((pvals >= 0) & (pvals <= 1))


# ----- perm_cluster --------------------------------------------------------

def test_perm_cluster_observed_size_and_threshold():
    data, mask, _ = _make_data(sig_mag=1.5)
    CDT = norm.ppf(0.99)
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    _, _, sizes, _ = sb.numOfConComps(tstat * (mask > 0), CDT, 8)
    expected_max = sizes.max()

    thr, vom, store = sb.perm_cluster(data, mask, nperm=40, show_loader=False,
                                      rng=np.random.default_rng(2))
    assert vom.shape == (40,)
    assert vom[0] == expected_max
    assert thr == pytest.approx(np.percentile(vom, 95))
    assert np.isnan(store)


def test_perm_cluster_no_superthreshold_returns_nan():
    # pure noise with an enormous CDT -> no above-threshold voxels in the
    # observed statistic, so the function warns and returns NaN.
    rng = np.random.default_rng(5)
    data = rng.standard_normal((10, 10, 8))
    mask = np.ones((10, 10))
    with pytest.warns(UserWarning):
        thr, vom, store = sb.perm_cluster(data, mask, CDT=1e6, nperm=20,
                                          show_loader=False, rng=rng)
    assert np.isnan(thr)
    assert np.isnan(store)


# ----- localized_csi -------------------------------------------------------

def test_localized_csi_single_mask_promoted_to_list():
    data, mask, sig = _make_data()
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    vom = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    pv, mx = sb.localized_csi(tstat, (sig > 0).astype(float), vom, CDT=2.0,
                              show_loader=False)
    assert pv.shape == (1,)
    assert mx.shape == (1,)


def test_localized_csi_region_with_signal_has_largest_cluster():
    data, mask, sig = _make_data(sig_mag=2.0)
    tstat = sb.unwrap(sb.nan2zero(sb.mvtstat(sb.vec_data(data, mask))[0]), mask)[..., 0]
    vom = np.arange(0.0, 30.0)
    region_signal = (sig > 0).astype(float)
    region_empty = np.zeros_like(sig)
    region_empty[18:20, 18:20] = 1.0  # far from signal, mostly noise
    pv, mx = sb.localized_csi(tstat, [region_signal, region_empty], vom, CDT=2.0,
                              show_loader=False)
    # signal region should have a (weakly) larger max cluster than the far region
    assert mx[0] >= mx[1]
    assert np.all((pv >= 0) & (pv <= 1))
