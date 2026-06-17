"""Tests for Wave 4 surface ports.

Built on a deterministic octahedron mesh; expectations cross-checked against
MATLAB R2024b. MATLAB faces are 1-based, so they are converted to 0-based here.
"""

import numpy as np
import pytest

import statbrainz as sf


@pytest.fixture
def octahedron():
    V = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    F = (
        np.array(
            [[1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
             [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]]
        )
        - 1
    )
    return sf.make_srf(V, F)


def test_surfstatedg(octahedron):
    edg = sf.SurfStatEdg(octahedron)
    assert edg.shape == (12, 2)  # octahedron has 12 edges


def test_srf_face_area_matches_matlab(octahedron):
    fa, va = sf.srf_face_area(octahedron)
    assert fa.sum() == pytest.approx(6.9282, abs=1e-4)
    np.testing.assert_allclose(va, np.full(6, 1.1547), atol=1e-4)


def test_surfstatsmooth_matches_matlab(octahedron):
    data = np.zeros(6)
    data[0] = 1
    sm = sf.SurfStatSmooth(octahedron, data, 0, "ones", 3)
    np.testing.assert_allclose(
        sm, [0.2344, 0.1094, 0.1641, 0.1641, 0.1641, 0.1641], atol=1e-4
    )


def test_adjacency_matrix_matches_matlab(octahedron):
    A = sf.adjacency_matrix(octahedron, "ones")
    assert int(A.sum()) == 24  # 12 edges x 2 (symmetric)
    # every vertex of an octahedron has degree 4
    assert np.all(np.asarray(A.sum(axis=1)).ravel() == 4)


def test_srf_fwhm2niters_matches_matlab(octahedron):
    assert sf.srf_fwhm2niters(10, octahedron) == 81


def test_graph_cc(octahedron):
    A = sf.adjacency_matrix(octahedron, "ones")
    gdata = np.array([1, 1, 0, 0, 1, 0])
    ncl, occ, csz, idx = sf.graph_cc(gdata, 0.5, A)
    assert ncl == 1  # vertices 1,2,5 are connected on the octahedron


def test_srf_dilate_mask_matches_matlab(octahedron):
    mask = np.zeros(6)
    mask[4] = 1  # apex vertex 5
    dm = sf.srf_dilate_mask(octahedron, mask, 1)
    np.testing.assert_array_equal(np.nonzero(dm)[0] + 1, [1, 2, 3, 4, 5])
    em = sf.srf_dilate_mask(octahedron, dm, -1)
    np.testing.assert_array_equal(np.nonzero(em)[0] + 1, [5])


def test_srf_contour_matches_matlab(octahedron):
    mask = np.zeros(6)
    mask[4] = 1
    inner, outer = sf.srf_contour(octahedron, mask)
    assert inner.sum() == 1
    np.testing.assert_array_equal(np.nonzero(outer)[0] + 1, [1, 2, 3, 4])


def test_resample_srf_nn_identity(octahedron):
    # resampling a surface onto itself is the identity mapping
    nn = sf.resample_srf_nn(octahedron, octahedron)
    np.testing.assert_array_equal(nn, np.arange(6))
    data = np.array([10.0, 20, 30, 40, 50, 60])
    out = sf.resample_srf(data, octahedron, octahedron)
    np.testing.assert_array_equal(out, data)


def test_srf_noise_shape(octahedron):
    rng = np.random.default_rng(0)
    noise = sf.srf_noise(octahedron, FWHM=0, nsubj=3, rng=rng)
    assert noise.shape == (6, 3)


def test_spin_surface_shape():
    rng = np.random.default_rng(0)
    V = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    F = (
        np.array([[1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
                  [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]]) - 1
    )
    srf = sf.make_srf(V, F)
    sphere = {"lh": srf, "rh": srf}
    data = {"lh": np.arange(6.0), "rh": np.arange(6.0)}
    lr, rr = sf.spin_surface(data, sphere, nperm=5, show_loader=False, rng=rng)
    assert lr.shape == (6, 5) and rr.shape == (6, 5)
    # first column is the original (include_orig default True)
    np.testing.assert_array_equal(lr[:, 0], np.arange(6.0))
