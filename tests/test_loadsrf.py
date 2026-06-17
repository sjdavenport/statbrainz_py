"""Tests for loadsrf (bundled template surfaces).

Geometry cross-checked against MATLAB R2024b (fsaverage5 white).
"""

import numpy as np
import pytest

from statbrainz import surface as sf


def test_loadsrf_fs5_white_matches_matlab():
    srf = sf.loadsrf("fs5", "white")
    lh = srf["lh"]
    assert lh["nvertices"] == 10242
    assert lh["nfaces"] == 20480
    np.testing.assert_allclose(
        lh["vertices"][0], [-36.785484, -18.600445, 64.821304], atol=1e-5
    )
    # faces are 0-based here; MATLAB's 1-based first face is [1, 2565, 2563]
    np.testing.assert_array_equal(lh["faces"][0] + 1, [1, 2565, 2563])
    fa, _ = sf.srf_face_area(lh)
    assert fa.sum() == pytest.approx(66661.7988, abs=1e-3)


def test_loadsrf_defaults_to_fs5_white():
    a = sf.loadsrf()
    b = sf.loadsrf("fs5", "white")
    np.testing.assert_array_equal(a["lh"]["vertices"], b["lh"]["vertices"])


def test_loadsrf_all_fsaverage_densities():
    expected = {"fs3": 642, "fs4": 2562, "fs5": 10242, "fs6": 40962, "fs7": 163842}
    for fsid, nvert in expected.items():
        srf = sf.loadsrf(fsid, "sphere")
        assert srf["lh"]["nvertices"] == nvert
        assert srf["rh"]["nvertices"] == nvert
        assert "hemi" in srf["lh"]


def test_loadsrf_hcp():
    hcp = sf.loadsrf("hcp")
    assert hcp["lh"]["nvertices"] == 32492
    assert hcp["rh"]["nvertices"] == 32492


def test_loadsrf_surface_types():
    for stype in ("white", "pial", "sphere", "inflated"):
        srf = sf.loadsrf("fs5", stype)
        assert srf["lh"]["nvertices"] == 10242


def test_loadsrf_invalid_id():
    with pytest.raises(ValueError):
        sf.loadsrf("fs99", "white")


def test_loadsrf_invalid_type():
    with pytest.raises(ValueError):
        sf.loadsrf("fs5", "nope")


def test_loadsrf_bert_not_supported():
    with pytest.raises(NotImplementedError):
        sf.loadsrf("bert", "white")


def test_loadsrf_feeds_smoothing():
    # a loaded surface should work directly with the surface ops
    srf = sf.loadsrf("fs3", "white")  # small mesh
    data = np.zeros(srf["lh"]["nvertices"])
    data[0] = 1.0
    smoothed = sf.SurfStatSmooth(srf["lh"], data, 0, "ones", 2)
    assert smoothed.shape == data.shape
    # smoothing spreads the spike to neighbours (the peak vertex drops below 1)
    assert 0 < smoothed[0] < 1
    assert np.count_nonzero(smoothed) > 1
