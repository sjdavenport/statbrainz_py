"""Tests for Wave 2 brain functions (data-dependent: MNI mask).

Expectations cross-checked against MATLAB R2024b.
"""

import numpy as np
import pytest

import statbrainz as brain
from statbrainz import convind


def test_imgload_mnimask():
    m = brain.imgload("MNImask")
    assert m.shape == (91, 109, 91)
    assert m.dtype == bool
    assert int(m.sum()) == 228483


def test_imgload_missing():
    with pytest.raises(FileNotFoundError):
        brain.imgload("does_not_exist")


def test_voxinMNI_matches_matlab():
    m = brain.imgload("MNImask")
    firstvox = int(np.nonzero(m.ravel(order="F"))[0][0] + 1)  # 1-based
    assert firstvox == 3502
    assert brain.voxinMNI(firstvox) is True
    assert brain.voxinMNI(convind(firstvox)) is True
    assert brain.voxinMNI([1, 1, 1]) is False


def test_getMNImask_matches_matlab():
    mm, bounds = brain.getMNImask()
    assert mm.shape == (72, 90, 77)
    assert int(mm.sum()) == 228483
    assert len(bounds) == 3


def test_gen_mask_intersection():
    # 3 subjects, full-MNI-length vectors; intersect with MNI mask
    m = brain.imgload("MNImask").astype(float).ravel(order="F")
    nvox = m.size
    data = np.ones((3, nvox))
    out = brain.gen_mask(data, use_MNI=True, make3D=True)
    assert out.shape == (91, 109, 91)
    # intersecting all-ones with MNI gives back the MNI mask
    assert int(out.sum()) == 228483


def test_gen_mask_drops_voxels():
    m = brain.imgload("MNImask").astype(float).ravel(order="F")
    nvox = m.size
    data = np.ones((2, nvox))
    data[0, :] = 0  # one subject masks everything out
    out = brain.gen_mask(data, use_MNI=True, make3D=False)
    assert out.sum() == 0


def test_mvtstat_dep_matches_matlab():
    data = np.arange(1, 25).reshape((4, 6), order="F").astype(float)
    t, xb, sd, cd = brain.mvtstat_dep(data)
    np.testing.assert_allclose(
        t, [3.8730, 10.0698, 16.2665, 22.4633, 28.6601, 34.8569], atol=1e-4
    )
    np.testing.assert_array_equal(xb, [2.5, 6.5, 10.5, 14.5, 18.5, 22.5])


def test_imgsave_roundtrip(tmp_path):
    arr = np.random.default_rng(0).standard_normal((5, 6, 7))
    path = brain.imgsave(arr, "test_out", str(tmp_path))
    loaded = brain.imgload(path)
    np.testing.assert_allclose(loaded, arr, atol=1e-5)
