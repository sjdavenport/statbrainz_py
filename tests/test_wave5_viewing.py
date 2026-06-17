"""Tests for Wave 5 viewing.

Helper (array-building) functions are cross-checked against MATLAB R2024b.
Rendering functions are smoke-tested with the non-interactive Agg backend.
"""

import numpy as np
import pytest

from statbrainz import viewing as vw
from statbrainz.brain import imgload


# ----- pure helpers (vs MATLAB) -------------------------------------------

def test_colorRegion_named():
    m = np.array([[1, 0], [0, 1]], float)
    cr = vw.colorRegion(m, "yellow")
    np.testing.assert_array_equal(cr[:, :, 0], m)  # red channel
    np.testing.assert_array_equal(cr[:, :, 1], m)  # green channel
    np.testing.assert_array_equal(cr[:, :, 2], np.zeros((2, 2)))  # blue off


def test_colorRegion_triple():
    m = np.array([[1, 0], [0, 1]], float)
    cr = vw.colorRegion(m, [0.2, 0.4, 0.6])
    assert cr[0, 0, 1] == pytest.approx(0.4)
    assert cr[0, 0, 2] == pytest.approx(0.6)


def test_custom_colormap_matches_matlab():
    cmap = vw.custom_colormap([0, 0, 0], [1, 1, 1], 5)
    np.testing.assert_allclose(
        cmap,
        np.column_stack([np.linspace(0, 1, 5)] * 3),
    )


def test_peak2circle_matches_matlab():
    pc = vw.peak2circle([20, 25, 30])
    assert pc.shape == (182, 218, 182)
    assert int(pc.sum()) == 98  # 5^3 - 3^3 = 125 - 27 = 98


def test_viewthresh_image():
    map_ = np.array([[0.0, 1.0]])
    im = vw.viewthresh_image(map_, [1, 0, 0], [0, 0, 1])
    # foreground pixel red, background pixel blue
    np.testing.assert_array_equal(im[0, 1], [1, 0, 0])
    np.testing.assert_array_equal(im[0, 0], [0, 0, 1])


def test_combine_brains_matches_matlab():
    mask = imgload("MNImask") > 0
    ci = vw.combine_brains(mask.astype(float), [30, 40, 50], mask, 5)
    assert ci.shape == (90, 244)
    assert int(ci.sum()) == 12553


def test_combine_brains_no_bounds():
    vol = np.zeros((91, 109, 91))
    vol[30, :, :] = 1
    ci = vw.combine_brains(vol, [31, 41, 51], vol, 5, use_bounds=False)
    assert ci.shape == (109, 291)


# ----- rendering smoke tests (Agg) ----------------------------------------

@pytest.fixture(autouse=True)
def _agg_backend():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")


def test_viewthresh_renders():
    import matplotlib.pyplot as plt

    im = vw.viewthresh(np.array([[0.0, 1.0], [1.0, 0.0]]), [1, 0, 0])
    assert im.shape == (2, 2, 3)
    plt.close("all")


def test_viewdata_renders():
    import matplotlib.pyplot as plt

    data = np.random.default_rng(0).standard_normal((10, 10))
    mask = np.ones((10, 10))
    region = np.zeros((10, 10))
    region[2:5, 2:5] = 1
    ax = vw.viewdata(data, mask, [region], ["red"], rotate=1)
    assert ax is not None
    plt.close("all")


def test_viewbrain_renders():
    import matplotlib.pyplot as plt

    mask = imgload("MNImask") > 0
    brain = mask.astype(float)
    ax = vw.viewbrain(brain, slice_=(30, 40, 50), brain_mask=mask)
    assert ax is not None
    plt.close("all")
