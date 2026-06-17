"""Test for Wave 4 spintest (depends on spin_surface)."""

import numpy as np

from statbrainz import inference as inf, surface as sf


def _octa_sphere():
    V = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    F = (
        np.array([[1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
                  [3, 1, 6], [2, 3, 6], [4, 2, 6], [1, 4, 6]]) - 1
    )
    srf = sf.make_srf(V, F)
    return {"lh": srf, "rh": srf}


def test_spintest_runs():
    sphere = _octa_sphere()
    X = {"lh": np.arange(6.0), "rh": np.arange(6.0)}
    Y = {"lh": np.arange(6.0)[::-1].copy(), "rh": np.arange(6.0)}
    thr, rho = inf.spintest(
        X, Y, sphere, nperm=20, show_loader=False, rng=np.random.default_rng(0)
    )
    assert rho.shape == (20,)
    assert np.isfinite(thr)
    assert np.all(np.abs(rho[np.isfinite(rho)]) <= 1.0 + 1e-9)
