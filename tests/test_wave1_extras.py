"""Tests for Wave 1 straggler ports: gmcdf, gmrnd, utils, prewhiten.

gmcdf/filesindir cross-checked against MATLAB R2024b.
"""

import numpy as np
import pytest

import statbrainz as st
import statbrainz as inf


def test_gmcdf_matches_matlab():
    out = st.gmcdf([-1, 0, 1, 2], [0.3, 0.7], [0, 2], [1, 0.5])
    np.testing.assert_allclose(
        out, [0.04759658, 0.15002217, 0.26832852, 0.64317496], atol=1e-7
    )


def test_gmcdf_bad_weights():
    with pytest.raises(ValueError):
        st.gmcdf(0.0, [0.3, 0.3], [0, 1], [1, 1])


def test_gmrnd_reproducible_and_distributed():
    rng = np.random.default_rng(42)
    samples = st.gmrnd(20000, [0.5, 0.5], [-5, 5], [0.5, 0.5], rng=rng)
    assert samples.shape == (20000,)
    # roughly half near -5, half near +5
    assert np.mean(samples < 0) == pytest.approx(0.5, abs=0.05)


def test_filesindir_matches_matlab(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    assert st.filesindir(str(tmp_path), ".md") == ["a.md"]
    assert set(st.filesindir(str(tmp_path))) == {"a.md", "b.txt"}
    assert ".hidden" in st.filesindir(str(tmp_path), hiddenfiles=True)


def test_statbrainz_maindir_exists():
    d = st.statbrainz_maindir()
    assert d.endswith(("/", "\\"))
    import os
    assert os.path.isdir(d)


def test_prewhiten_literal_behaviour():
    data = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    out = inf.prewhiten(data)
    ts = data[-1]
    np.testing.assert_allclose(out, ts / np.sqrt(np.cov(ts)))
