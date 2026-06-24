"""Tests for loadmask (bundled fsaverage .annot region masks)."""

import numpy as np
import pytest

import statbrainz as sb

# fsaverage vertex counts per hemisphere (lower densities are subsampled).
_FS_VERTS = {"fs3": 642, "fs4": 2562, "fs5": 10242, "fs6": 40962, "fs7": 163842}


def test_loadmask_fs5_returns_lh_rh():
    mask = sb.loadmask("fs5", "superiorfrontal")
    assert set(mask) == {"lh", "rh"}
    assert mask["lh"].size == _FS_VERTS["fs5"]
    assert mask["rh"].size == _FS_VERTS["fs5"]
    assert mask["lh"].dtype == bool
    # superiorfrontal is a sizeable region but not the whole hemisphere
    assert 0 < mask["lh"].sum() < mask["lh"].size
    assert 0 < mask["rh"].sum() < mask["rh"].size


@pytest.mark.parametrize("srf", list(_FS_VERTS))
def test_loadmask_vertex_counts(srf):
    mask = sb.loadmask(srf, "unknown")
    assert mask["lh"].size == _FS_VERTS[srf]
    assert mask["rh"].size == _FS_VERTS[srf]


def test_loadmask_bad_id_raises():
    with pytest.raises(ValueError):
        sb.loadmask("fs2", "unknown")


def test_loadmask_bad_region_raises():
    with pytest.raises(ValueError):
        sb.loadmask("fs5", "not_a_real_region")
