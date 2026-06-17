"""Tests for the Atlases ports (Harvard-Oxford, bundled 2mm).

Expectations cross-checked against MATLAB R2024b.
"""

import numpy as np
import pytest

import statbrainz as at


def test_getBrainRegionNames_matches_matlab():
    import os
    from statbrainz.Atlases._shared import _atlas_dir
    loc = os.path.join(_atlas_dir(), "HarvardOxford")
    names = at.getBrainRegionNames(os.path.join(loc, "HarvardOxford-Cortical.xml"))
    assert len(names) == 48
    assert names[0] == "Frontal Pole"
    assert names[47] == "Occipital Pole"


def test_get_mask_frontal_pole_matches_matlab():
    mask, indices, areas = at.get_mask("HOc", "Frontal Pole")
    assert int(mask.sum()) == 15397
    assert indices == [1]
    assert areas == ["Frontal Pole"]


def test_get_mask_all():
    mask, indices, areas = at.get_mask("HOc", "all")
    assert len(indices) == 48
    assert len(areas) == 48
    assert mask.sum() > 0


def test_get_mask_separate_labels():
    # 'gyrus' matches multiple regions; with together=False they get distinct vals
    mask_t, idx_t, _ = at.get_mask("HOc", "gyrus", together=True)
    mask_s, idx_s, _ = at.get_mask("HOc", "gyrus", together=False)
    assert len(idx_t) > 1
    assert set(np.unique(mask_t)) <= {0, 1}
    assert mask_s.max() == len(idx_t)  # last region labelled by its position


def test_get_mask_unknown_atlas():
    with pytest.raises(ValueError):
        at.get_mask("NopeAtlas", "all")


def test_getregion_matches_matlab():
    # MATLAB found a Frontal Pole voxel at (39, 82, 23) [1-based]
    hoc, hosc = at.getregion([39, 82, 23], "HOc")
    assert hoc[0] == "Frontal Pole"


def test_getregion_multiple_points():
    hoc, hosc = at.getregion([[39, 82, 23], [1, 1, 1]], "HOc")
    assert hoc[0] == "Frontal Pole"
    assert hoc[1] is None  # corner voxel: no region


def test_atlas_masks_matches_matlab():
    region_masks, region_names = at.atlas_masks("HOc")
    assert len(region_masks) == 48
    assert len(region_names) == 48
    assert int(region_masks[0].sum()) == 15397  # Frontal Pole


def test_atlas_masks_boundary():
    region_masks, _ = at.atlas_masks("HOc", get_boundary=True)
    # a boundary has fewer voxels than the filled region
    filled, _ = at.atlas_masks("HOc", get_boundary=False)
    assert region_masks[0].sum() < filled[0].sum()
