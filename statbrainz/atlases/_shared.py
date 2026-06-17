"""_shared (mirrors StatBrainz/Atlases/_shared.m)."""

import os

from statbrainz.ImageViewing.imgload import data_dir

__all__ = ['_atlas_dir', '_ATLASES', '_resolve']


def _atlas_dir():
    return os.path.join(data_dir(), "Atlases")


_ATLASES = {
    "HOc": (
        "HarvardOxford",
        "HarvardOxford-Cortical.xml",
        "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
    ),
    "HOsc": (
        "HarvardOxford",
        "HarvardOxford-Subcortical.xml",
        "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
    ),
}


def _resolve(atlas_name):
    if atlas_name not in _ATLASES:
        raise ValueError(
            f"Atlas {atlas_name!r} not bundled; available: {sorted(_ATLASES)}"
        )
    subdir, xml, nii = _ATLASES[atlas_name]
    loc = os.path.join(_atlas_dir(), subdir)
    return loc, os.path.join(loc, xml), os.path.join(loc, nii)
