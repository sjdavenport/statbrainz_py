"""loadmask (mirrors StatBrainz/Surface/ReadSurfaceFiles/loadmask.m)."""

import os

from statbrainz.ImageViewing.imgload import data_dir
from .fsannot2mask import fsannot2mask

__all__ = ['loadmask']

_ACCEPTED_IDS = ("fs3", "fs4", "fs5", "fs6", "fs7")


def loadmask(srf, maskname):
    """Load a bilateral region mask from bundled FreeSurfer annotation files.

    The MATLAB original read ``.annot`` files from a hard-coded local path; this
    port reads the ``lh.aparc.annot`` / ``rh.aparc.annot`` files bundled under
    ``data_dir()/Surface/fsaverageN/`` (an intentional, documented divergence,
    matching how :func:`loadsrf` locates its data).

    Parameters
    ----------
    srf : str
        Surface identifier, e.g. ``'fs5'``; the ``'fs'`` prefix is expanded to
        ``'fsaverage'`` to locate the ``.annot`` files.
    maskname : str
        Region name to extract (passed to :func:`fsannot2mask`).

    Returns
    -------
    dict
        ``{'lh': lh_mask, 'rh': rh_mask}`` of boolean per-vertex masks.
    """
    if srf not in _ACCEPTED_IDS:
        raise ValueError("The supplied srf is not available")

    # 'fs5' -> 'fsaverage5'
    full_id = "fsaverage" + srf[2]
    surf_dir = os.path.join(data_dir(), "Surface", full_id)

    annotfilelh = os.path.join(surf_dir, "lh.aparc.annot")
    annotfilerh = os.path.join(surf_dir, "rh.aparc.annot")
    if not os.path.isfile(annotfilelh) or not os.path.isfile(annotfilerh):
        raise FileNotFoundError(
            f"Annotation files not bundled for {full_id}: {surf_dir}")

    mask = {}
    mask["lh"] = fsannot2mask(annotfilelh, maskname)[0]
    mask["rh"] = fsannot2mask(annotfilerh, maskname)[0]
    return mask
