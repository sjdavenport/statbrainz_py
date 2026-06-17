"""fs2surf (mirrors StatBrainz/Surface/ReadSurfaceFiles/fs2surf.m)."""

import numpy as np

from statbrainz._helpers import make_srf
from .read_fs_geometry import read_fs_geometry

__all__ = ['fs2surf']


def fs2surf(path4fs, path4fsright=None):
    """Build a surface dict from FreeSurfer geometry file(s).

    Parameters
    ----------
    path4fs : str
        Left (or single) hemisphere geometry file.
    path4fsright : str, optional
        Right hemisphere file; if given, returns a bilateral dict.

    Returns
    -------
    dict
        A surface dict, or ``{'lh':..., 'rh':...}`` with ``hemi`` tags.
    """
    if path4fsright is not None:
        lh = fs2surf(path4fs)
        rh = fs2surf(path4fsright)
        lh["hemi"] = "lh"
        rh["hemi"] = "rh"
        return {"lh": lh, "rh": rh}
    vertices, faces = read_fs_geometry(path4fs)
    return make_srf(vertices, faces)
