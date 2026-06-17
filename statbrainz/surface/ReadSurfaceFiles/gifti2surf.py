"""gifti2surf (mirrors StatBrainz/Surface/ReadSurfaceFiles/gifti2surf.m)."""

import numpy as np

from statbrainz._helpers import make_srf
from .gifti.load_gifti import load_gifti

__all__ = ['gifti2surf']


def gifti2surf(path4gifti, path4giftiright=None):
    """Build a surface dict from GIfTI file(s)."""
    if path4giftiright is not None:
        lh = gifti2surf(path4gifti)
        rh = gifti2surf(path4giftiright)
        lh["hemi"] = "lh"
        rh["hemi"] = "rh"
        return {"lh": lh, "rh": rh}
    vertices, faces = load_gifti(path4gifti)
    return make_srf(vertices, faces)
