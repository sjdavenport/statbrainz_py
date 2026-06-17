"""load_gifti (mirrors StatBrainz/Surface/ReadSurfaceFiles/gifti/load_gifti.m)."""

import numpy as np
import nibabel as nib

__all__ = ['load_gifti']


def load_gifti(filepath):
    """Load a GIfTI surface, returning ``(vertices, faces)`` (faces 0-based)."""
    g = nib.load(filepath)
    arrays = g.darrays
    vertices = None
    faces = None
    for da in arrays:
        code = da.intent
        name = nib.nifti1.intent_codes.label.get(code, "")
        if name == "pointset" or da.data.ndim == 2 and da.data.shape[1] == 3 and da.data.dtype.kind == "f":
            vertices = np.asarray(da.data, dtype=float)
        elif name == "triangle" or (da.data.dtype.kind in "iu"):
            faces = np.asarray(da.data, dtype=np.int64)
    if vertices is None or faces is None:
        # fall back to the first two arrays
        vertices = np.asarray(arrays[0].data, dtype=float)
        faces = np.asarray(arrays[1].data, dtype=np.int64)
    return vertices, faces
