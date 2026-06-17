"""read_fs_geometry (mirrors StatBrainz/Surface/ReadSurfaceFiles/read_fs_geometry.m)."""

import numpy as np
import nibabel as nib

__all__ = ['read_fs_geometry']


def read_fs_geometry(filepath):
    """Read a FreeSurfer geometry file (via nibabel).

    Returns
    -------
    vertices : numpy.ndarray
        ``(nvertices, 3)``.
    faces : numpy.ndarray
        ``(nfaces, 3)`` 0-based.
    """
    vertices, faces = nib.freesurfer.read_geometry(filepath)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
