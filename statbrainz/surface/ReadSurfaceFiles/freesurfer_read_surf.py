"""freesurfer_read_surf (mirrors StatBrainz/Surface/ReadSurfaceFiles/freesurfer_read_surf.m)."""

from .read_fs_geometry import read_fs_geometry

__all__ = ['freesurfer_read_surf']


def freesurfer_read_surf(fname):
    """Alias of :func:`read_fs_geometry` (matches the MATLAB function name)."""
    return read_fs_geometry(fname)
