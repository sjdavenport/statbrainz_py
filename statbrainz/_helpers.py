"""_helpers (mirrors StatBrainz/_helpers.m)."""

import numpy as np

__all__ = ['make_srf', 'viewthresh_image']


def make_srf(vertices, faces):
    """Build a surface dict from vertices and (0-based) faces."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    return {
        "vertices": vertices,
        "faces": faces,
        "nvertices": vertices.shape[0],
        "nfaces": faces.shape[0],
    }


def viewthresh_image(map_, frontgroundcolor, backgroundcolor=(0, 0, 0)):
    """Build the RGB image used by ``viewthresh`` (no rendering).

    Parameters
    ----------
    map_ : numpy.ndarray
        2D map in ``[0, 1]``.
    frontgroundcolor : sequence of float
        Foreground RGB.
    backgroundcolor : sequence of float, optional
        Background RGB. Default black.

    Returns
    -------
    numpy.ndarray
        ``(r, c, 3)`` RGB image.
    """
    map_ = np.asarray(map_, dtype=float)
    fg = np.asarray(frontgroundcolor, dtype=float)
    bg = np.asarray(backgroundcolor, dtype=float)
    im = np.zeros((*map_.shape, 3))
    for k in range(3):
        im[:, :, k] = map_ * fg[k] + (1 - map_) * bg[k]
    return im
