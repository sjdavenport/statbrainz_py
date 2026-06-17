"""peak2circle (mirrors StatBrainz/ImageViewing/peak2circle.m)."""

import numpy as np

__all__ = ['peak2circle']


def peak2circle(point_coordinates):
    """Hollow 5x5x5 cube shell around a peak in a ``182x218x182`` image.

    Parameters
    ----------
    point_coordinates : sequence of int
        1-based ``(x, y, z)`` peak coordinate (in the 91x109x91 grid; it is
        doubled internally, matching MATLAB).

    Returns
    -------
    numpy.ndarray
        Boolean ``182x218x182`` shell mask.
    """
    p = 2 * np.asarray(point_coordinates, dtype=int)
    circlemask = np.zeros((182, 218, 182))
    px, py, pz = p[0] - 1, p[1] - 1, p[2] - 1  # 1-based -> 0-based
    circlemask[px - 2:px + 3, py - 2:py + 3, pz - 2:pz + 3] = 1
    circlemask[px - 1:px + 2, py - 1:py + 2, pz - 1:pz + 2] = 0
    return circlemask.astype(bool)
