"""colorRegion (mirrors StatBrainz/ImageViewing/colorRegion.m)."""

import numpy as np

__all__ = ['colorRegion']


_NAMED_COLORS = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "yellow": (1, 1, 0),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "gray": (0.5, 0.5, 0.5),
    "grey": (0.5, 0.5, 0.5),
    "white": (1, 1, 1),
    "black": (0, 0, 0),
}


def colorRegion(region_mask, color):
    """Build an RGB image of ``region_mask`` painted in ``color``.

    Parameters
    ----------
    region_mask : numpy.ndarray
        2D mask (values scale the colour).
    color : str, sequence of float, or nan
        A named colour, an ``(r, g, b)`` triple, or ``nan`` (red).

    Returns
    -------
    numpy.ndarray
        ``(r, c, 3)`` RGB image.
    """
    region_mask = np.asarray(region_mask, dtype=float)
    r, c = region_mask.shape
    out = np.zeros((r, c, 3))
    if isinstance(color, str):
        rgb = _NAMED_COLORS.get(color)
        if rgb is None:
            out[:, :, 0] = region_mask  # otherwise -> red channel
            return out
        for k in range(3):
            out[:, :, k] = region_mask * rgb[k]
    elif np.isscalar(color) and np.isnan(color):
        out[:, :, 0] = region_mask
    else:
        color = np.asarray(color, dtype=float)
        for k in range(3):
            out[:, :, k] = color[k] * region_mask
    return out
