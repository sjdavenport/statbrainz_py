"""viewthresh (mirrors StatBrainz/ImageViewing/viewthresh.m)."""

import numpy as np

from statbrainz._helpers import viewthresh_image
from ._display_helpers import _get_ax

__all__ = ['viewthresh']


def viewthresh(map_, frontgroundcolor, backgroundcolor=(0, 0, 0), ax=None):
    """Display a thresholded map as a 2-colour image.

    Parameters
    ----------
    map_ : numpy.ndarray
        2D map in ``[0, 1]``.
    frontgroundcolor, backgroundcolor : sequence of float
        Foreground / background RGB.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new one is created if omitted.

    Returns
    -------
    numpy.ndarray
        The RGB image that was displayed.
    """
    im = viewthresh_image(map_, frontgroundcolor, backgroundcolor)
    ax = _get_ax(ax)
    ax.imshow(im)
    ax.axis("off")
    return im
