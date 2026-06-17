"""viewbrain (mirrors StatBrainz/ImageViewing/viewbrain.m)."""

import numpy as np

from .combine_brains import combine_brains
from .viewdata import viewdata
from .imgload import imgload

__all__ = ['viewbrain']


def viewbrain(brain_im, slice_=(30, 40, 50), brain_mask=None, padding=15, ax=None):
    """Display a 3-view montage of a brain volume.

    Parameters
    ----------
    brain_im : numpy.ndarray
        Brain volume.
    slice_ : sequence of int, optional
        1-based slice per axis. Default ``(30, 40, 50)``.
    brain_mask : numpy.ndarray, optional
        Brain mask. Defaults to the bundled MNI mask.
    padding : int, optional
        Gap between views. Default ``15``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if brain_mask is None:
        from ..brain.io import imgload

        brain_mask = imgload("MNImask") > 0
    brain_im4D = combine_brains(brain_im, slice_, brain_mask, padding)
    brain_mask4D = combine_brains(brain_mask, slice_, brain_mask, padding)
    return viewdata(brain_im4D, brain_mask4D, None, None, 1, None, ax=ax)
