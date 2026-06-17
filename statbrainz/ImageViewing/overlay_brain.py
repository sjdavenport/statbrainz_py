"""overlay_brain (mirrors StatBrainz/ImageViewing/overlay_brain.m)."""

import numpy as np

from .combine_brains import combine_brains
from .viewdata import viewdata
from .imgload import imgload

__all__ = ['overlay_brain']


def overlay_brain(
    slice_,
    region_masks=None,
    colors2use=None,
    alpha_val=1,
    underim=None,
    applybrainmask=True,
    outerpadding=([9, 9], [9, 10]),
    ax=None,
):
    """Overlay region masks (and/or an under-image) on an MNI montage.

    Matplotlib rewrite of the MATLAB ``overlay_brain`` (2mm path only; the
    ``upsample`` 1mm path needs the 1mm MNI volumes which are not bundled).

    Parameters
    ----------
    slice_ : sequence of int
        1-based slice per axis.
    region_masks : sequence of numpy.ndarray, optional
        Regions to overlay.
    colors2use : sequence, optional
        Per-region colours.
    alpha_val : float, optional
        Overlay alpha. Default ``1``.
    underim : numpy.ndarray, optional
        Statistic under-image to display instead of the anatomical.
    applybrainmask : bool, optional
        Mask outside the brain. Default ``True``.
    outerpadding : tuple, optional
        Outer padding.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    from ..brain.io import imgload

    slice_ = np.asarray(slice_, dtype=int)
    brain_im = imgload("MNIbrain.nii.gz")
    brain_mask = imgload("MNImask") > 0
    brain_im = brain_im * brain_mask
    padding = 8

    brain_im4D = combine_brains(brain_im, slice_, brain_mask, padding)
    regions4D = None
    if region_masks is not None:
        regions4D = [
            combine_brains(rm, slice_, brain_mask, padding) for rm in region_masks
        ]
    if applybrainmask:
        brain_mask4D = combine_brains(brain_mask, slice_, brain_mask, padding)
    else:
        brain_mask4D = np.ones_like(brain_im4D)

    if underim is not None:
        underim4D = combine_brains(underim, slice_, brain_mask, padding)
        return viewdata(underim4D, brain_mask4D, regions4D, colors2use, 1, None, alpha_val, ax=ax)
    return viewdata(brain_im4D, brain_mask4D, regions4D, colors2use, 1, None, alpha_val, ax=ax)
