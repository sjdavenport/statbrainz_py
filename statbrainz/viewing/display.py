"""Matplotlib rendering ported from StatBrainz ImageViewing.

These are NOT 1:1 ports — MATLAB's ``imagesc``/``patch``/``hold on``/figure and
``uicontrol`` GUI code has no faithful Python equivalent. They are rewritten on
matplotlib, preserving the image-construction logic (delegated to
:mod:`statbrainz.viewing.helpers`) while using matplotlib for display.

Rewritten: viewthresh, viewdata, viewbrain, overlay_brain.
NOT PORTED (interactive MATLAB GUIs / figure-window shaping with no analog):
brainmove, pan3, viewdata2, sliderGUI, slidergui3, srfgui, fullscreen,
fullscreen2, screenshape, spherescreen, squarescreen, surfscreen, plot_compact,
add_region (patch-based), loadbrains, loadsubs, mytiles, plotImagesInTile,
animatefun, BigFont, cope_display, srf_cope_display*.

matplotlib is an optional dependency (install the ``viz`` extra).
"""

import numpy as np

from ..statistics.mask_functions import nan2zero
from .helpers import colorRegion, viewthresh_image, combine_brains

__all__ = ["viewthresh", "viewdata", "viewbrain", "overlay_brain"]


def _get_ax(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


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


def _apply_rotate(arr, rotate):
    if rotate == 2:
        return arr.T
    if rotate == 3:
        return np.flipud(arr)
    if rotate == 4:
        return np.flipud(arr.T)
    return arr


def viewdata(
    data,
    brain_mask,
    region_masks=None,
    colors2use=None,
    rotate=4,
    bounds=None,
    alpha_val=None,
    outside_color=0,
    ax=None,
):
    """Display ``data`` with optional coloured region overlays and a mask.

    A matplotlib rewrite of the MATLAB ``viewdata``; the overlay/masking logic
    matches, the rendering uses ``imshow`` with alpha.

    Parameters
    ----------
    data : numpy.ndarray
        2D image to display.
    brain_mask : numpy.ndarray
        Mask; outside it is filled with ``outside_color``.
    region_masks : sequence of numpy.ndarray, optional
        Region masks to overlay.
    colors2use : sequence, optional
        One colour per region (default white).
    rotate : int, optional
        Orientation code 1-4 (default 4, matching MATLAB).
    bounds : sequence of slice, optional
        Crop applied to all inputs first.
    alpha_val : float or sequence, optional
        Per-region overlay alpha (default 1).
    outside_color : float, optional
        Greyscale fill outside the mask. Default ``0``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    data = np.asarray(data, dtype=float)
    brain_mask = np.asarray(brain_mask)
    use_regions = region_masks is not None
    if not use_regions:
        region_masks = []
    elif not isinstance(region_masks, (list, tuple)):
        region_masks = [region_masks]

    n_reg = max(len(region_masks), 1)
    if alpha_val is None:
        alpha_val = np.ones(n_reg)
    else:
        alpha_val = np.atleast_1d(alpha_val)
        if alpha_val.size == 1:
            alpha_val = np.repeat(alpha_val[0], n_reg)
    if colors2use is None:
        colors2use = ["white"] * n_reg
    elif not isinstance(colors2use, (list, tuple)):
        colors2use = [colors2use] * n_reg

    if bounds is not None:
        data = data[tuple(bounds)]
        brain_mask = brain_mask[tuple(bounds)]
        region_masks = [rm[tuple(bounds)] for rm in region_masks]

    if data.ndim < 3:
        data = _apply_rotate(data, rotate)
        brain_mask = _apply_rotate(brain_mask, rotate)
        region_masks = [_apply_rotate(np.asarray(rm), rotate) for rm in region_masks]

    ax = _get_ax(ax)
    data_mask = np.isnan(data)
    ax.imshow(nan2zero(data), alpha=(~data_mask).astype(float) if data.ndim < 3 else None)

    if use_regions:
        for I, rm in enumerate(region_masks):
            colored = colorRegion(rm, colors2use[I])
            ax.imshow(colored, alpha=alpha_val[I] * np.asarray(rm, dtype=float))

    outside = np.ones((*brain_mask.shape, 3)) * outside_color
    ax.imshow(outside, alpha=(1 - brain_mask).astype(float))
    ax.axis("off")
    return ax


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
