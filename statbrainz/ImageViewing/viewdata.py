"""viewdata (mirrors StatBrainz/ImageViewing/viewdata.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from .colorRegion import colorRegion
from ._display_helpers import _get_ax, _apply_rotate

__all__ = ['viewdata']


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
