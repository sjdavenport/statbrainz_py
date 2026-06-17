"""Pure (renderer-free) image-building helpers ported from StatBrainz ImageViewing.

Faithful ports of the array-producing parts of: colorRegion, peak2circle,
viewthresh (image build), combine_brains, custom_colormap.

These return numpy arrays (RGB images, masks, montages, colormaps) and contain
no plotting calls, so they are testable and match MATLAB exactly. The actual
``imagesc``/``patch`` rendering lives in :mod:`statbrainz.viewing.display`.
"""

import numpy as np

from ..statistics.image_ops import pad_im
from ..statistics.mask_bounds import mask_bounds

__all__ = [
    "colorRegion",
    "peak2circle",
    "viewthresh_image",
    "combine_brains",
    "custom_colormap",
]

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


def combine_brains(brain_im, slice_, brain_mask, padding=5, outerpad=(0, 0), use_bounds=True):
    """Build a 3-view montage (sagittal/coronal/axial) of a brain volume.

    Parameters
    ----------
    brain_im : numpy.ndarray
        ``91x109x91`` (or ``182x218x182``) volume.
    slice_ : sequence of int
        1-based slice index per axis.
    brain_mask : numpy.ndarray
        Mask used to crop to bounds (when ``use_bounds``).
    padding : int, optional
        Gap between views. Default ``5`` (doubled at 2x resolution).
    outerpad : tuple, optional
        Outer padding (``{lower, upper}`` style; passed to :func:`pad_im`).
    use_bounds : bool, optional
        Crop to the mask bounds first. Default ``True``.

    Returns
    -------
    numpy.ndarray
        2D montage image.
    """
    brain_im = np.asarray(brain_im, dtype=float)
    slice_ = list(np.asarray(slice_, dtype=int))

    if brain_im.shape == (182, 218, 182):
        scale = 2
        padding = 2 * padding
    else:
        scale = 1

    def _view(vol, axis, idx0):
        if axis == 0:
            plane = vol[idx0, :, :]
        elif axis == 1:
            plane = vol[:, idx0, :]
        else:
            plane = vol[:, :, idx0]
        # MATLAB: flipud(squeeze(...)') -> transpose then flip rows
        return np.flipud(plane.T)

    if not use_bounds:
        b1 = _view(brain_im, 0, slice_[0] - 1)
        b2 = _view(brain_im, 1, slice_[1] - 1)
        b3 = _view(brain_im, 2, slice_[2] - 1)
        combined2 = np.hstack([b1, b2])
        combined = np.zeros((scale * 109, scale * 291))
        combined[scale * 8:scale * 99, 0:scale * 200] = combined2
        combined[:, scale * 200:scale * 291] = b3
        if np.any(np.asarray(outerpad) > 0):
            combined = pad_im(combined, outerpad)
        return combined

    bounds, _ = mask_bounds(brain_mask)
    for i in range(3):
        slice_[i] = slice_[i] - (bounds[i].start + 1) + 1  # adjust into crop (1-based)
    brain_im = brain_im[tuple(bounds)]

    b1 = _view(brain_im, 0, slice_[0] - 1)
    b2 = _view(brain_im, 1, slice_[1] - 1)
    b3 = _view(brain_im, 2, slice_[2] - 1)

    xlen1 = b1.shape[1]
    xlen2 = b1.shape[1] + b2.shape[1] + padding
    xlength = b1.shape[1] + b2.shape[1] + b3.shape[1] + 2 * padding
    combined = np.zeros((b3.shape[0], xlength))
    ystart = int(np.floor((b3.shape[0] - b1.shape[0]) / 2)) + 3
    combined[ystart:ystart + b1.shape[0], 0:xlen1] = b1
    combined[ystart:ystart + b1.shape[0], xlen1 + padding:xlen2] = b2
    combined[:, xlen2 + padding:xlength] = b3
    return combined


def custom_colormap(color1, color2, num_colors=64):
    """Linear RGB colormap interpolating ``color1`` -> ``color2``.

    Parameters
    ----------
    color1, color2 : sequence of float
        Endpoint RGB triples.
    num_colors : int, optional
        Number of colours. Default ``64``.

    Returns
    -------
    numpy.ndarray
        ``(num_colors, 3)`` colormap.
    """
    color1 = np.asarray(color1, dtype=float)
    color2 = np.asarray(color2, dtype=float)
    return np.column_stack(
        [np.linspace(color1[k], color2[k], num_colors) for k in range(3)]
    )
