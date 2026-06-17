"""combine_brains (mirrors StatBrainz/ImageViewing/combine_brains.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.pad_im import pad_im
from statbrainz.Statistics_Functions.Mask_functions.mask_bounds import mask_bounds

__all__ = ['combine_brains']


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
