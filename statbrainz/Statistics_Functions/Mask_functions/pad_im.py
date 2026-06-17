"""pad_im (mirrors StatBrainz/Statistics_Functions/Mask_functions/pad_im.m)."""

import numpy as np

__all__ = ['pad_im']


def pad_im(im, lower_padding, upper_padding=None):
    """Pad (or crop) an image at the lower and upper extents of each dimension.

    Parameters
    ----------
    im : numpy.ndarray
        A 2D, 3D, or 4D array.
    lower_padding : int or sequence of int
        Padding at the lower end of each dimension. Negative values crop.
    upper_padding : int or sequence of int, optional
        Padding at the upper end. Defaults to ``lower_padding``.

    Returns
    -------
    numpy.ndarray
        The padded (or cropped) image.
    """
    im = np.asarray(im)
    lower = np.atleast_1d(np.asarray(lower_padding, dtype=int))
    if upper_padding is None:
        upper = lower.copy()
    else:
        upper = np.atleast_1d(np.asarray(upper_padding, dtype=int))

    if lower[0] >= 0:
        # Pad with zeros.
        pad_width = [(int(lower[d]), int(upper[d])) for d in range(im.ndim)]
        return np.pad(im, pad_width, mode="constant", constant_values=0)

    # Crop. MATLAB only implements crop for 3D (all dims) and 4D (spatial only).
    lo = np.abs(lower)
    up = np.abs(upper)
    slices = []
    for d in range(im.ndim):
        if im.ndim == 4 and d == 3:
            slices.append(slice(None))
        else:
            slices.append(slice(int(lo[d]), im.shape[d] - int(up[d])))
    return im[tuple(slices)]
