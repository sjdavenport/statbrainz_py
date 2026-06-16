"""Image padding / signal generation ported from StatBrainz.

Faithful ports of: pad_im, square_signal.
"""

import numpy as np

__all__ = ["pad_im", "square_signal"]


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


def square_signal(dim, radii, center_locs=None):
    """Generate a binary image with square (hypercuboid) regions of ones.

    Parameters
    ----------
    dim : sequence of int
        Output image dimensions.
    radii : float or sequence of float
        Half-side-length of each square region. Scalar applies to all peaks.
    center_locs : sequence of array_like, optional
        Centre coordinate (1-based, matching MATLAB) per peak. Default is a
        single centred peak at ``dim/2 + 1/2``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``dim`` with ones in the square regions, zeros elsewhere.
    """
    dim = tuple(int(d) for d in dim)
    D = len(dim)
    if center_locs is None:
        center_locs = [np.array(dim) / 2 + 0.5]

    npeaks = len(center_locs)
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    if radii.size == 1:
        radii = np.repeat(radii[0], npeaks)
    elif radii.size != npeaks:
        raise ValueError("Number of radii does not match number of centres")

    signal = np.zeros(dim)
    for j, centre in enumerate(center_locs):
        centre = np.atleast_1d(np.asarray(centre, dtype=float))
        idx = []
        for i in range(D):
            # MATLAB: floor(c-r) : ceil(c+r), 1-based inclusive.
            lo1 = int(np.floor(centre[i] - radii[j]))
            hi1 = int(np.ceil(centre[i] + radii[j]))
            # convert 1-based inclusive range to 0-based slice, clip to bounds
            lo0 = max(lo1 - 1, 0)
            hi0 = min(hi1 - 1, dim[i] - 1)
            idx.append(slice(lo0, hi0 + 1))
        signal[tuple(idx)] = 1
    return signal
