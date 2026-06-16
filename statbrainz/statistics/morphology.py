"""Binary mask morphology ported from StatBrainz Mask_functions.

Faithful ports of: dilate_mask, mask_bndry, expand2mask, doubleim, region_bndry2D.

MATLAB ``imdilate(mask, ones((2k+1)^D))`` is reproduced with
``scipy.ndimage.binary_dilation`` using an all-ones box structuring element of
side ``2k+1`` (a full/box neighbourhood, NOT the default cross).
"""

import numpy as np
from scipy import ndimage

from .mask_bounds import mask_bounds

__all__ = ["dilate_mask", "mask_bndry", "expand2mask", "doubleim", "region_bndry2D"]


def dilate_mask(mask, dilation=1):
    """Dilate (``dilation>0``) or erode (``dilation<0``) a binary mask.

    Uses a full box structuring element of side ``2*abs(dilation)+1``.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    dilation : int, optional
        Amount to dilate (positive) or erode (negative). Default ``1``.

    Returns
    -------
    numpy.ndarray
        Boolean dilated/eroded mask.
    """
    mask = np.asarray(mask) > 0
    D = mask.ndim
    if dilation == 0:
        return mask.copy()
    k = abs(int(dilation))
    structure = np.ones((2 * k + 1,) * D, dtype=bool)
    if dilation > 0:
        out = ndimage.binary_dilation(mask, structure=structure)
    else:
        out = ~ndimage.binary_dilation(~mask, structure=structure)
    return out.astype(bool)


def mask_bndry(mask, nonboundary_mask=None):
    """Outer and inner one-voxel boundaries of a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    nonboundary_mask : numpy.ndarray, optional
        If given, boundary voxels inside this mask are removed.

    Returns
    -------
    outer_bndry, inner_bndry : numpy.ndarray
        Boundary masks (as float arrays, matching MATLAB subtraction output).
    """
    mask_f = (np.asarray(mask) > 0).astype(float)
    dilated = dilate_mask(mask, 1).astype(float)
    shrunk = dilate_mask(mask, -1).astype(float)
    inner_bndry = mask_f - shrunk
    outer_bndry = dilated - mask_f
    if nonboundary_mask is not None:
        nb = np.asarray(nonboundary_mask, dtype=float)
        inner_bndry = inner_bndry * (1 - nb)
        outer_bndry = outer_bndry * (1 - nb)
    return outer_bndry, inner_bndry


def expand2mask(im, mask, padding=0):
    """Place a cropped sub-image ``im`` back into a full 2D MNI-sized image.

    Mirrors the MATLAB ``expand2mask`` which is hard-coded to a ``91 x 109``
    output for 2D input.

    Parameters
    ----------
    im : numpy.ndarray
        Cropped image to place back.
    mask : numpy.ndarray
        Mask defining the crop bounds.
    padding : int, optional
        Padding used when the bounds were computed. Default ``0``.

    Returns
    -------
    numpy.ndarray
        ``91 x 109`` array with ``im`` placed at the mask bounds.
    """
    bounds, _ = mask_bounds(mask, padding)
    D = np.asarray(im).ndim
    if len(bounds) == 3 and D == 2:
        bounds = bounds[:2]
    out = np.zeros((91, 109))
    out[tuple(bounds)] = im
    return out


def doubleim(orig_data):
    """Double the resolution of a ``91 x 109 x 91`` image to ``182 x 218 x 182``.

    Each original voxel is replicated into a ``2 x 2 x 2`` block, matching the
    MATLAB ``doubleim``.

    Parameters
    ----------
    orig_data : numpy.ndarray
        ``91 x 109 x 91`` array.

    Returns
    -------
    numpy.ndarray
        ``182 x 218 x 182`` array.
    """
    orig_data = np.asarray(orig_data, dtype=float)
    new_data = np.zeros((182, 218, 182))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                new_data[i:182:2, j:218:2, k:182:2] = orig_data
    return new_data


def region_bndry2D(regions, mask):
    """2D boundary image (axial slice 80) summed over a list of 3D regions.

    Mirrors MATLAB ``region_bndry2D`` (hard-coded to 2mm MNI size and slice 80).

    Parameters
    ----------
    regions : sequence of numpy.ndarray
        List of ``91 x 109 x 91`` region masks.
    mask : numpy.ndarray
        ``91 x 109 x 91`` brain mask.

    Returns
    -------
    numpy.ndarray
        ``182 x 218`` boundary-sum image.
    """
    # MATLAB slice index 80 (1-based) -> 0-based 79.
    z = 79
    masksum2D = np.zeros((182, 218))
    nonboundary_mask = (1 - doubleim(mask))[:, :, z]
    for region in regions:
        doublemask = doubleim(region)
        outer, _ = mask_bndry(doublemask[:, :, z], nonboundary_mask)
        masksum2D = masksum2D + outer
        nonboundary_mask = nonboundary_mask + doublemask[:, :, z]
    return masksum2D
