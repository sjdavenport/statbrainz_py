"""region_bndry2D (mirrors StatBrainz/Statistics_Functions/Mask_functions/region_bndry2D.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.doubleim import doubleim
from statbrainz.Statistics_Functions.Mask_functions.mask_bndry import mask_bndry

__all__ = ['region_bndry2D']


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
