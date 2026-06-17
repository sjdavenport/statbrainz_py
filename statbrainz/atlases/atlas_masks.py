"""atlas_masks (mirrors StatBrainz/Atlases/atlas_masks.m)."""

import numpy as np

from .getBrainRegionNames import getBrainRegionNames
from ._shared import _resolve
from statbrainz.ImageViewing.imgload import imgload
from statbrainz.Statistics_Functions.Mask_functions.mask_bndry import mask_bndry

__all__ = ['atlas_masks']


def atlas_masks(atlas, get_boundary=False):
    """Return per-region masks and names for a bundled atlas.

    Parameters
    ----------
    atlas : str
        Atlas key (``'HOc'`` or ``'HOsc'``).
    get_boundary : bool, optional
        Return region boundaries instead of filled masks. Default ``False``.

    Returns
    -------
    region_masks : list of numpy.ndarray
        One mask per region.
    region_names : list of str
        Region names.
    """
    loc, xml_path, nii_path = _resolve(atlas)
    region_names = getBrainRegionNames(xml_path)
    atlas_img = imgload(nii_path)
    region_masks = [(atlas_img == (i + 1)).astype(float) for i in range(len(region_names))]
    if get_boundary:
        region_masks = [mask_bndry(rm)[0] for rm in region_masks]
    return region_masks, region_names
