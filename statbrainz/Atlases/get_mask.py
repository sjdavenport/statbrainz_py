"""get_mask (mirrors StatBrainz/Atlases/get_mask.m)."""

import numpy as np

from .getBrainRegionNames import getBrainRegionNames
from ._shared import _resolve
from statbrainz.ImageViewing.imgload import imgload

__all__ = ['get_mask']


def get_mask(atlas_name, region_name, together=True):
    """Build a mask for one or more atlas regions by name.

    Parameters
    ----------
    atlas_name : str
        Atlas key (e.g. ``'HOc'``, ``'HOsc'``).
    region_name : str
        Region name substring (case-insensitive), or ``'all'`` for every region.
    together : bool, optional
        If ``True`` (default) all matching regions share value 1; otherwise the
        I-th matching region gets value ``I`` (1-based).

    Returns
    -------
    mask : numpy.ndarray
        The region mask.
    indices : list of int
        1-based atlas indices of the matched regions.
    areas : list of str
        Matched region names.
    """
    loc, xml_path, nii_path = _resolve(atlas_name)
    names = getBrainRegionNames(xml_path)

    if region_name == "all":
        indices = list(range(1, len(names) + 1))
        areas = list(names)
    else:
        rl = region_name.lower()
        indices = []
        areas = []
        for i, name in enumerate(names):
            if rl in name.lower():
                indices.append(i + 1)  # 1-based atlas value
                areas.append(name)

    atlas = imgload(nii_path)
    mask = np.zeros(atlas.shape)
    for I, idx in enumerate(indices):
        if together:
            mask = mask + (atlas == idx)
        else:
            mask = mask + (I + 1) * (atlas == idx)
    return mask, indices, areas
