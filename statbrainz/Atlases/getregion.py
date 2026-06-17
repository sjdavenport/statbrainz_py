"""getregion (mirrors StatBrainz/Atlases/getregion.m)."""

import numpy as np

from .getBrainRegionNames import getBrainRegionNames
from ._shared import _resolve
from statbrainz.ImageViewing.imgload import imgload

__all__ = ['getregion']


def getregion(points, atlas="default"):
    """Look up the atlas region name(s) at given voxel coordinate(s).

    Parameters
    ----------
    points : sequence of int or sequence of such
        A single 1-based ``(x, y, z)`` coordinate, or a list of them.
    atlas : str, optional
        ``'default'`` / ``'HOc'`` (cortical, plus subcortical when 'default').

    Returns
    -------
    HOc_regions : list
        Cortical region name (or None) per point.
    HOsc_regions : list
        Subcortical region name (or None) per point (only for 'default').
    """
    cort_loc, cort_xml, cort_nii = _resolve("HOc")
    HOc_names = getBrainRegionNames(cort_xml)
    HOc_atlas = imgload(cort_nii)

    do_sub = atlas == "default"
    if do_sub:
        _, sub_xml, sub_nii = _resolve("HOsc")
        HOsc_names = getBrainRegionNames(sub_xml)
        HOsc_atlas = imgload(sub_nii)

    single = np.ndim(points[0]) == 0
    if single:
        points = [points]

    HOc_regions = [None] * len(points)
    HOsc_regions = [None] * len(points)
    for I, p in enumerate(points):
        x, y, z = (int(p[0]) - 1, int(p[1]) - 1, int(p[2]) - 1)  # 1-based -> 0-based
        region_num = int(HOc_atlas[x, y, z])
        if region_num != 0:
            HOc_regions[I] = HOc_names[region_num - 1].rstrip()
        if do_sub:
            sub_num = int(HOsc_atlas[x, y, z])
            if sub_num != 0:
                HOsc_regions[I] = HOsc_names[sub_num - 1].rstrip()
    return HOc_regions, HOsc_regions
