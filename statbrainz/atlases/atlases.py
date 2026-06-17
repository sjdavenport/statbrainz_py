"""Brain atlas region lookups ported from StatBrainz Atlases.

Faithful ports of: getBrainRegionNames, get_mask, getregion, atlas_masks.

Atlas data (Harvard-Oxford cortical + subcortical, 2mm) is bundled under
``statbrainz/data/Atlases/``. The MATLAB helpers ``findstrings`` / ``capstr``
(not present in the package) are reimplemented inline.

Region indexing
---------------
Atlas voxel value ``k`` (k >= 1) corresponds to the ``k``-th region name (the
``names`` list is 0-based in Python, so value ``k`` -> ``names[k-1]``). Value 0
means "no region". This matches MATLAB's 1-based ``names{regionNum}``.
"""

import os
import re

import numpy as np

from ..brain.io import imgload, data_dir
from ..statistics.morphology import mask_bndry

__all__ = ["getBrainRegionNames", "get_mask", "getregion", "atlas_masks"]


def _atlas_dir():
    return os.path.join(data_dir(), "Atlases")


# atlas_name -> (subdir, xml file, atlas nifti file)
_ATLASES = {
    "HOc": (
        "HarvardOxford",
        "HarvardOxford-Cortical.xml",
        "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz",
    ),
    "HOsc": (
        "HarvardOxford",
        "HarvardOxford-Subcortical.xml",
        "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz",
    ),
}

_LABEL_RE = re.compile(r"<label.*?>(.*)</label>")


def getBrainRegionNames(xml_file):
    """Extract region names from an FSL-style atlas XML file.

    Parameters
    ----------
    xml_file : str
        Path to the atlas ``.xml`` file.

    Returns
    -------
    list of str
        Region names in label order.
    """
    names = []
    with open(xml_file, "r", encoding="ISO-8859-1") as fh:
        for line in fh:
            m = _LABEL_RE.search(line)
            if m:
                names.append(m.group(1))
    return names


def _resolve(atlas_name):
    if atlas_name not in _ATLASES:
        raise ValueError(
            f"Atlas {atlas_name!r} not bundled; available: {sorted(_ATLASES)}"
        )
    subdir, xml, nii = _ATLASES[atlas_name]
    loc = os.path.join(_atlas_dir(), subdir)
    return loc, os.path.join(loc, xml), os.path.join(loc, nii)


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
