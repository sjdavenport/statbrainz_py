"""fsannot2mask (mirrors StatBrainz/Surface/ReadSurfaceFiles/fsannot2mask.m)."""

import numpy as np

from .freesurferfiles.read_annotation import read_annotation

__all__ = ['fsannot2mask']


def fsannot2mask(annotfile, region):
    """Extract a region mask from a FreeSurfer annotation file.

    Parameters
    ----------
    annotfile : str
        Path to the ``.annot`` file.
    region : str
        Region name (``'medial_wall'`` is treated as ``'unknown'``).

    Returns
    -------
    mask : numpy.ndarray
        Boolean per-vertex mask.
    region_names : list of str
        All region names in the annotation.
    """
    _, vertex_labels, region_codes = read_annotation(annotfile)
    region_names = region_codes["struct_names"]
    encoded_ids = region_codes["table"][:, 4]
    if region == "medial_wall":
        region = "unknown"
    region_idx = None
    for i, name in enumerate(region_names):
        if name == region:
            region_idx = i
    if region_idx is None:
        raise ValueError("region not found")
    mask = vertex_labels == encoded_ids[region_idx]
    return mask, region_names
