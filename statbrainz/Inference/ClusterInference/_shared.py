"""_shared (mirrors StatBrainz/Inference/ClusterInference/_shared.m)."""

import numpy as np
from scipy import ndimage

__all__ = ['_structure', '_pixel_idx_lists']


def _structure(D, connectivity_criterion):
    if D == 2:
        if connectivity_criterion == 4:
            return ndimage.generate_binary_structure(2, 1)
        if connectivity_criterion == 8:
            return ndimage.generate_binary_structure(2, 2)
        raise ValueError("In 2D the connectivity criterion must be 4 or 8")
    if D == 3:
        if connectivity_criterion == 6:
            return ndimage.generate_binary_structure(3, 1)
        if connectivity_criterion == 18:
            return ndimage.generate_binary_structure(3, 2)
        if connectivity_criterion == 26:
            return ndimage.generate_binary_structure(3, 3)
        raise ValueError("In 3D the connectivity criterion must be 6, 18 or 26")
    raise ValueError("The dimension must be 2 or 3.")


def _pixel_idx_lists(labeled, num):
    """Per-label 1-based column-major flat indices (like bwconncomp PixelIdxList)."""
    flat = labeled.ravel(order="F")
    order = np.argsort(flat, kind="stable")
    sorted_labels = flat[order]
    out = []
    for lab in range(1, num + 1):
        lo = np.searchsorted(sorted_labels, lab, side="left")
        hi = np.searchsorted(sorted_labels, lab, side="right")
        out.append(order[lo:hi] + 1)  # 1-based
    return out
