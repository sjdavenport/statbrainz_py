"""mask_bounds (mirrors StatBrainz/Statistics_Functions/Mask_functions/mask_bounds.m)."""

import numpy as np

__all__ = ['mask_bounds']


def mask_bounds(mask, padding=0):
    """Find the tight bounds of a binary ``mask`` and crop it.

    Parameters
    ----------
    mask : numpy.ndarray
        A ``D``-dimensional array of zeros and ones.
    padding : int or sequence of int, optional
        Extra voxels to keep around the tight bounds in each direction
        (clipped to the array extent). Scalar applies to all dimensions.
        Default ``0``.

    Returns
    -------
    bounds : list of slice
        Length-``D`` list; ``bounds[d]`` is a 0-based ``slice`` over dimension
        ``d`` (usable directly as ``mask[tuple(bounds)]``).
    bounded_mask : numpy.ndarray
        Boolean array equal to ``mask`` restricted to ``bounds``.
    """
    mask = np.asarray(mask)
    D = mask.ndim
    padding = np.atleast_1d(padding)
    if padding.size < D:
        padding = np.repeat(padding[0], D)

    bounds = []
    for d in range(D):
        axes = tuple(a for a in range(D) if a != d)
        # which slices along axis d contain any non-zero voxel
        nonempty = np.any(mask > 0, axis=axes) if D > 1 else (mask > 0)
        nz = np.nonzero(nonempty)[0]
        lower = max(int(nz[0]) - int(padding[d]), 0)
        upper = min(int(nz[-1]) + int(padding[d]), mask.shape[d] - 1)
        bounds.append(slice(lower, upper + 1))

    bounded_mask = mask[tuple(bounds)].astype(bool)
    return bounds, bounded_mask
