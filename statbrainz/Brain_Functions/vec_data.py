"""vec_data (mirrors StatBrainz/Brain_Functions/vec_data.m)."""

import numpy as np

__all__ = ['vec_data']


def vec_data(data, mask):
    """Extract in-mask voxels from an image array into an ``nvox x nsubj`` matrix.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array; the last axis indexes subjects.
    mask : numpy.ndarray
        ``spatial_dims`` binary array; non-zero entries select voxels.

    Returns
    -------
    numpy.ndarray
        ``nvox x nsubj`` matrix where ``nvox = (mask > 0).sum()``.
    """
    data = np.asarray(data, dtype=float)
    mask = np.asarray(mask)
    sel = (mask > 0).ravel(order="F")
    nsubj = data.shape[-1]
    out = np.empty((int(sel.sum()), nsubj), dtype=float)
    for i in range(nsubj):
        img = data[..., i]
        out[:, i] = img.ravel(order="F")[sel]
    return out
