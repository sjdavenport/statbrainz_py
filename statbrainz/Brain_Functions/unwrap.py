"""unwrap (mirrors StatBrainz/Brain_Functions/unwrap.m)."""

import numpy as np

__all__ = ['unwrap']


def unwrap(data, mask):
    """Place vectorized voxel data back into image space.

    Parameters
    ----------
    data : numpy.ndarray or list of numpy.ndarray
        ``nvox x nsubj`` matrix (or a list of such matrices).
    mask : numpy.ndarray
        ``D``-dimensional array whose non-zero entries identify the voxels
        corresponding to the rows of ``data``.

    Returns
    -------
    numpy.ndarray or list
        Array of shape ``(*mask.shape, nsubj)`` with data at in-mask voxels and
        zeros elsewhere. If ``data`` is a list, a list is returned.
    """
    if isinstance(data, (list, tuple)):
        return [unwrap(d, mask) for d in data]

    data = np.asarray(data, dtype=float)
    mask = np.asarray(mask)
    if data.ndim == 1:
        data = data[:, None]
    nsubj = data.shape[1]
    sel = (mask > 0).ravel(order="F")

    out = np.zeros((*mask.shape, nsubj), dtype=float)
    for i in range(nsubj):
        flat = np.zeros(mask.size, dtype=float)
        flat[sel] = data[:, i]
        out[..., i] = flat.reshape(mask.shape, order="F")
    return out
