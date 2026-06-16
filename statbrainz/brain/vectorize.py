"""Voxel vectorization helpers ported from StatBrainz Brain Functions.

Faithful ports of: vec_data, unwrap.

MATLAB extracts in-mask voxels with ``img(mask>0)``, which walks the array in
**column-major (Fortran)** order. To round-trip identically we use Fortran
ordering for the mask flattening here. (If you only ever pair ``vec_data`` with
``unwrap`` from this package the ordering is internally consistent regardless,
but Fortran order is what matches MATLAB voxel indices.)
"""

import numpy as np

__all__ = ["vec_data", "unwrap"]


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
