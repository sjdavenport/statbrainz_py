"""tfce (mirrors StatBrainz/Inference/ClusterInference/TFCE/tfce.m)."""

import numpy as np
from scipy import ndimage

from statbrainz.Inference.ClusterInference._shared import _structure, _pixel_idx_lists

__all__ = ['tfce']


def _default_connectivity(D):
    return 8 if D == 2 else 26


def tfce(image, H=2, E=0.5, connectivity=None, dh=0.1, h0=0):
    """Threshold-Free Cluster Enhancement of a statistic image.

    Parameters
    ----------
    image : numpy.ndarray
        2D or 3D statistic image.
    H : float, optional
        Height exponent. Default ``2``.
    E : float, optional
        Extent exponent. Default ``0.5``.
    connectivity : int, optional
        2D: 4/8 (default 8). 3D: 6/18/26 (default 26).
    dh : float, optional
        Height step. Default ``0.1``.
    h0 : float, optional
        Starting height. Default ``0``.

    Returns
    -------
    numpy.ndarray
        TFCE-enhanced image, same shape as ``image``.
    """
    image = np.asarray(image, dtype=float)
    D = image.ndim
    if connectivity is None:
        connectivity = _default_connectivity(D)
    structure = _structure(D, connectivity)

    threshs = np.arange(h0, image.max() + dh, dh)
    threshs = threshs[1:]  # drop the first (== h0), matching MATLAB
    nvox = image.size
    vals = np.zeros(nvox)

    flat_order = "F"
    for h in threshs:
        labeled, num = ndimage.label(image >= h, structure=structure)
        clustsize = np.zeros(nvox)
        if num:
            idx_lists = _pixel_idx_lists(labeled, num)
            for idx in idx_lists:
                clustsize[idx - 1] = len(idx)  # 1-based flat -> 0-based
        vals = vals + (clustsize**E) * (h**H)

    tfce_im = (vals * dh).reshape(image.shape, order=flat_order)
    return tfce_im
