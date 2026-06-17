"""Threshold-Free Cluster Enhancement ported from StatBrainz.

Faithful ports of: tfce, voxLCE, LCE.
"""

import numpy as np
from scipy import ndimage

from ..statistics.mask_functions import nan2zero
from ..statistics.utils import loader as _loader
from ..statistics.stats_functions import distbn2pval
from .cluster import _structure, _pixel_idx_lists, index2mask

__all__ = ["tfce", "voxLCE", "LCE"]


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


def voxLCE(tfce_tstat, tfce_threshold, H=2, h0=0):
    """Voxelwise significance from a TFCE threshold (inverts the height term).

    Parameters
    ----------
    tfce_tstat : numpy.ndarray
        TFCE image.
    tfce_threshold : float
        TFCE threshold.
    H : float, optional
        Height exponent used to produce ``tfce_tstat``. Default ``2``.
    h0 : float, optional
        Starting height. Default ``0``.

    Returns
    -------
    numpy.ndarray
        Boolean image of significant voxels.
    """
    tfce_tstat = np.asarray(tfce_tstat, dtype=float)
    voxlce_threshold = (tfce_threshold * (H + 1) + h0 ** (H + 1)) ** (1 / (H + 1))
    return tfce_tstat > voxlce_threshold


def LCE(
    tstat_orig,
    region_masks,
    vec_of_maxima,
    H=2,
    E=0.5,
    connectivity_criterion=None,
    dh=0.1,
    h0=0,
    show_loader=True,
):
    """Localized cluster enhancement p-values per region.

    Parameters
    ----------
    tstat_orig : numpy.ndarray
        Original statistic image.
    region_masks : numpy.ndarray or sequence
        One or more region masks (each same shape as ``tstat_orig``, or flat
        index lists convertible via :func:`index2mask`).
    vec_of_maxima : numpy.ndarray
        Null distribution of region-max TFCE values (for the p-value calc).
    H, E, connectivity_criterion, dh, h0 : see :func:`tfce`.
    show_loader : bool, optional
        Print progress. Default ``True``.

    Returns
    -------
    pvals : numpy.ndarray
        Empirical p-value per region.
    max_tfce_within_region : numpy.ndarray
        Maximum within-region TFCE per region.
    """
    if not isinstance(region_masks, (list, tuple)):
        region_masks = [region_masks]
    D = np.asarray(region_masks[0]).ndim
    if connectivity_criterion is None:
        connectivity_criterion = _default_connectivity(D)

    nmasks = len(region_masks)
    max_tfce_within_region = np.zeros(nmasks)
    for i in range(nmasks):
        if show_loader:
            _loader(i + 1, nmasks, "Progress:")
        region_mask = np.asarray(region_masks[i])
        if region_mask.shape != np.asarray(tstat_orig).shape:
            region_mask = index2mask(region_mask, np.asarray(tstat_orig).shape)
        tfce_region = tfce(
            nan2zero(tstat_orig * region_mask),
            H,
            E,
            connectivity_criterion,
            dh,
            h0,
        )
        max_tfce_within_region[i] = tfce_region.max()

    pvals = distbn2pval(vec_of_maxima, max_tfce_within_region)
    return pvals, max_tfce_within_region
