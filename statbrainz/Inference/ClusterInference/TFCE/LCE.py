"""LCE (mirrors StatBrainz/Inference/ClusterInference/TFCE/LCE.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from statbrainz.Statistics_Functions.Aux_functions.SystemFunctions.loader import loader as _loader
from statbrainz.Statistics_Functions.Stats_functions.distbn2pval import distbn2pval
from statbrainz.Inference.ClusterInference.Clusterextent.index2mask import index2mask
from .tfce import tfce, _default_connectivity

__all__ = ['LCE']


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
