"""localized_csi (mirrors StatBrainz/Inference/ClusterInference/Clusterextent/localized_csi.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from statbrainz.Statistics_Functions.Stats_functions.distbn2pval import distbn2pval
from statbrainz.Statistics_Functions.Aux_functions.SystemFunctions.loader import loader
from statbrainz.Inference.ClusterInference.numOfConComps import numOfConComps

__all__ = ['localized_csi']


def localized_csi(tstat_orig, region_masks, vec_of_maxima,
                  connectivity_criterion=None, CDT=3.1, show_loader=True):
    """Localized cluster size inference giving a p-value per region.

    For each region the largest cluster size within the region is compared to
    the permutation/bootstrap distribution of maximum cluster sizes.

    Parameters
    ----------
    tstat_orig : numpy.ndarray
        2D or 3D test-statistic image.
    region_masks : numpy.ndarray or sequence of numpy.ndarray
        A single region mask or a list of region masks.
    vec_of_maxima : array_like
        Distribution of permutation/bootstrap maxima.
    connectivity_criterion : int, optional
        Connectivity for connected components (default 8 in 2D, 26 in 3D).
    CDT : float, optional
        Cluster defining threshold (default 3.1).
    show_loader : bool, optional
        Display a progress loader (default True).

    Returns
    -------
    pvals : numpy.ndarray
        One p-value per region mask.
    max_cluster_within_region : numpy.ndarray
        Maximum cluster size within each region.
    """
    if isinstance(region_masks, np.ndarray):
        region_masks = [region_masks]

    D = np.asarray(region_masks[0]).ndim
    if connectivity_criterion is None:
        connectivity_criterion = 8 if D == 2 else 26

    tstat_orig = np.asarray(tstat_orig, dtype=float)
    nmasks = len(region_masks)
    max_cluster_within_region = np.zeros(nmasks)
    for I in range(nmasks):
        if show_loader:
            loader(I + 1, nmasks, 'Progress:')
        region_mask = np.asarray(region_masks[I])
        number_of_clusters, _, sizes, _ = numOfConComps(
            nan2zero(tstat_orig * region_mask), CDT, connectivity_criterion)
        if number_of_clusters > 0:
            max_cluster_within_region[I] = np.max(sizes)

    pvals = distbn2pval(vec_of_maxima, max_cluster_within_region)

    return pvals, max_cluster_within_region
