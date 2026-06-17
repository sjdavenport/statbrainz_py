"""localized_vi (mirrors StatBrainz/Inference/MHT/localized_vi.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from statbrainz.Statistics_Functions.Stats_functions.distbn2pval import distbn2pval

__all__ = ['localized_vi']


def localized_vi(tstat_orig, region_masks, vec_of_maxima):
    """Localized inference: per-region max t-stat and its empirical p-value.

    Parameters
    ----------
    tstat_orig : numpy.ndarray
        Original statistic image.
    region_masks : numpy.ndarray or sequence
        One or more region masks (same shape as ``tstat_orig``).
    vec_of_maxima : numpy.ndarray
        Null distribution of region-max statistics.

    Returns
    -------
    pvals : numpy.ndarray
        Empirical p-value per region.
    max_tstat_within_region : numpy.ndarray
        Maximum within-region statistic per region.
    """
    if not isinstance(region_masks, (list, tuple)):
        region_masks = [region_masks]
    nmasks = len(region_masks)
    max_tstat_within_region = np.zeros(nmasks)
    for i in range(nmasks):
        masked_tstat = nan2zero(tstat_orig * region_masks[i])
        max_tstat_within_region[i] = masked_tstat.max()
    pvals = distbn2pval(vec_of_maxima, max_tstat_within_region)
    return pvals, max_tstat_within_region
