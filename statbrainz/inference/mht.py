"""Multiple-hypothesis-testing leaves ported from StatBrainz Inference/MHT.

Faithful ports of: fdp_calc, fdrBH, fdrthresh.
"""

import numpy as np
from scipy import stats

from ..statistics.mvtstat import mvtstat
from ..statistics.convolution import fast_conv
from ..statistics.mask_functions import nan2zero
from ..statistics.stats_functions import distbn2pval

__all__ = [
    "fdp_calc",
    "fdrBH",
    "fdrthresh",
    "imBH",
    "imBH_data",
    "spatialBH",
    "localized_vi",
]


def fdp_calc(rejection_loc_mask, signal_loc_mask):
    """False discovery proportion of a set of rejections.

    Parameters
    ----------
    rejection_loc_mask : numpy.ndarray
        Binary array marking rejected locations.
    signal_loc_mask : numpy.ndarray
        Binary array marking true signal locations.

    Returns
    -------
    float
        Proportion of rejections that are false positives (0 if no rejections).
    """
    rejection_loc_mask = np.asarray(rejection_loc_mask)
    signal_loc_mask = np.asarray(signal_loc_mask)
    if rejection_loc_mask.shape != signal_loc_mask.shape:
        raise ValueError("Mask size mismatch")
    no_signal = 1 - signal_loc_mask
    n_false = np.sum(rejection_loc_mask * no_signal)
    n_rej = np.sum(rejection_loc_mask)
    return n_false / max(n_rej, 1)


def fdrBH(pvalues, alpha=0.05, doBY=False):
    """Benjamini-Hochberg (or Benjamini-Yekutieli) FDR control.

    Parameters
    ----------
    pvalues : numpy.ndarray
        Array of p-values (any shape).
    alpha : float, optional
        Significance level. Default ``0.05``.
    doBY : bool, optional
        If ``True`` use the Benjamini-Yekutieli procedure. Default ``False``.

    Returns
    -------
    rejection_ind : numpy.ndarray
        Boolean array the same shape as ``pvalues``; ``True`` where rejected.
    nrejections : int
        Total number of rejections.
    rejection_locs : numpy.ndarray
        0-based flat (column-major) indices of the rejections.
    maxp : float
        Largest rejected p-value (``nan`` if no rejections).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    Dim = pvalues.shape

    if doBY:
        npv = pvalues.size
        mfactor = np.sum(1.0 / np.arange(1, npv + 1))
        alpha = alpha / mfactor

    flat = pvalues.ravel(order="F")
    npvals = flat.size
    sort_index = np.argsort(flat, kind="stable")
    sorted_pvalues = flat[sort_index]

    BH_upper = np.arange(1, npvals + 1) * alpha / npvals
    BH_vector = sorted_pvalues <= BH_upper
    rejected = np.nonzero(BH_vector)[0]
    if rejected.size == 0:
        nrejections = 0
        rejection_ind = np.zeros(Dim, dtype=bool)
        return rejection_ind, 0, np.array([], dtype=int), np.nan

    nrejections = int(rejected[-1]) + 1  # 1-based "last" count
    rejection_locs = np.sort(sort_index[:nrejections])

    rejection_ind_flat = np.zeros(npvals, dtype=bool)
    rejection_ind_flat[rejection_locs] = True
    rejection_ind = rejection_ind_flat.reshape(Dim, order="F")

    maxp = np.max(pvalues[rejection_ind])
    return rejection_ind, nrejections, rejection_locs, maxp


def fdrthresh(pvals, pval_rejection_ind, df):
    """t-statistic threshold for the largest rejected p-value of a BH procedure.

    Parameters
    ----------
    pvals : numpy.ndarray
        Vector of p-values.
    pval_rejection_ind : numpy.ndarray
        Boolean array indicating which p-values are rejected.
    df : float
        Degrees of freedom.

    Returns
    -------
    float
        ``tinv(maxp, df)`` for the largest rejected p-value.
    """
    pvals = np.asarray(pvals, dtype=float)
    maxp = np.max(pvals[np.asarray(pval_rejection_ind, dtype=bool)])
    return stats.t.ppf(maxp, df)


def imBH(pvals, mask):
    """Benjamini-Hochberg FDR control over the in-mask voxels of a p-value image.

    Parameters
    ----------
    pvals : numpy.ndarray
        Image of p-values.
    mask : numpy.ndarray
        Binary mask (same shape) selecting voxels to test.

    Returns
    -------
    rejection_ind : numpy.ndarray
        Binary image of rejected voxels (``pvals <= maxp``).
    n_rejections : int
        Number of rejected voxels.
    """
    pvals = np.asarray(pvals, dtype=float)
    mask = np.asarray(mask)
    if pvals.shape != mask.shape:
        raise ValueError("The size of pvals must be the same as the size of mask")
    pvals2use = pvals[mask.astype(bool)]
    pval_rejection_ind, _, _, _ = fdrBH(pvals2use)
    if not pval_rejection_ind.any():
        rejection_ind = np.zeros(mask.shape)
    else:
        maxp = np.max(pvals2use[pval_rejection_ind])
        rejection_ind = (pvals <= maxp + np.finfo(float).eps).astype(float)
    return rejection_ind, int(rejection_ind.sum())


def imBH_data(data, mask):
    """BH FDR control on a t-statistic image computed from subject ``data``.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array.
    mask : numpy.ndarray
        Binary spatial mask.

    Returns
    -------
    rejection_ind : numpy.ndarray
        Binary image of rejected voxels.
    n_rejections : int
        Number of rejected voxels.
    """
    data = np.asarray(data, dtype=float)
    nsubj = data.shape[-1]
    data_tstat, _, _, _ = mvtstat(data)
    pvals = 2 * (1 - stats.t.cdf(np.abs(data_tstat), nsubj - 1))
    mask = np.asarray(mask)
    pvals2use = pvals[mask.astype(bool)]
    pval_rejection_ind, _, _, _ = fdrBH(pvals2use)
    if not pval_rejection_ind.any():
        rejection_ind = np.zeros(data.shape[:-1])
    else:
        maxp = np.max(pvals2use[pval_rejection_ind])
        rejection_ind = (pvals <= maxp + np.finfo(float).eps).astype(float)
    return rejection_ind, int(rejection_ind.sum())


def spatialBH(data, FWHM=np.nan):
    """Spatial BH: smooth, compute a t-image, and run BH on its p-values.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array.
    FWHM : float, optional
        Smoothing FWHM applied before the t-statistic. ``nan`` (default) skips
        smoothing.

    Returns
    -------
    rej_locs : numpy.ndarray
        Boolean rejection image.
    nrejections : int
        Number of rejections.
    """
    data = np.asarray(data, dtype=float)
    nsubj = data.shape[-1]
    D = data.ndim - 1
    if not np.isnan(FWHM):
        data, _ = fast_conv(data, FWHM, D)
    tstateval, _, _, _ = mvtstat(data)
    pvals = 1 - stats.t.cdf(tstateval, nsubj - 1)
    rej_locs, nrejections, _, _ = fdrBH(pvals)
    return rej_locs, nrejections


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
