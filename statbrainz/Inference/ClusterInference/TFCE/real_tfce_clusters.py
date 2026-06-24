"""real_tfce_clusters (mirrors StatBrainz/Inference/ClusterInference/TFCE/real_tfce_clusters.m)."""

import numpy as np

from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from statbrainz.Statistics_Functions.Stats_functions.distbn2pval import distbn2pval
from statbrainz.Inference.ClusterInference.numOfConComps import numOfConComps
from statbrainz.Inference.ClusterInference.cluster_im import cluster_im
from .tfce import tfce

__all__ = ['real_tfce_clusters']


def real_tfce_clusters(tstat_orig, mask, tfce_threshold, H=2, E=0.5,
                       connectivity_criterion=None, dh=0.1, h0=0,
                       vec_of_maxima=None):
    """Identify TFCE clusters in observed data that survive a TFCE threshold.

    Parameters
    ----------
    tstat_orig : numpy.ndarray
        2D or 3D test-statistic image.
    mask : numpy.ndarray
        Binary mask the same size as ``tstat_orig``.
    tfce_threshold : float
        TFCE threshold (e.g. from :func:`perm_tfce`) above which a cluster is
        declared significant.
    H : float, optional
        Height exponent (default 2).
    E : float, optional
        Extent exponent (default 0.5).
    connectivity_criterion : int, optional
        Connectivity for connected components (default 8 in 2D, 26 in 3D).
    dh : float, optional
        Step size for cluster formation (default 0.1).
    h0 : float, optional
        Cluster forming threshold (default 0).
    vec_of_maxima : array_like, optional
        Permutation maxima used to compute cluster p-values. If omitted,
        ``cluster_pvals`` is ``None``.

    Returns
    -------
    true_tfce_cluster_im : numpy.ndarray
        Binary image of voxels in significant TFCE clusters.
    true_tfce_clusters : list of numpy.ndarray
        Surviving clusters' 1-based flat index lists.
    cluster_pvals : numpy.ndarray or None
        P-value per (originally detected) cluster, or ``None`` if
        ``vec_of_maxima`` is not provided.
    """
    mask = np.asarray(mask)
    D = mask.ndim
    if connectivity_criterion is None:
        connectivity_criterion = 8 if D == 2 else 26

    tstat_orig = np.asarray(tstat_orig, dtype=float) * (mask > 0)

    eps = np.finfo(float).eps
    real_number_of_tfce_clusters, _, _, real_index_locations = numOfConComps(
        tstat_orig, h0 + eps, connectivity_criterion)
    _, _, real_surviving_tfce_clusters_vec = cluster_im(
        mask.shape, real_index_locations, 0.5)

    max_tfce_within_real_clusters = np.zeros(len(real_surviving_tfce_clusters_vec))
    for I, cluster in enumerate(real_surviving_tfce_clusters_vec):
        cluster_mask, _, _ = cluster_im(mask.shape, [cluster], 0.5)
        tfce_region = tfce(nan2zero(tstat_orig * cluster_mask), H, E,
                           connectivity_criterion, dh)
        max_tfce_within_real_clusters[I] = np.max(tfce_region)

    real_survivor_indices = max_tfce_within_real_clusters > tfce_threshold
    true_tfce_clusters = [c for c, keep in
                          zip(real_surviving_tfce_clusters_vec, real_survivor_indices)
                          if keep]
    true_tfce_cluster_im, _, _ = cluster_im(mask.shape, true_tfce_clusters, 0.5)

    cluster_pvals = None
    if vec_of_maxima is not None:
        cluster_pvals = distbn2pval(vec_of_maxima, max_tfce_within_real_clusters)

    return true_tfce_cluster_im, true_tfce_clusters, cluster_pvals
