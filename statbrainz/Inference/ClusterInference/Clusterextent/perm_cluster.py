"""perm_cluster (mirrors StatBrainz/Inference/ClusterInference/Clusterextent/perm_cluster.m)."""

import warnings

import numpy as np
from scipy.stats import norm

from statbrainz.Brain_Functions.vec_data import vec_data
from statbrainz.Brain_Functions.unwrap import unwrap
from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz.Statistics_Functions.Aux_functions.SystemFunctions.loader import loader
from statbrainz.Inference.ClusterInference.numOfConComps import numOfConComps

__all__ = ['perm_cluster']


def perm_cluster(data, mask, CDT=None, connectivity=None, alpha=0.05,
                 nperm=1000, show_loader=True, store_perms=False, rng=None):
    """Sign-flip permutation cluster-size threshold for voxelwise inference.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array of subject data.
    mask : numpy.ndarray
        Binary spatial mask of size ``spatial_dims``.
    CDT : float, optional
        Cluster defining threshold (default ``norminv(0.99)`` ~ 2.33).
    connectivity : int, optional
        Connectivity for connected components (default 8 in 2D, 26 in 3D).
    alpha : float, optional
        Significance level (default 0.05).
    nperm : int, optional
        Number of permutations including the original (default 1000).
    show_loader : bool, optional
        Display a progress loader (default True).
    store_perms : bool, optional
        Store the permuted (vectorized) t-statistics (default False).
    rng : numpy.random.Generator, optional
        Random generator for the sign flips. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    threshold : float
        The ``100*(1-alpha)`` percentile of the maximum cluster sizes, or NaN
        if no above-threshold voxels exist in the observed statistic.
    vec_of_max_cluster_sizes : numpy.ndarray
        Length-``nperm`` vector of maximum cluster sizes.
    permuted_tstat_store : numpy.ndarray or float
        ``nvox x nperm`` matrix of permuted t-statistics if ``store_perms``,
        else ``numpy.nan``.
    """
    mask = np.asarray(mask)
    D = mask.ndim
    if connectivity is None:
        connectivity = 8 if D == 2 else 26
    if CDT is None:
        CDT = norm.ppf(0.99)

    data_vectorized = vec_data(data, mask)
    nsubj = data_vectorized.shape[1]
    mask_bool = mask > 0

    if rng is None:
        rng = np.random.default_rng()

    tstat = unwrap(nan2zero(mvtstat(data_vectorized)[0]), mask)[..., 0]
    _, _, sizes, _ = numOfConComps(tstat * mask_bool, CDT, connectivity)

    vec_of_max_cluster_sizes = np.zeros(nperm)
    if sizes.size == 0:
        warnings.warn('No superthreshold t-stat voxels found')
        threshold = np.nan
        return threshold, vec_of_max_cluster_sizes, np.nan
    vec_of_max_cluster_sizes[0] = np.max(sizes)

    random_berns = 2 * (rng.binomial(1, 0.5, size=(nsubj, nperm)) - 0.5)

    if store_perms:
        permuted_tstat_store = np.zeros((int(mask_bool.sum()), nperm))
        permuted_tstat_store[:, 0] = tstat[mask_bool]
    else:
        permuted_tstat_store = np.nan

    for I in range(1, nperm):
        if show_loader:
            loader(I, nperm - 1, 'cluster perm progress:')

        signs = random_berns[:, I]
        data_perm = data_vectorized.copy()
        neg = signs < 0
        data_perm[:, neg] = -data_vectorized[:, neg]

        tstat_perm = unwrap(nan2zero(mvtstat(data_perm)[0]), mask)[..., 0]
        _, _, sizes_perm, _ = numOfConComps(tstat_perm * mask_bool, CDT, connectivity)

        if store_perms:
            permuted_tstat_store[:, I] = tstat_perm[mask_bool]

        vec_of_max_cluster_sizes[I] = np.max(sizes_perm) if sizes_perm.size else 0

    threshold = np.percentile(vec_of_max_cluster_sizes, 100 * (1 - alpha))

    return threshold, vec_of_max_cluster_sizes, permuted_tstat_store
