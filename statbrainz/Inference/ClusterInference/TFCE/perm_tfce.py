"""perm_tfce (mirrors StatBrainz/Inference/ClusterInference/TFCE/perm_tfce.m)."""

import numpy as np

from statbrainz.Brain_Functions.vec_data import vec_data
from statbrainz.Brain_Functions.unwrap import unwrap
from statbrainz.Statistics_Functions.Mask_functions.nan2zero import nan2zero
from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz.Statistics_Functions.Aux_functions.SystemFunctions.loader import loader
from .tfce import tfce

__all__ = ['perm_tfce']


def perm_tfce(data, mask, H=2, E=0.5, connectivity=None, dh=0.1, h0=0,
              alpha=0.05, nperm=1000, show_loader=True, store_perms=False,
              rng=None):
    """Sign-flip permutation threshold for Threshold-Free Cluster Enhancement.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array of subject data.
    mask : numpy.ndarray
        Binary spatial mask of size ``spatial_dims``.
    H : float, optional
        Height exponent (default 2).
    E : float, optional
        Extent exponent (default 0.5).
    connectivity : int, optional
        Connectivity for connected components (default 8 in 2D, 26 in 3D).
    dh : float, optional
        Step size for cluster formation (default 0.1).
    h0 : float, optional
        Cluster forming threshold (default 0).
    alpha : float, optional
        Significance level (default 0.05).
    nperm : int, optional
        Number of permutations including the original (default 1000).
    show_loader : bool, optional
        Display a progress loader (default True).
    store_perms : bool, optional
        Store the permuted (vectorized) TFCE statistics (default False).
    rng : numpy.random.Generator, optional
        Random generator for the sign flips. Defaults to ``np.random.default_rng()``.

    Returns
    -------
    threshold : float
        The ``100*(1-alpha)`` percentile of the permutation maxima.
    vec_of_maxima : numpy.ndarray
        Length-``nperm`` vector of TFCE maxima (first entry is the observed one).
    permuted_tstat_store : numpy.ndarray or float
        ``nvox x nperm`` matrix of permuted TFCE statistics if ``store_perms``,
        else ``numpy.nan``.
    """
    mask = np.asarray(mask)
    dim = mask.shape
    D = mask.ndim
    if connectivity is None:
        connectivity = 8 if D == 2 else 26

    data_vectorized = vec_data(data, mask)
    nsubj = data_vectorized.shape[1]
    mask_bool = mask > 0

    if rng is None:
        rng = np.random.default_rng()

    tstat = unwrap(nan2zero(mvtstat(data_vectorized)[0]), mask)[..., 0]
    tstat_tfce = tfce(tstat * mask_bool, H, E, connectivity, dh, h0)

    vec_of_maxima = np.zeros(nperm)
    vec_of_maxima[0] = np.max(tstat_tfce)

    # Bernoulli sign flips: +/- 1 per subject per permutation.
    random_berns = 2 * (rng.binomial(1, 0.5, size=(nsubj, nperm)) - 0.5)

    if store_perms:
        # Faithful to the MATLAB source: column 0 holds the raw t-stat, while
        # later columns hold TFCE statistics (an inconsistency in the original).
        permuted_tstat_store = np.zeros((int(mask_bool.sum()), nperm))
        permuted_tstat_store[:, 0] = tstat[mask_bool]
    else:
        permuted_tstat_store = np.nan

    for I in range(1, nperm):
        if show_loader:
            loader(I, nperm - 1, 'tfce perm progress:')

        signs = random_berns[:, I]
        data_perm = data_vectorized.copy()
        neg = signs < 0
        data_perm[:, neg] = -data_vectorized[:, neg]

        tstat_perm = unwrap(nan2zero(mvtstat(data_perm)[0]), mask)[..., 0]
        tstat_tfce_perm = tfce(tstat_perm * mask_bool, H, E, connectivity, dh, h0)

        if store_perms:
            permuted_tstat_store[:, I] = tstat_tfce_perm[mask_bool]

        vec_of_maxima[I] = np.max(tstat_tfce_perm[mask_bool])

    threshold = np.percentile(vec_of_maxima, 100 * (1 - alpha))

    return threshold, vec_of_maxima, permuted_tstat_store
