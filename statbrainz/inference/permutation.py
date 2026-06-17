"""Spatial permutation tests ported from StatBrainz Inference/Permutation.

Faithful port of: spintest.

NOTE: ``permutation_power`` in MATLAB is a 2-line demo script, not a function,
so it is not ported.
"""

import numpy as np

from ..surface.ops import spin_surface

__all__ = ["spintest"]


def spintest(X, Y, srf_sphere, nperm=1000, alpha=0.05, show_loader=True, rng=None):
    """Spin-test null threshold for the correlation between two surface maps.

    Parameters
    ----------
    X : dict
        ``{'lh':..., 'rh':...}`` map that is spun.
    Y : dict
        ``{'lh':..., 'rh':...}`` reference map.
    srf_sphere : dict
        Bilateral sphere surfaces ``{'lh':..., 'rh':...}``.
    nperm : int, optional
        Number of permutations. Default ``1000``.
    alpha : float, optional
        Significance level. Default ``0.05``.
    show_loader : bool, optional
        Print progress. Default ``True``.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    threshold : float
        The ``100*(1-alpha)`` percentile of the spun correlations.
    rho_store : numpy.ndarray
        Correlation for each permutation.
    """
    left_rotations, right_rotations = spin_surface(
        X, srf_sphere, nperm, True, show_loader, rng
    )

    allY = np.concatenate([np.asarray(Y["lh"], float).ravel(),
                           np.asarray(Y["rh"], float).ravel()])
    orignan = np.isnan(allY)

    rho_store = np.zeros(nperm)
    for I in range(nperm):
        allperm = np.concatenate([left_rotations[:, I], right_rotations[:, I]])
        permnan = np.isnan(allperm)
        joint = (orignan | permnan)
        a = allperm[~joint]
        b = allY[~joint]
        rho_store[I] = np.corrcoef(a, b)[0, 1]

    threshold = np.percentile(rho_store, 100 * (1 - alpha))
    return threshold, rho_store
