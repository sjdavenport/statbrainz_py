"""fdrBH (mirrors StatBrainz/Inference/MHT/fdrBH.m)."""

import numpy as np

__all__ = ['fdrBH']


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
