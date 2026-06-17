"""imBH (mirrors StatBrainz/Inference/MHT/imBH.m)."""

import numpy as np

from statbrainz.Inference.MHT.fdrBH import fdrBH

__all__ = ['imBH']


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
