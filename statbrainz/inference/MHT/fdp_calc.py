"""fdp_calc (mirrors StatBrainz/Inference/MHT/fdp_calc.m)."""

import numpy as np

__all__ = ['fdp_calc']


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
