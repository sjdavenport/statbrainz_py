"""voxLCE (mirrors StatBrainz/Inference/ClusterInference/TFCE/voxLCE.m)."""

import numpy as np

__all__ = ['voxLCE']


def voxLCE(tfce_tstat, tfce_threshold, H=2, h0=0):
    """Voxelwise significance from a TFCE threshold (inverts the height term).

    Parameters
    ----------
    tfce_tstat : numpy.ndarray
        TFCE image.
    tfce_threshold : float
        TFCE threshold.
    H : float, optional
        Height exponent used to produce ``tfce_tstat``. Default ``2``.
    h0 : float, optional
        Starting height. Default ``0``.

    Returns
    -------
    numpy.ndarray
        Boolean image of significant voxels.
    """
    tfce_tstat = np.asarray(tfce_tstat, dtype=float)
    voxlce_threshold = (tfce_threshold * (H + 1) + h0 ** (H + 1)) ** (1 / (H + 1))
    return tfce_tstat > voxlce_threshold
