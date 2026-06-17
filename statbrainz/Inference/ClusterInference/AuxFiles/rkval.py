"""rkval (mirrors StatBrainz/Inference/ClusterInference/AuxFiles/rkval.m)."""

import numpy as np

__all__ = ['rkval']


def _fdk(k, d):
    if d == 0 or k == 0:
        return 0
    floork1overd = int(np.floor(k ** (1 / d)))
    ldk = int(
        np.floor(
            (np.log(k) - d * np.log(floork1overd))
            / (np.log(floork1overd + 1) - np.log(floork1overd))
        )
    )
    bdkplus = (floork1overd + 1) ** (d - ldk) * (floork1overd + 2) ** ldk
    bdk = floork1overd ** (d - ldk) * (floork1overd + 1) ** ldk
    return bdkplus + _fdk(k - bdk, d - 1)


def rkval(k, d=3):
    """Compute the ``r_k`` constant used in the cluster TP lower bound.

    Parameters
    ----------
    k : int
        Cluster (size) threshold.
    d : int, optional
        Dimension. Default ``3``.

    Returns
    -------
    float
        ``min_i (fdk(i, d) - i) / fdk(i, d)`` over ``i = 1..k``.
    """
    min_value = np.inf
    for i in range(1, k + 1):
        fdki = _fdk(i, d)
        if fdki != 0:
            min_value = min(min_value, (fdki - i) / fdki)
    return min_value
