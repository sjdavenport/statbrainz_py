"""resample_srf_nn (mirrors StatBrainz/Surface/resample_srf_nn.m)."""

import numpy as np
from scipy.spatial import cKDTree

__all__ = ['resample_srf_nn']


def resample_srf_nn(srfin, srfout):
    """Nearest-neighbour vertex indices mapping ``srfout`` onto ``srfin``.

    Parameters
    ----------
    srfin, srfout : dict
        Source and target surfaces (or bilateral dicts).

    Returns
    -------
    numpy.ndarray or dict
        0-based indices into ``srfin``'s vertices for each ``srfout`` vertex.
    """
    if "lh" in srfin or "rh" in srfin:
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in srfin:
                out[hemi] = resample_srf_nn(srfin[hemi], srfout[hemi])
        return out
    tree = cKDTree(srfin["vertices"])
    _, idx = tree.query(srfout["vertices"])
    return idx
