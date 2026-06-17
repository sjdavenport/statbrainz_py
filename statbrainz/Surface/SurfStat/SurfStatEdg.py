"""SurfStatEdg (mirrors StatBrainz/Surface/SurfStat/SurfStatEdg.m)."""

import numpy as np

__all__ = ['SurfStatEdg']


def SurfStatEdg(srf):
    """Unique edges of a triangular mesh as 0-based vertex-index pairs.

    Parameters
    ----------
    srf : dict
        Surface with a ``faces`` array (0-based).

    Returns
    -------
    numpy.ndarray
        ``(nedges, 2)`` array of sorted, unique vertex-index pairs.
    """
    tri = np.sort(np.asarray(srf["faces"]), axis=1)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [0, 2]], tri[:, [1, 2]]])
    return np.unique(edges, axis=0)
