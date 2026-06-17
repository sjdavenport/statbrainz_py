"""adjacency_matrix (mirrors StatBrainz/Surface/adjacency_matrix.m)."""

import numpy as np
from scipy import sparse

from statbrainz.Surface.SurfStat.SurfStatEdg import SurfStatEdg

__all__ = ['adjacency_matrix']


def adjacency_matrix(srf, metric="ones"):
    """Sparse vertex adjacency matrix of a surface.

    Parameters
    ----------
    srf : dict
        Surface.
    metric : {'ones', 'dist'}, optional
        ``'ones'`` (default) gives a 0/1 adjacency; ``'dist'`` weights edges by
        Euclidean length.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``nvertices x nvertices`` symmetric adjacency matrix.
    """
    edg = SurfStatEdg(srf)
    e0, e1 = edg[:, 0], edg[:, 1]
    nv = srf["nvertices"]
    if metric in ("dist", "distance"):
        first = srf["vertices"][e0]
        second = srf["vertices"][e1]
        dist = np.sqrt(np.sum((first - second) ** 2, axis=1))
    else:
        dist = np.ones(edg.shape[0])
    A = sparse.coo_matrix((dist, (e0, e1)), shape=(nv, nv))
    A = A + sparse.coo_matrix((dist, (e1, e0)), shape=(nv, nv))
    return A.tocsr()
