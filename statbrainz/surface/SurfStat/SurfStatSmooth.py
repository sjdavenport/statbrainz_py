"""SurfStatSmooth (mirrors StatBrainz/Surface/SurfStat/SurfStatSmooth.m)."""

import numpy as np

from statbrainz.Surface.SurfStat.SurfStatEdg import SurfStatEdg

__all__ = ['SurfStatSmooth']


def SurfStatSmooth(srf, Y, FWHM=None, metric="ones", niters=None):
    """Iteratively smooth surface data by averaging over edges.

    Parameters
    ----------
    srf : dict
        Surface.
    Y : numpy.ndarray
        Data to smooth: ``(nsurf, nvertices)`` (or ``(nvertices,)`` for one map).
    FWHM : float, optional
        Used only to derive a default ``niters`` if ``niters`` is omitted.
    metric : {'ones', 'dist'}, optional
        Edge weighting. ``'ones'`` (default) uses uniform weights; ``'dist'``
        weights by inverse Euclidean edge length.
    niters : int, optional
        Number of smoothing iterations. Defaults to
        ``ceil(FWHM**2 / (2*log(2)))``.

    Returns
    -------
    numpy.ndarray
        Smoothed data, same shape as ``Y``.
    """
    Y = np.asarray(Y, dtype=float)
    squeeze_back = False
    if Y.ndim == 1:
        Y = Y[None, :]
        squeeze_back = True
    n, v = Y.shape

    if niters is None:
        if FWHM is None:
            raise ValueError("Provide niters or FWHM")
        niters = int(np.ceil(FWHM**2 / (2 * np.log(2))))

    edg = SurfStatEdg(srf)
    e0, e1 = edg[:, 0], edg[:, 1]

    if metric == "dist":
        first = srf["vertices"][e0]
        second = srf["vertices"][e1]
        dist = np.sqrt(np.sum((first - second) ** 2, axis=1))
        dist = 1.0 / dist
        Y1 = (
            np.bincount(e0, weights=dist, minlength=v)
            + np.bincount(e1, weights=dist, minlength=v)
        )
        Y1 = 2 * Y1
    else:
        Y1 = (
            np.bincount(e0, weights=np.full(e0.shape, 2.0), minlength=v)
            + np.bincount(e1, weights=np.full(e1.shape, 2.0), minlength=v)
        )

    out = Y.copy()
    for i in range(n):
        Ys = out[i].copy()
        for _ in range(niters):
            Yedg = Ys[e0] + Ys[e1]
            if metric == "dist":
                Yedg = Yedg * dist
            Ys = (
                np.bincount(e0, weights=Yedg, minlength=v)
                + np.bincount(e1, weights=Yedg, minlength=v)
            ) / Y1
        out[i] = Ys

    if squeeze_back:
        return out[0]
    return out
