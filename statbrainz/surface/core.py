"""Core surface representation + ops ported from StatBrainz Surface.

Faithful ports of: SurfStatEdg, SurfStatSmooth, adjacency_matrix, graph_cc,
srf_face_area, srf_fwhm2niters, smooth_surface, resample_srf_nn, resample_srf.

Surface representation
----------------------
A surface ``srf`` is a dict with:

* ``vertices`` : ``(nvertices, 3)`` float array
* ``faces``    : ``(nfaces, 3)`` int array, **0-based** vertex indices
* ``nvertices``, ``nfaces`` : ints

This differs from MATLAB (which uses 1-based faces); nibabel surface readers
return 0-based faces, so we standardise on that. ``SurfStatEdg`` therefore
returns 0-based vertex-index pairs. All numerical results (smoothing, areas,
adjacency, connected components) are identical to MATLAB regardless of the index
base.

Bilateral surfaces use a dict ``{'lh': srf_l, 'rh': srf_r}`` (and similarly for
data); the relevant functions recurse over hemispheres, matching MATLAB.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

__all__ = [
    "make_srf",
    "SurfStatEdg",
    "SurfStatSmooth",
    "adjacency_matrix",
    "graph_cc",
    "srf_face_area",
    "srf_fwhm2niters",
    "smooth_surface",
    "resample_srf_nn",
    "resample_srf",
]


def make_srf(vertices, faces):
    """Build a surface dict from vertices and (0-based) faces."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    return {
        "vertices": vertices,
        "faces": faces,
        "nvertices": vertices.shape[0],
        "nfaces": faces.shape[0],
    }


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


def graph_cc(data, thresh, adj_matrix):
    """Connected components of the supra-threshold sub-graph.

    Parameters
    ----------
    data : numpy.ndarray
        Per-vertex statistic.
    thresh : float
        Vertices with ``data > thresh`` survive.
    adj_matrix : scipy.sparse matrix
        Vertex adjacency.

    Returns
    -------
    number_of_clusters : int
    occurences : numpy.ndarray
        Counts per distinct cluster size (sizes > 1).
    cluster_sizes : numpy.ndarray
        Distinct cluster sizes (> 1).
    index_locations : list of numpy.ndarray
        0-based vertex indices per cluster.
    """
    from scipy.sparse.csgraph import connected_components

    data = np.asarray(data).ravel()
    survived = data > thresh
    A = adj_matrix.tolil(copy=True)
    notsurv = np.nonzero(~survived)[0]
    A[notsurv, :] = 0
    A[:, notsurv] = 0
    A = A.tocsr()

    n_comp, labels = connected_components(A, directed=False)
    labels = labels.copy()
    labels[notsurv] = -1  # mark removed vertices

    valid = [lab for lab in np.unique(labels) if lab >= 0]
    index_locations = [np.nonzero(labels == lab)[0] for lab in valid]
    sizes = np.array([len(idx) for idx in index_locations])

    cluster_sizes = np.unique(sizes[sizes > 1]) if sizes.size else np.array([], int)
    occurences = np.array(
        [np.sum(sizes == s) for s in cluster_sizes], dtype=int
    )
    return len(index_locations), occurences, cluster_sizes, index_locations


def srf_face_area(srf):
    """Per-face triangle areas and per-vertex areas (1/3 of incident faces).

    Returns
    -------
    face_areas : numpy.ndarray
        ``(nfaces,)`` triangle areas.
    vertex_areas : numpy.ndarray
        ``(nvertices,)`` vertex areas.
    """
    vertices = srf["vertices"]
    faces = srf["faces"]
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1)

    vertex_areas = np.zeros(srf["nvertices"])
    dpf3 = face_areas / 3
    np.add.at(vertex_areas, faces[:, 0], dpf3)
    np.add.at(vertex_areas, faces[:, 1], dpf3)
    np.add.at(vertex_areas, faces[:, 2], dpf3)
    return face_areas, vertex_areas


def srf_fwhm2niters(FWHM, srf, fudge_factor=None):
    """Convert a smoothing FWHM to a number of edge-averaging iterations.

    Parameters
    ----------
    FWHM : float
        Target FWHM.
    srf : dict
        Surface.
    fudge_factor : float, optional
        Defaults to ``1.478 * (69/40)`` (matches FreeSurfer smoothing).

    Returns
    -------
    int
        Number of iterations.
    """
    if fudge_factor is None:
        fudge_factor = 1.478 * (69 / 40)
    face_area, _ = srf_face_area(srf)
    surface_area = face_area.sum()
    area_per_vertex = surface_area / srf["nvertices"]
    numerator = 1.14 * 4 * np.pi * FWHM**2
    denominator = 8 * np.log(2) * 7 * area_per_vertex
    return int(np.floor(fudge_factor * numerator / denominator + 0.5))


def smooth_surface(srf, data, FWHM, metric="ones", niters=None):
    """Smooth per-vertex ``data`` on ``srf`` (recurses over hemispheres).

    Parameters
    ----------
    srf : dict
        Surface, or a bilateral ``{'lh':..., 'rh':...}`` dict.
    data : numpy.ndarray or dict
        Per-vertex data (or ``{'lh':..., 'rh':...}``).
    FWHM : float
        Smoothing FWHM.
    metric : {'ones', 'dist'}, optional
        Edge weighting.
    niters : int, optional
        Iterations; defaults to :func:`srf_fwhm2niters`.

    Returns
    -------
    numpy.ndarray or dict
        Smoothed data.
    """
    if "lh" in srf or "rh" in srf:
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in srf:
                out[hemi] = smooth_surface(srf[hemi], data[hemi], FWHM, metric)
        return out
    if niters is None:
        niters = srf_fwhm2niters(FWHM, srf)
    return SurfStatSmooth(srf, data, FWHM, metric, niters)


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


def resample_srf(surface_data, srfin, srfout, intertype="nn"):
    """Resample ``surface_data`` from ``srfin`` onto ``srfout`` (nearest neighbour).

    Parameters
    ----------
    surface_data : numpy.ndarray or dict
        Per-vertex data on ``srfin`` (or bilateral dict).
    srfin, srfout : dict
        Source and target surfaces.
    intertype : {'nn'}, optional
        Interpolation type. Only nearest-neighbour is implemented (as in MATLAB).

    Returns
    -------
    numpy.ndarray or dict
        Resampled data on ``srfout``.
    """
    if isinstance(surface_data, dict) and ("lh" in surface_data or "rh" in surface_data):
        nn = resample_srf_nn(srfin, srfout)
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in surface_data:
                out[hemi] = np.asarray(surface_data[hemi])[nn[hemi]]
        return out
    if intertype in ("nn", "nearestneighbour"):
        nn_idx = resample_srf_nn(srfin, srfout)
        return np.asarray(surface_data)[nn_idx]
    raise NotImplementedError(f"intertype {intertype!r} not implemented")
