"""Surface file I/O ported from StatBrainz Surface/ReadSurfaceFiles.

Faithful ports of: read_fs_geometry, freesurfer_read_surf, fs2surf, gifti2surf,
load_gifti, read_annotation, fsannot2mask, dpvread.

FreeSurfer / GIfTI parsing is delegated to nibabel rather than reimplementing
the binary format readers. Faces are returned **0-based** (nibabel convention),
matching the surface representation in :mod:`statbrainz.surface.core`.
"""

import numpy as np
import nibabel as nib

from .core import make_srf

__all__ = [
    "read_fs_geometry",
    "freesurfer_read_surf",
    "fs2surf",
    "gifti2surf",
    "load_gifti",
    "read_annotation",
    "fsannot2mask",
    "dpvread",
]


def read_fs_geometry(filepath):
    """Read a FreeSurfer geometry file (via nibabel).

    Returns
    -------
    vertices : numpy.ndarray
        ``(nvertices, 3)``.
    faces : numpy.ndarray
        ``(nfaces, 3)`` 0-based.
    """
    vertices, faces = nib.freesurfer.read_geometry(filepath)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def freesurfer_read_surf(fname):
    """Alias of :func:`read_fs_geometry` (matches the MATLAB function name)."""
    return read_fs_geometry(fname)


def load_gifti(filepath):
    """Load a GIfTI surface, returning ``(vertices, faces)`` (faces 0-based)."""
    g = nib.load(filepath)
    arrays = g.darrays
    vertices = None
    faces = None
    for da in arrays:
        code = da.intent
        name = nib.nifti1.intent_codes.label.get(code, "")
        if name == "pointset" or da.data.ndim == 2 and da.data.shape[1] == 3 and da.data.dtype.kind == "f":
            vertices = np.asarray(da.data, dtype=float)
        elif name == "triangle" or (da.data.dtype.kind in "iu"):
            faces = np.asarray(da.data, dtype=np.int64)
    if vertices is None or faces is None:
        # fall back to the first two arrays
        vertices = np.asarray(arrays[0].data, dtype=float)
        faces = np.asarray(arrays[1].data, dtype=np.int64)
    return vertices, faces


def fs2surf(path4fs, path4fsright=None):
    """Build a surface dict from FreeSurfer geometry file(s).

    Parameters
    ----------
    path4fs : str
        Left (or single) hemisphere geometry file.
    path4fsright : str, optional
        Right hemisphere file; if given, returns a bilateral dict.

    Returns
    -------
    dict
        A surface dict, or ``{'lh':..., 'rh':...}`` with ``hemi`` tags.
    """
    if path4fsright is not None:
        lh = fs2surf(path4fs)
        rh = fs2surf(path4fsright)
        lh["hemi"] = "lh"
        rh["hemi"] = "rh"
        return {"lh": lh, "rh": rh}
    vertices, faces = read_fs_geometry(path4fs)
    return make_srf(vertices, faces)


def gifti2surf(path4gifti, path4giftiright=None):
    """Build a surface dict from GIfTI file(s)."""
    if path4giftiright is not None:
        lh = gifti2surf(path4gifti)
        rh = gifti2surf(path4giftiright)
        lh["hemi"] = "lh"
        rh["hemi"] = "rh"
        return {"lh": lh, "rh": rh}
    vertices, faces = load_gifti(path4gifti)
    return make_srf(vertices, faces)


def read_annotation(annotfile):
    """Read a FreeSurfer annotation file (via nibabel).

    Returns
    -------
    vertices : numpy.ndarray
        Vertex indices (``arange(nvertices)``), to mirror the MATLAB signature.
    vertex_labels : numpy.ndarray
        Per-vertex annotation values (the encoded RGBA ids).
    colortable : dict
        ``{'struct_names': [...], 'table': ndarray}`` mirroring the MATLAB
        ``region_codes`` struct (``table[:, 4]`` is the encoded id).
    """
    labels, ctab, names = nib.freesurfer.read_annot(annotfile)
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    # nibabel maps labels to row indices into ctab; recover the encoded ids so
    # the MATLAB-style `vertex_labels == encoded_id` comparison works.
    encoded = ctab[:, 4]
    vertex_labels = np.where(labels >= 0, encoded[labels], -1)
    colortable = {"struct_names": names, "table": ctab}
    vertices = np.arange(labels.shape[0])
    return vertices, vertex_labels, colortable


def fsannot2mask(annotfile, region):
    """Extract a region mask from a FreeSurfer annotation file.

    Parameters
    ----------
    annotfile : str
        Path to the ``.annot`` file.
    region : str
        Region name (``'medial_wall'`` is treated as ``'unknown'``).

    Returns
    -------
    mask : numpy.ndarray
        Boolean per-vertex mask.
    region_names : list of str
        All region names in the annotation.
    """
    _, vertex_labels, region_codes = read_annotation(annotfile)
    region_names = region_codes["struct_names"]
    encoded_ids = region_codes["table"][:, 4]
    if region == "medial_wall":
        region = "unknown"
    region_idx = None
    for i, name in enumerate(region_names):
        if name == region:
            region_idx = i
    if region_idx is None:
        raise ValueError("region not found")
    mask = vertex_labels == encoded_ids[region_idx]
    return mask, region_names


def dpvread(filename, nvertices):
    """Read a FreeSurfer data-per-vertex (.dpv) text file.

    Parameters
    ----------
    filename : str
        Path to the whitespace-delimited file (5 columns per vertex).
    nvertices : int
        Number of vertices.

    Returns
    -------
    numpy.ndarray
        ``(nvertices, 5)`` array.
    """
    vals = np.fromfile(filename, sep=" ", count=5 * nvertices)
    return vals.reshape((nvertices, 5))
