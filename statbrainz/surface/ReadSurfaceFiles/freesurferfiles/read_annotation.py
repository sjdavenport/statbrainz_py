"""read_annotation (mirrors StatBrainz/Surface/ReadSurfaceFiles/freesurferfiles/read_annotation.m)."""

import numpy as np
import nibabel as nib

__all__ = ['read_annotation']


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
