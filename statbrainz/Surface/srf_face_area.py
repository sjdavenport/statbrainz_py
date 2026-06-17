"""srf_face_area (mirrors StatBrainz/Surface/srf_face_area.m)."""

import numpy as np

__all__ = ['srf_face_area']


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
