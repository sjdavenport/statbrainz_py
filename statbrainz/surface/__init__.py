"""StatBrainz surface functions (ported from MATLAB)."""

from .core import (
    make_srf,
    SurfStatEdg,
    SurfStatSmooth,
    adjacency_matrix,
    graph_cc,
    srf_face_area,
    srf_fwhm2niters,
    smooth_surface,
    resample_srf_nn,
    resample_srf,
)
from .ops import srf_dilate_mask, srf_contour, srf_noise, spin_surface
from .io import (
    read_fs_geometry,
    freesurfer_read_surf,
    fs2surf,
    gifti2surf,
    load_gifti,
    read_annotation,
    fsannot2mask,
    dpvread,
)

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
    "srf_dilate_mask",
    "srf_contour",
    "srf_noise",
    "spin_surface",
    "read_fs_geometry",
    "freesurfer_read_surf",
    "fs2surf",
    "gifti2surf",
    "load_gifti",
    "read_annotation",
    "fsannot2mask",
    "dpvread",
]
