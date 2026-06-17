"""Surface mask / noise / spin operations ported from StatBrainz Surface.

Faithful ports of: srf_dilate_mask, srf_contour, srf_noise, spin_surface.
"""

import numpy as np
from scipy.spatial import cKDTree

from ..statistics.utils import loader as _loader
from .core import smooth_surface

__all__ = ["srf_dilate_mask", "srf_contour", "srf_noise", "spin_surface"]


def srf_dilate_mask(srf, mask, dilation):
    """Dilate (``dilation>0``) or erode (``dilation<0``) a mask on a surface.

    Dilation by ``k`` is ``k`` rounds of edge-averaging thresholded at ``>0``;
    erosion dilates the complement.

    Parameters
    ----------
    srf : dict
        Surface.
    mask : numpy.ndarray
        Per-vertex binary mask.
    dilation : int
        Amount to dilate (positive) or erode (negative).

    Returns
    -------
    numpy.ndarray
        Dilated/eroded mask (boolean for dilation; matches MATLAB for erosion).
    """
    if dilation == 0:
        return mask
    if dilation > 0:
        m = np.asarray(mask, dtype=float)
        return smooth_surface(srf, m, 0, "ones", dilation) > 0
    outside_mask = 1 - np.asarray(mask, dtype=float)
    outside_dilated = srf_dilate_mask(srf, outside_mask, -dilation)
    return 1 - outside_dilated


def srf_contour(srf, mask, adjacent_set_mask=None):
    """Inner and outer one-ring contours of a mask on a surface.

    Parameters
    ----------
    srf : dict
        Surface (or bilateral dict).
    mask : numpy.ndarray or dict
        Per-vertex binary mask.
    adjacent_set_mask : numpy.ndarray, optional
        If given, restrict the contours to the adjacent set.

    Returns
    -------
    inner_contour, outer_contour : numpy.ndarray or dict
    """
    if "lh" in srf and "rh" in srf:
        inner = {}
        outer = {}
        for hemi in ("lh", "rh"):
            inner[hemi], outer[hemi] = srf_contour(srf[hemi], mask[hemi])
        return inner, outer

    mask_f = np.asarray(mask, dtype=float)
    dilated_mask = srf_dilate_mask(srf, mask, 1).astype(float)
    shrunk_mask = np.asarray(srf_dilate_mask(srf, mask, -1), dtype=float)
    inner_contour = mask_f - shrunk_mask
    outer_contour = dilated_mask - mask_f
    if adjacent_set_mask is not None:
        adj_dilated = srf_dilate_mask(srf, adjacent_set_mask, 1).astype(float)
        inner_contour = inner_contour * adj_dilated
        outer_contour = outer_contour * np.asarray(adjacent_set_mask, dtype=float)
    return inner_contour, outer_contour


def srf_noise(srf, FWHM=0, nsubj=1, metric="ones", mask=None, rng=None):
    """Generate (optionally smoothed, masked) Gaussian noise on a surface.

    Parameters
    ----------
    srf : dict
        Surface (or bilateral dict).
    FWHM : float, optional
        Smoothing FWHM. ``0`` (default) leaves the noise unsmoothed.
    nsubj : int, optional
        Number of subjects (columns). Default ``1``.
    metric : {'ones', 'dist'}, optional
        Smoothing metric.
    mask : numpy.ndarray or dict, optional
        Per-vertex mask; if given, noise is masked (and re-normalised after
        smoothing).
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    numpy.ndarray or dict
        ``(nvertices, nsubj)`` noise (or bilateral dict).
    """
    if rng is None:
        rng = np.random.default_rng()

    if "lh" in srf or "rh" in srf:
        out = {}
        for hemi in ("lh", "rh"):
            if hemi in srf:
                hemi_mask = mask[hemi] if mask is not None else None
                out[hemi] = srf_noise(srf[hemi], FWHM, nsubj, metric, hemi_mask, rng)
        return out

    data = rng.standard_normal((srf["nvertices"], nsubj))
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=float)
        data = data * mask_arr[:, None] if data.ndim == 2 else data * mask_arr
    if FWHM > 0:
        data = smooth_surface(srf, data.T, FWHM, metric).T if data.ndim == 2 else \
            smooth_surface(srf, data, FWHM, metric)
        if mask is not None:
            smoothed_mask = smooth_surface(srf, np.asarray(mask, dtype=float), FWHM, metric)
            data = data / (smoothed_mask[:, None] if data.ndim == 2 else smoothed_mask)
    return data


def spin_surface(surface_data, sphere, nperm=1000, include_orig=True, show_loader=True, rng=None):
    """Spin test: random rotations of spherical surface data (Alexander-Bloch).

    Parameters
    ----------
    surface_data : dict
        ``{'lh': data_l, 'rh': data_r}`` per-vertex data.
    sphere : dict
        Bilateral sphere surfaces ``{'lh': srf_l, 'rh': srf_r}``.
    nperm : int, optional
        Number of permutations (rotations), including the original. Default 1000.
    include_orig : bool, optional
        Include the unrotated data as the first column. Default ``True``.
    show_loader : bool, optional
        Print progress. Default ``True``.
    rng : numpy.random.Generator, optional
        Random generator.

    Returns
    -------
    left_rotations, right_rotations : numpy.ndarray
        ``(nvertices, nrot)`` rotated data per hemisphere.
    """
    if rng is None:
        rng = np.random.default_rng()

    vertices_left = np.asarray(sphere["lh"]["vertices"], dtype=float)
    vertices_right = np.asarray(sphere["rh"]["vertices"], dtype=float)

    if include_orig:
        left_cols = [np.asarray(surface_data["lh"], dtype=float)]
        right_cols = [np.asarray(surface_data["rh"], dtype=float)]
    else:
        left_cols = []
        right_cols = []

    reflection = np.eye(3)
    reflection[0, 0] = -1
    bl = vertices_left.copy()
    br = vertices_right.copy()
    tree_left = cKDTree(vertices_left)  # query target stays fixed

    for j in range(1, nperm):
        if show_loader:
            _loader(j, nperm - 1, "spin_surface progress:")
        A = rng.standard_normal((3, 3))
        TL, temp = np.linalg.qr(A)
        TL = TL @ np.diag(np.sign(np.diag(temp)))
        if np.linalg.det(TL) < 0:
            TL[:, 0] = -TL[:, 0]
        TR = reflection @ TL @ reflection
        bl = bl @ TL
        br = br @ TR
        # nearest rotated vertex for each original (matches knnsearch(bl, vertices_left))
        rotation_idx = cKDTree(bl).query(vertices_left)[1]
        left_cols.append(np.asarray(surface_data["lh"])[rotation_idx])
        right_cols.append(np.asarray(surface_data["rh"])[rotation_idx])

    left_rotations = np.column_stack(left_cols) if left_cols else np.empty((vertices_left.shape[0], 0))
    right_rotations = np.column_stack(right_cols) if right_cols else np.empty((vertices_right.shape[0], 0))
    return left_rotations, right_rotations
