"""spin_surface (mirrors StatBrainz/Surface/spin_surface.m)."""

import numpy as np
from scipy.spatial import cKDTree

from statbrainz.Statistics_Functions.Aux_functions.SystemFunctions.loader import loader as _loader

__all__ = ['spin_surface']


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
