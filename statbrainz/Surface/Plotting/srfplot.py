"""srfplot (mirrors StatBrainz/Surface/Plotting/srfplot.m)."""

import numpy as np

__all__ = ['srfplot']

# Named views -> matplotlib (azim, elev), mirroring srfplot.m's view(view_vec).
# MATLAB view([az, el]) uses azimuth measured from -y; matplotlib azim is
# measured from +x, so az_mpl = az_matlab - 90.
_NAMED_VIEWS = {
    "top": (0, 90),
    "back": (0, 0),
    "front": (-180, 0),
    "bottom": (-180, -90),
    "left": (-90, 0),
    "right": (270, 0),
}


def _resolve_view(view_vec, srf):
    """Resolve a view spec (string / 0 / 1 / [az, el]) to matplotlib (azim, elev)."""
    hemi = srf.get("hemi")
    if hemi == "rh":
        default_side, default_backside = (90, 0), (270, 0)
    else:
        default_side, default_backside = (270, 0), (90, 0)

    if view_vec is None:
        view_vec = "side" if hemi is not None else "top"

    if hemi is not None:
        if view_vec == "side" or view_vec == 0:
            view_vec = default_side
        elif view_vec == "backside" or view_vec == 1:
            view_vec = default_backside

    if view_vec == 0:
        view_vec = "top"
    if isinstance(view_vec, str):
        view_vec = _NAMED_VIEWS[view_vec]

    az_matlab, el = view_vec
    return az_matlab - 90, el


def srfplot(srf, surface_data=None, view_vec=None, edgealpha=None, ax=None):
    """Plot per-vertex surface data on a surface mesh.

    Parameters
    ----------
    srf : dict
        Surface structure with ``faces``/``vertices`` (or bilateral ``lh``/``rh``).
    surface_data : numpy.ndarray or dict, optional
        Per-vertex data: either an ``nvertices x 3`` RGB colour map or a length
        ``nvertices`` scalar field. For a bilateral surface, a dict with
        ``lh``/``rh``.
    view_vec : str, int or sequence, optional
        View specification, mirroring ``srfplot.m`` (``'side'``, ``'top'``,
        ``'left'``, ... or ``[azimuth, elevation]``). Defaults to ``'side'`` for
        a single hemisphere and ``'top'`` for a joint surface.
    edgealpha : float, optional
        Edge transparency. Defaults to ``0.05`` with data, ``0.2`` without.
    ax : matplotlib 3D axes, optional
        Axis to draw on. A new figure/axis is created if omitted.

    Returns
    -------
    matplotlib 3D axes
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Bilateral: build a joint surface (offset rh faces) and recurse.
    if "lh" in srf and "rh" in srf:
        lh, rh = srf["lh"], srf["rh"]
        jointsrf = {
            "vertices": np.vstack([lh["vertices"], rh["vertices"]]),
            "faces": np.vstack([lh["faces"], rh["faces"] + lh["nvertices"]]),
        }
        jointsrf["nvertices"] = jointsrf["vertices"].shape[0]
        jointsrf["nfaces"] = jointsrf["faces"].shape[0]
        if surface_data is not None:
            data_lh = np.asarray(surface_data["lh"])
            data_rh = np.asarray(surface_data["rh"])
            surface_data = np.concatenate([data_lh, data_rh], axis=0)
        return srfplot(jointsrf, surface_data, view_vec, edgealpha, ax)

    if edgealpha is None:
        edgealpha = 0.2 if surface_data is None else 0.05

    azim, elev = _resolve_view(view_vec, srf)

    vertices = np.asarray(srf["vertices"], dtype=float)
    faces = np.asarray(srf["faces"], dtype=np.int64)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("black")
        if hasattr(ax.figure, "set_facecolor"):
            ax.figure.set_facecolor("black")

    tris = vertices[faces]

    if surface_data is None:
        collection = Poly3DCollection(
            tris, facecolors="none", edgecolors=(1, 1, 1, edgealpha), linewidths=0.2
        )
    else:
        surface_data = np.asarray(surface_data, dtype=float)
        if surface_data.ndim == 2 and surface_data.shape[1] == 3:
            rgb = surface_data  # already an RGB colour map
        else:
            # Scalar field: map to grayscale-like RGB via min-max normalisation.
            v = surface_data.ravel()
            lo, hi = np.nanmin(v), np.nanmax(v)
            norm = np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)
            rgb = np.repeat(norm[:, None], 3, axis=1)
        # Per-face colour = mean of its vertices' colours (flat shading).
        face_colors = rgb[faces].mean(axis=1)
        face_colors = np.clip(face_colors, 0, 1)
        collection = Poly3DCollection(
            tris, facecolors=face_colors, edgecolors=(0, 0, 0, edgealpha),
            linewidths=0.1,
        )

    ax.add_collection3d(collection)

    X, Y, Z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    ax.set_xlim(X.min() - 1, X.max() + 1)
    ax.set_ylim(Y.min() - 1, Y.max() + 1)
    ax.set_zlim(Z.min() - 1, Z.max() + 1)
    ax.set_box_aspect(
        (X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min())
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return ax
