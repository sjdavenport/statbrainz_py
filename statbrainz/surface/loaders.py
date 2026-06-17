"""Standard-template surface loaders ported from StatBrainz Surface.

Faithful port of: loadsrf.

The MATLAB version loads custom ``srf`` structs from ``.mat`` files under
``BrainImages/Surface/``. Here the same geometry is bundled as compressed
``.npz`` files under ``statbrainz/data/Surface/`` (faces pre-converted to
0-based), and loaded into the standard surface dict used across
:mod:`statbrainz.surface`.

The MATLAB ``'bert'`` branch pointed at a hard-coded local FreeSurfer directory
on the original author's machine and is not portable; pass an explicit path to
:func:`statbrainz.surface.fs2surf` instead for arbitrary subjects.
"""

import os

import numpy as np

from ..brain.io import data_dir
from .core import make_srf

__all__ = ["loadsrf"]

_ACCEPTED_TYPES = ("white", "pial", "sphere", "inflated")
_ACCEPTED_IDS = ("fs3", "fs4", "fs5", "fs6", "fs7")


def _surface_dir():
    return os.path.join(data_dir(), "Surface")


def _load_npz(path):
    with np.load(path) as z:
        lh = make_srf(z["lh_vertices"], z["lh_faces"])
        rh = make_srf(z["rh_vertices"], z["rh_faces"])
    lh["hemi"] = "lh"
    rh["hemi"] = "rh"
    return {"lh": lh, "rh": rh}


def loadsrf(surface_id="fs5", surface_type="white"):
    """Load a standard bilateral template surface.

    Parameters
    ----------
    surface_id : str, optional
        ``'fs3'``..``'fs7'`` (fsaverage densities) or ``'hcp'`` (32k_fs_LR).
        Default ``'fs5'``.
    surface_type : str, optional
        ``'white'``, ``'pial'``, ``'sphere'``, or ``'inflated'``. Default
        ``'white'``. Ignored for ``'hcp'``.

    Returns
    -------
    dict
        Bilateral surface ``{'lh': srf_l, 'rh': srf_r}``, each a surface dict
        with 0-based ``faces``.
    """
    if surface_id == "hcp":
        path = os.path.join(_surface_dir(), "hcp", "hcp_32k.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HCP surface not bundled: {path}")
        return _load_npz(path)

    if surface_id == "bert":
        raise NotImplementedError(
            "The 'bert' branch used a hard-coded local FreeSurfer path; load a "
            "subject surface explicitly with statbrainz.surface.fs2surf instead."
        )

    if surface_type not in _ACCEPTED_TYPES:
        raise ValueError("The supplied surface_type is not available")
    if surface_id not in _ACCEPTED_IDS:
        raise ValueError("The supplied surface_id is not available")

    # 'fs5' -> 'fsaverage5'
    full_id = "fsaverage" + surface_id[2]
    path = os.path.join(_surface_dir(), full_id, f"{surface_type}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Surface not bundled: {path}")
    return _load_npz(path)
