"""Volumetric image I/O ported from StatBrainz (imgload/imgsave).

``imgload`` resolves a bare name (e.g. ``'MNImask'``) to a bundled NIfTI file
under ``statbrainz/data/Volume/`` and loads it with nibabel, or loads an explicit
path. Like the MATLAB version, a 0/1 image is returned as a boolean mask.

``imgsave`` is a clean nibabel-based writer. The MATLAB original is tied to SPM
and a set of global path variables; here it simply writes ``array`` to a NIfTI
file at an explicit path (this is an intentional, documented divergence).
"""

import os

import numpy as np
import nibabel as nib

__all__ = ["imgload", "imgsave", "data_dir"]


def data_dir():
    """Return the bundled data directory (``statbrainz/data``)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _volume_dir():
    return os.path.join(data_dir(), "Volume")


def imgload(filename):
    """Load a volumetric image by bundled name or explicit path.

    Parameters
    ----------
    filename : str
        Either a bare name resolved against the bundled ``Volume/`` directory
        (e.g. ``'MNImask'`` -> ``MNImask.nii`` / ``.nii.gz``), or a path to a
        ``.nii``/``.nii.gz`` file.

    Returns
    -------
    numpy.ndarray
        The image as a float array, or a boolean mask if the image is 0/1.
    """
    vol = _volume_dir()
    candidates = []
    if filename.endswith(".gz") or filename.endswith(".nii"):
        candidates = [os.path.join(vol, filename), filename]
    else:
        candidates = [
            os.path.join(vol, filename + ".nii"),
            os.path.join(vol, filename + ".nii.gz"),
            os.path.join(vol, filename),
            filename,
        ]

    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(f"This file is not available: {filename!r}")

    img = np.asarray(nib.load(path).dataobj).astype(float)
    uniq = np.unique(img)
    if uniq.size == 2 and uniq[0] == 0 and uniq[1] == 1:
        img = img > 0
    return img


def imgsave(array, filename, directory=".", header=None, affine=None):
    """Write ``array`` to a NIfTI file (nibabel-based; diverges from SPM original).

    Parameters
    ----------
    array : numpy.ndarray
        Image data. A length-902629 vector is reshaped to ``(91, 109, 91)``.
    filename : str
        Output file stem (``.nii`` is appended if no extension is present).
    directory : str, optional
        Output directory. Default current directory.
    header : nibabel.Nifti1Header, optional
        Header to use. Default ``None`` (a fresh header).
    affine : numpy.ndarray, optional
        4x4 affine. Default identity.

    Returns
    -------
    str
        The path written.
    """
    array = np.asarray(array)
    if array.size == 902629:
        array = array.reshape((91, 109, 91), order="F")
    if affine is None:
        affine = np.eye(4)
    if not os.path.isdir(directory):
        raise FileNotFoundError("This directory does not exist")
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        filename = filename + ".nii"
    path = os.path.join(directory, filename)
    nib.save(nib.Nifti1Image(array, affine, header), path)
    return path
