"""imgload (mirrors StatBrainz/ImageViewing/imgload.m)."""

import os

import numpy as np
import nibabel as nib

__all__ = ['imgload', 'data_dir']


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
