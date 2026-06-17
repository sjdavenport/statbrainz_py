"""imgsave (mirrors StatBrainz/ImageViewing/imgsave.m)."""

import os

import numpy as np
import nibabel as nib

__all__ = ['imgsave']


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
