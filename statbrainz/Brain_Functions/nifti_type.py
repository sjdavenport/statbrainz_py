"""nifti_type (mirrors StatBrainz/Brain_Functions/nifti_type.m)."""

import numpy as np

__all__ = ['nifti_type']


def nifti_type(files):
    """Classify file names as ``.nii.gz`` (1), ``.nii`` (2), or other (0).

    Parameters
    ----------
    files : str or sequence of str
        File name(s) to classify.

    Returns
    -------
    numpy.ndarray
        Integer code per file: 0 = neither, 1 = ``.nii.gz``, 2 = ``.nii``.
    """
    if isinstance(files, str):
        files = [files]
    out = np.zeros(len(files), dtype=int)
    for i, f in enumerate(files):
        if ".nii.gz" in f:
            out[i] = 1
        elif ".nii" in f:
            out[i] = 2
        else:
            out[i] = 0
    return out
