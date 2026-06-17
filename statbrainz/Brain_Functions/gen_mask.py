"""gen_mask (mirrors StatBrainz/Brain_Functions/gen_mask.m)."""

import numpy as np

from statbrainz.ImageViewing.imgload import imgload

_STDSIZE = (91, 109, 91)

__all__ = ['gen_mask']


def gen_mask(data, use_MNI=True, make3D=True, stdsize=_STDSIZE):
    """Intersection mask of per-subject masks (optionally with the MNI mask).

    Parameters
    ----------
    data : numpy.ndarray
        ``nsubj x nvox`` matrix of per-subject masks (rows).
    use_MNI : bool, optional
        Start from the MNI mask. Default ``True``.
    make3D : bool, optional
        Reshape the result to ``(91, 109, 91)``. Default ``True``.
    stdsize : sequence of int, optional
        Standard volume size used when ``use_MNI`` is ``False`` and for the 3D
        reshape. Default ``(91, 109, 91)``.

    Returns
    -------
    numpy.ndarray
        The intersection mask (3D if ``make3D``, else a vector).
    """
    data = np.asarray(data, dtype=float)
    if use_MNI:
        mask = imgload("MNImask").astype(float).ravel(order="F")
    else:
        mask = np.ones(int(np.prod(stdsize)))

    nsubj = data.shape[0]
    for subj in range(nsubj):
        mask = mask * data[subj, :]
    if make3D:
        mask = mask.reshape((91, 109, 91), order="F")
    return mask
