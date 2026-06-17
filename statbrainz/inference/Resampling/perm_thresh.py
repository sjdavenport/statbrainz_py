"""perm_thresh (mirrors StatBrainz/Inference/Resampling/perm_thresh.m)."""

import numpy as np

from statbrainz.Statistics_Functions.ImageOperations.fast_conv import fast_conv
from statbrainz.Statistics_Functions.Stats_functions.mvtstat import mvtstat
from statbrainz.Statistics_Functions.Aux_functions.SystemFunctions.loader import loader as _loader

__all__ = ['perm_thresh']


def perm_thresh(
    data,
    stat="T",
    twosample=False,
    stepdown=0,
    alpha=0.05,
    FWHM=np.nan,
    mask=None,
    demean=False,
    niters=1000,
    include_original=True,
    show_loader=True,
    rng=None,
):
    """Sign-flip permutation threshold for a one-sample statistic image.

    Parameters
    ----------
    data : numpy.ndarray
        ``[spatial_dims..., nsubj]`` array.
    stat : {'T', 'Z'}, optional
        ``'T'`` uses the t-statistic (:func:`mvtstat`), ``'Z'`` the mean.
        Default ``'T'``.
    twosample : bool, optional
        Use the maximum of the absolute statistic. Default ``False``.
    stepdown : int, optional
        Step-down depth (1-4). Default ``0`` (no step-down).
    alpha : float, optional
        Significance level. Default ``0.05``.
    FWHM : float, optional
        Pre-smoothing FWHM. ``nan`` (default) skips smoothing.
    mask : None
        NOT supported (the MATLAB mask path needs the missing ``lmindices``).
    demean : bool, optional
        Demean the data before permuting. Default ``False``.
    niters : int, optional
        Number of permutations. Default ``1000``.
    include_original : bool, optional
        Include the observed statistic as the first permutation. Default
        ``True``.
    show_loader : bool, optional
        Print progress. Default ``True``.
    rng : numpy.random.Generator, optional
        Random generator. Default ``np.random.default_rng()``.

    Returns
    -------
    im_perm : numpy.ndarray
        Boolean image of voxels exceeding the threshold.
    threshold : float
        The ``100*(1-alpha)`` percentile of the permutation maxima.
    vec_of_maxima : numpy.ndarray
        The permutation maxima.
    """
    if mask is not None:
        raise NotImplementedError(
            "perm_thresh with a mask requires lmindices, which is missing from "
            "the StatBrainz MATLAB source; only the global-maximum path is ported."
        )
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=float)
    sD = data.shape
    nsubj = sD[-1]
    D = data.ndim - 1

    if not np.isnan(FWHM):
        data, _ = fast_conv(data, FWHM, D)

    def _stat(arr):
        if stat == "Z":
            return np.mean(arr, axis=-1)
        return mvtstat(arr)[0]

    stat_image = _stat(data)

    if demean:
        data = data - np.mean(data, axis=-1, keepdims=True)

    vec_of_maxima = np.zeros(niters)
    if include_original:
        start = 1
        combined_orig = _stat(data)
        vec_of_maxima[0] = combined_orig.max()
    else:
        start = 0

    random_berns = 2 * (rng.binomial(1, 0.5, (nsubj, niters)) - 0.5)
    for it in range(start, niters):
        if show_loader:
            _loader(it - start + 1, niters - start, "Progress:")
        flips = random_berns[:, it]
        data2 = data.copy()
        neg = flips < 0
        data2[..., neg] = -data[..., neg]
        combined = _stat(data2)
        if twosample:
            vec_of_maxima[it] = np.max(np.abs(combined))
        else:
            vec_of_maxima[it] = combined.max()

    threshold = np.percentile(vec_of_maxima, 100 * (1 - alpha))
    if twosample:
        im_perm = np.abs(stat_image) > threshold
    else:
        im_perm = stat_image > threshold

    if 0 < stepdown < 5:
        sub = data[im_perm, :]
        im_perm, threshold, vec_of_maxima = perm_thresh(
            sub, stat, twosample, stepdown + 1, alpha, np.nan, None,
            demean, niters, include_original, show_loader, rng,
        )
    return im_perm, threshold, vec_of_maxima
