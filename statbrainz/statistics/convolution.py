"""Separable Gaussian smoothing ported from StatBrainz fast_conv.

Faithful port of: fast_conv (1D/2D/3D), including the kernel sum-of-squares
``ss`` used for variance normalisation.

The MATLAB code smooths with a truncated Gaussian kernel evaluated by ``Gker``
and convolves with ``conv``/``convn`` in ``'same'`` mode. We reproduce that with
``scipy.signal.convolve(..., mode='same')`` applied separably per axis.
"""

import numpy as np
from scipy.signal import convolve

from .transforms import fwhm2sigma
from .kernels import Gker

__all__ = ["fast_conv"]


def _kernel(fwhm_d, trunc_lo, trunc_hi, dx_d, adjust_d):
    pts = np.arange(-trunc_lo, trunc_hi + dx_d / 2, dx_d) + adjust_d
    val, _, _ = Gker(pts, fwhm_d)
    return val


def fast_conv(data, FWHM, D=None, truncation=None, dx=1, adjust_kernel=None):
    """Smooth ``data`` with a separable truncated Gaussian kernel.

    Parameters
    ----------
    data : numpy.ndarray
        1D, 2D, or 3D data (optionally with a trailing subject axis when
        ``D`` is given and less than ``data.ndim``).
    FWHM : float or sequence of float
        Smoothing FWHM (scalar applies to every dimension). ``0`` returns the
        data unchanged.
    D : int, optional
        Number of spatial dimensions. Inferred from ``data`` if omitted.
    truncation : int or array_like, optional
        Half-width(s) at which the kernel is truncated. Default
        ``ceil(fwhm2sigma(FWHM) * 4)``.
    dx : float or sequence of float, optional
        Sampling spacing. Default ``1``.
    adjust_kernel : sequence of float, optional
        Per-dimension shift of the kernel centre. Default zeros.

    Returns
    -------
    smoothed_data : numpy.ndarray
        Smoothed data, same shape as ``data``.
    ss : float
        Sum of squares of the (separable) kernel over the truncation box,
        scaled by the voxel volume ``prod(dx)``.
    """
    data = np.asarray(data, dtype=float)
    FWHM = np.atleast_1d(np.asarray(FWHM, dtype=float))
    if np.all(FWHM == 0):
        return data, 1.0

    if D is None:
        s = data.shape
        D = data.ndim
        if D == 2 and (s[0] == 1 or s[1] == 1):
            D = 1
            data = data.ravel()

    if FWHM.size == 1:
        FWHM = np.repeat(FWHM[0], D)
    if D > 3:
        raise ValueError("fast_conv not coded for dimension > 3")

    dx = np.atleast_1d(np.asarray(dx, dtype=float))
    if dx.size == 1:
        dx = np.repeat(dx[0], D)
    if adjust_kernel is None:
        adjust_kernel = np.zeros(D)
    else:
        adjust_kernel = np.asarray(adjust_kernel, dtype=float)

    if truncation is None:
        truncation = np.ceil(fwhm2sigma(FWHM) * 4)
    truncation = np.atleast_1d(np.asarray(truncation, dtype=float))
    if truncation.size == 1:
        truncation = np.tile(truncation[0], (2, D))
    elif truncation.size == D:
        truncation = np.vstack([truncation, truncation])

    # Subject loop: trailing axis beyond the D spatial dims.
    if D < data.ndim:
        out = np.zeros_like(data)
        ss = 1.0
        for j in range(data.shape[-1]):
            tmp, ss = fast_conv(
                data[..., j], FWHM, D, truncation, dx, adjust_kernel
            )
            out[..., j] = tmp
        return out, ss

    kernels = [
        _kernel(FWHM[d], truncation[0, d], truncation[1, d], dx[d], adjust_kernel[d])
        for d in range(D)
    ]

    if D == 1:
        smoothed = convolve(data, kernels[0], mode="same")
        ss = np.sum(kernels[0] ** 2) * dx[0]
        return smoothed, ss

    if D == 2:
        smoothed = convolve(data, kernels[0][:, None], mode="same")
        smoothed = convolve(smoothed, kernels[1][None, :], mode="same")
        outer = np.outer(kernels[0], kernels[1])
        ss = np.sum(outer**2) * dx[0] * dx[1]
        return smoothed, ss

    # D == 3
    smoothed = convolve(data, kernels[0][:, None, None], mode="same")
    smoothed = convolve(smoothed, kernels[1][None, :, None], mode="same")
    smoothed = convolve(smoothed, kernels[2][None, None, :], mode="same")
    prod = (
        kernels[0][:, None, None]
        * kernels[1][None, :, None]
        * kernels[2][None, None, :]
    )
    ss = np.sum(prod**2) * dx[0] * dx[1] * dx[2]
    return smoothed, ss
