"""_display_helpers (mirrors StatBrainz/ImageViewing/_display_helpers.m)."""

import numpy as np

__all__ = ['_get_ax', '_apply_rotate']


def _get_ax(ax):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def _apply_rotate(arr, rotate):
    if rotate == 2:
        return arr.T
    if rotate == 3:
        return np.flipud(arr)
    if rotate == 4:
        return np.flipud(arr.T)
    return arr
