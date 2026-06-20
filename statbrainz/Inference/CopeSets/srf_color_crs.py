"""srf_color_crs (mirrors StatBrainz/Inference/CopeSets/srf_color_crs.m)."""

from statbrainz.Surface.Plotting.srf_colour import srf_colour
from .srf_scb2cope import srf_scb2cope

__all__ = ['srf_color_crs']


def srf_color_crs(srf, lower_band, upper_band, xbar, c, posneg=1,
                  colorscheme="bowring", usecontour=1):
    """Colour map for surface cope-set display.

    Parameters
    ----------
    srf : dict
        Single-hemisphere surface structure.
    lower_band, upper_band : numpy.ndarray
        Lower/upper confidence bands over vertices.
    xbar : numpy.ndarray
        Estimated mean field over vertices.
    c : float
        Threshold at which to generate the cope set.
    posneg : {1, -1}, optional
        Direction. Default ``1``.
    colorscheme : {'bowring', 'blue', 'red'}, optional
        Colour scheme. Default ``'bowring'``.
    usecontour : int, optional
        If truthy, use the contour set rather than the yellow set. Default ``1``.

    Returns
    -------
    numpy.ndarray
        ``nvertices x 3`` RGB colour map.
    """
    if posneg == 1:
        lower_set, upper_set, contour, yellow_set = srf_scb2cope(
            srf, lower_band, upper_band, xbar, c
        )
    elif posneg == -1:
        upper_set, lower_set, contour, yellow_set = srf_scb2cope(
            srf, -lower_band, -upper_band, -xbar, c
        )
    else:
        raise ValueError("posneg must be 1 or -1")

    if usecontour:
        sets = [lower_set, contour, upper_set]
    else:
        sets = [lower_set, yellow_set, upper_set]

    if colorscheme == "blue":
        colours = [[0.5, 0.5, 1], [0, 1, 1], [0, 0, 0.9]]
    elif colorscheme == "red":
        colours = [[0.5, 0.5, 1], [0, 1, 1], [0, 0, 0.9]]
    else:  # 'bowring'
        colours = [[0, 0, 1], [1, 1, 0], [1, 0, 0]]

    return srf_colour(srf, sets, colours)
