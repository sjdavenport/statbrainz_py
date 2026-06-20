"""srf_colour (mirrors StatBrainz/Surface/Plotting/srf_colour.m)."""

import numpy as np

__all__ = ['srf_colour']


def srf_colour(srf, region_masks, colours, background_gradient=0.7):
    """Build an RGB colour map for a surface from region masks.

    Parameters
    ----------
    srf : dict
        Surface structure (or bilateral dict with ``lh``/``rh``).
    region_masks : sequence
        Per-region binary masks. For a bilateral surface each entry is a dict
        with ``lh``/``rh`` fields.
    colours : sequence
        ``[R, G, B]`` colour vector per region.
    background_gradient : float, optional
        Brightness for unmasked vertices. Default ``0.7``.

    Returns
    -------
    numpy.ndarray or dict
        ``nvertices x 3`` RGB matrix (or bilateral dict).
    """
    if "lh" in srf and "rh" in srf:
        masks_lh = [m["lh"] for m in region_masks]
        masks_rh = [m["rh"] for m in region_masks]
        return {
            "lh": srf_colour(srf["lh"], masks_lh, colours, background_gradient),
            "rh": srf_colour(srf["rh"], masks_rh, colours, background_gradient),
        }

    nvertices = np.asarray(region_masks[0]).shape[0]
    color_map = np.ones((nvertices, 3)) * background_gradient
    for region_mask, colour in zip(region_masks, colours):
        mask = np.asarray(region_mask).astype(bool).ravel()
        color_map[mask, 0] = colour[0]
        color_map[mask, 1] = colour[1]
        color_map[mask, 2] = colour[2]
    return color_map
