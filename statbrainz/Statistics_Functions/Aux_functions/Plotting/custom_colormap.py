"""custom_colormap (mirrors StatBrainz/Statistics_Functions/Aux_functions/Plotting/custom_colormap.m)."""

import numpy as np

__all__ = ['custom_colormap']


def custom_colormap(color1, color2, num_colors=64):
    """Linear RGB colormap interpolating ``color1`` -> ``color2``.

    Parameters
    ----------
    color1, color2 : sequence of float
        Endpoint RGB triples.
    num_colors : int, optional
        Number of colours. Default ``64``.

    Returns
    -------
    numpy.ndarray
        ``(num_colors, 3)`` colormap.
    """
    color1 = np.asarray(color1, dtype=float)
    color2 = np.asarray(color2, dtype=float)
    return np.column_stack(
        [np.linspace(color1[k], color2[k], num_colors) for k in range(3)]
    )
