"""StatBrainz viewing functions (ported from MATLAB).

Two layers:

* :mod:`statbrainz.viewing.helpers` — pure array builders (RGB images, montages,
  colormaps). No plotting dependency; faithful 1:1 ports.
* :mod:`statbrainz.viewing.display` — matplotlib rewrites of the rendering
  functions. NOT 1:1 (MATLAB figure/patch/GUI code has no faithful analog).
  ``matplotlib`` is an optional dependency (install the ``viz`` extra).

Interactive MATLAB GUIs and figure-window shaping helpers (brainmove, pan3,
sliderGUI, fullscreen, spherescreen, ...) are intentionally not ported.
"""

from .helpers import (
    colorRegion,
    peak2circle,
    viewthresh_image,
    combine_brains,
    custom_colormap,
)
from .display import viewthresh, viewdata, viewbrain, overlay_brain

__all__ = [
    "colorRegion",
    "peak2circle",
    "viewthresh_image",
    "combine_brains",
    "custom_colormap",
    "viewthresh",
    "viewdata",
    "viewbrain",
    "overlay_brain",
]
