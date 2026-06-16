"""StatBrainz: statistical inference and visualization of brain imaging data.

Python port of the MATLAB StatBrainz package. Submodules mirror the MATLAB areas:

- ``statbrainz.statistics`` - statistics functions (kernels, transforms, mvtstat, ...)
- ``statbrainz.brain``      - volumetric brain helpers (masks, vectorization, MNI, ...)
- ``statbrainz.surface``    - surface I/O, smoothing, resampling, spinning
- ``statbrainz.inference``  - cluster inference, CoPE sets, MHT/FDR, TDP bounds
- ``statbrainz.atlases``    - atlas region masks and lookups
- ``statbrainz.viewing``    - volumetric/surface visualization (matplotlib/nilearn)
"""

__version__ = "0.0.1"
