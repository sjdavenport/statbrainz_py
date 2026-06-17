"""index2mask (mirrors StatBrainz/Inference/ClusterInference/Clusterextent/index2mask.m)."""

import numpy as np

__all__ = ['index2mask']


def index2mask(indices, dim=(91, 109, 91)):
    """Set 1-based column-major flat ``indices`` to 1 in a zero image of ``dim``."""
    dim = tuple(int(d) for d in dim)
    flat = np.zeros(int(np.prod(dim)))
    indices = np.atleast_1d(np.asarray(indices, dtype=int))
    flat[indices - 1] = 1
    return flat.reshape(dim, order="F")
