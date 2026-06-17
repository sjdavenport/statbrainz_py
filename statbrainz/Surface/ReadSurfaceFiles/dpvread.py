"""dpvread (mirrors StatBrainz/Surface/ReadSurfaceFiles/dpvread.m)."""

import numpy as np

__all__ = ['dpvread']


def dpvread(filename, nvertices):
    """Read a FreeSurfer data-per-vertex (.dpv) text file.

    Parameters
    ----------
    filename : str
        Path to the whitespace-delimited file (5 columns per vertex).
    nvertices : int
        Number of vertices.

    Returns
    -------
    numpy.ndarray
        ``(nvertices, 5)`` array.
    """
    vals = np.fromfile(filename, sep=" ", count=5 * nvertices)
    return vals.reshape((nvertices, 5))
