"""statbrainz_maindir (mirrors StatBrainz/Statistics_Functions/Aux_functions/SystemFunctions/statbrainz_maindir.m)."""

import os

__all__ = ['statbrainz_maindir']


def statbrainz_maindir():
    """Return the StatBrainz package root directory (with trailing separator).

    The MATLAB version returns the package install directory; here we return the
    ``statbrainz`` package directory, under which bundled data would live.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pkg_dir + os.sep
