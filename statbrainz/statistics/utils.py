"""System / progress utilities ported from StatBrainz Aux_functions.

Faithful (Pythonic) ports of: loader, modul, filesindir, statbrainz_maindir.
"""

import os
import sys

__all__ = ["loader", "modul", "filesindir", "statbrainz_maindir"]


def loader(I, totalI, message="Percent done:"):
    """Print an in-place percent-done progress indicator.

    Parameters
    ----------
    I : int
        Current iteration (1-based, matching MATLAB).
    totalI : int
        Total number of iterations.
    message : str, optional
        Label printed before the percentage (only on the first call).
    """
    percent_done = 100 * I / totalI
    if I == 1:
        sys.stdout.write("-" * 55 + "\n")
        sys.stdout.write(message + "   ")
    sys.stdout.write("\r" + message + f"   {percent_done:.1f}")
    sys.stdout.flush()
    if I == totalI:
        sys.stdout.write("\n")
        sys.stdout.flush()


def modul(I, modval):
    """Print ``I`` whenever it is divisible by ``modval``."""
    if I % modval == 0:
        print(I)


def filesindir(directory="./", pattern=None, hiddenfiles=False):
    """List the files in ``directory``, optionally filtered by substring.

    Parameters
    ----------
    directory : str, optional
        Directory to list. Default current directory.
    pattern : str or sequence of str or None, optional
        If given, only files whose name contains the pattern (any of the
        patterns, if a list) are returned.
    hiddenfiles : bool, optional
        Include dotfiles. Default ``False``.

    Returns
    -------
    list of str
        Matching file names (not full paths), matching MATLAB ``filesindir``.
    """
    if isinstance(pattern, (list, tuple)):
        out = []
        for p in pattern:
            out.extend(filesindir(directory, p, hiddenfiles))
        return out

    entries = sorted(os.listdir(directory))
    files = [f for f in entries if os.path.isfile(os.path.join(directory, f))]
    if not hiddenfiles:
        files = [f for f in files if not f.startswith(".")]
    if pattern is not None:
        files = [f for f in files if pattern in f]
    return files


def statbrainz_maindir():
    """Return the StatBrainz package root directory (with trailing separator).

    The MATLAB version returns the package install directory; here we return the
    ``statbrainz`` package directory, under which bundled data would live.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return pkg_dir + os.sep
