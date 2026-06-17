"""filesindir (mirrors StatBrainz/Statistics_Functions/Aux_functions/SystemFunctions/filesindir.m)."""

import os

__all__ = ['filesindir']


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
