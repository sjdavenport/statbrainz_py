"""loader (mirrors StatBrainz/Statistics_Functions/Aux_functions/SystemFunctions/loader.m)."""

import sys

__all__ = ['loader']


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
