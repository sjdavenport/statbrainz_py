"""modul (mirrors StatBrainz/Statistics_Functions/Aux_functions/SystemFunctions/modul.m)."""

__all__ = ['modul']


def modul(I, modval):
    """Print ``I`` whenever it is divisible by ``modval``."""
    if I % modval == 0:
        print(I)
