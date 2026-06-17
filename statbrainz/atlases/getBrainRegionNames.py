"""getBrainRegionNames (mirrors StatBrainz/Atlases/getBrainRegionNames.m)."""

import re

_LABEL_RE = re.compile(r"<label.*?>(.*)</label>")

__all__ = ['getBrainRegionNames']


def getBrainRegionNames(xml_file):
    """Extract region names from an FSL-style atlas XML file.

    Parameters
    ----------
    xml_file : str
        Path to the atlas ``.xml`` file.

    Returns
    -------
    list of str
        Region names in label order.
    """
    names = []
    with open(xml_file, "r", encoding="ISO-8859-1") as fh:
        for line in fh:
            m = _LABEL_RE.search(line)
            if m:
                names.append(m.group(1))
    return names
