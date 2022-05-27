"""
This module deals with the pre-processing of HFI data from Planck maps.
These functions transform all-sky maps into HEALPix tiles, according to which the data is split
into training, validation, and test. After that, the samples themselves are generated.

Planck data can be accessed `here
<https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/>`_.
"""
import numpy as np
import os
from typing import List, Dict
from astropy.io import fits


def extract_data_key(path: str, key: str, idx: int = 1) -> np.ndarray:
    """From fits file extract data by key.

    :param path: Input path.
    :type path: str
    :param key: Keyword to exract.
    :type key: str
    :rtype: np.ndarray
    """
    with fits.open(path) as hdul:
        return hdul[idx].data[key]


def match_channels(indir: str, channels: List[int]) -> Dict[int, str]:
    """For each channel find corresponding file

    :param indir: Input dir.
    :type indir: str
    :param channels: channels.
    :type channels: List[int]
    :rtype: Dict[int, str]
    """
    files = sorted(os.listdir(indir))
    files_by_ch = {}
    for channel in channels:
        file = list(filter(lambda x: str(channel) in x, files))[0]
        files_by_ch[channel] = os.path.join(indir, file)
    return files_by_ch
