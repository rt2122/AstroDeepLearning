from astropy.io import fits
from astropy.table import Table
import numpy as np
import pandas as pd
from typing import Union


def fits2df(fitspath: str, one_col: str = None) -> Union[pd.DataFrame, np.ndarray]:
    """Extract pandas dataframe from fits file.

    :param fitspath:
    :type fitspath: str
    :param one_col:
    :type one_col: str
    :rtype: Union[pd.DataFrame, np.ndarray]
    """

    df = None
    with fits.open(fitspath) as hdul:
        if not (one_col is None):
            return np.array(hdul[1].data[one_col])
        tbl = Table(hdul[1].data)
        names = [name for name in tbl.colnames if len(tbl[name].shape) <= 1]
        df = tbl[names].to_pandas()
    return df
