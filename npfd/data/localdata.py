"""
Custom dataset processing/generation functions should be added to this file
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def read_interim_file(filename, has_flag=True, col_names=None, year=None, utc_time=True,
                      resample_freq='10T', keep_valid=True):
    """Read a DMPS inverted .cle file

    Data is multiple space-delimited. The file contains processed particle number size
    distributions with the original time resolution of the instrument.

    Parameters
    ----------
    filename: path-like

    has_flag: boolean
        if true, last column is treated as a flag
    col_names: arry-like, optional
        list of column names to use. Duplicates in this list are not allowed
    resample_freq: str
        frequency to resample the data. Mean is used
    keep_valid: bool
        if true, return only data with flag 0 and replace the rest with nan's

    Returns
    -------
    data: DataFrame
    """

    # Infere year from filename
    # year = int(filename.stem[2:6])
    if year is None:
        raise Exception('Year is needed!')
    else:
        year = int(year)

    df = pd.read_csv(filename, delim_whitespace=True)

    if col_names is None:
        # Rename columns
        df.columns.values[0] = 'datetime'
        df.columns.values[1] = 'tot_conc'

        if has_flag:
            df.columns.values[-1] = 'flag'
    else:
        df.columns = col_names

    # Convert decimal day of year to datetime column to dataframe index
    df['datetime'] = df['datetime'].apply(lambda ddoy: datetime(year - 1, 12, 31) + timedelta(days=ddoy))
    df.set_index('datetime', inplace=True)

    if resample_freq is not None:
        df = df.resample(resample_freq).mean()

    df.drop('tot_conc', axis=1, inplace=True)

    flags = df['flag']

    if keep_valid:
        df.loc[flags == 1] = np.nan
        df.loc[flags == 2] = np.nan
        df.loc[flags == 3] = np.nan
        df.loc[flags == 4] = np.nan

    df.drop('flag', axis=1, inplace=True)

    df.columns = [float(i) * 1e9 for i in df.columns]

    if not utc_time:
        df.index = df.index - timedelta(hours=3)
        # df.index = df.index.tz_localize(tz='UTC').tz_convert(tz='America/Argentina/Buenos_Aires')
    return df, flags
