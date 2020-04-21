"""Input-output module for files with measurements

Author
------
Jenni Rinker
rink@dtu.dk
"""
import os

import numpy as np
import pandas as pd

from pyconturb.core.helpers import rotate_time_series

# hard-code parameters from .set file format
_v52_mast_hts = np.array([18, 31, 44, 57, 70])  # msmt hts for risø v52 mast


def v52_pickle_to_condata(path):
    """Convert pickled dataframe from V52 met mast to condata for gen_turb

    Arguments
    ---------
    path : str
        Path to pickled dataframe

    Returns
    -------
    con_data : dict
        Collection of 'con_spat_df', which has the spatial information of the
        constraining points, and 'con_turb_df', which has the constraining
        time series.
    """

    # spatial df
    con_spat_df = pd.DataFrame(columns=['k', 'p_id', 'x', 'y', 'z'],
                               index=range(len(_v52_mast_hts) * 3))
    for i_ht, ht in enumerate(_v52_mast_hts):
        con_spat_df.loc[i_ht*3, :] = ['vxt', f'p{i_ht}', 0, 0, ht]
        con_spat_df.loc[i_ht*3 + 1, :] = ['vyt', f'p{i_ht}', 0, 0, ht]
        con_spat_df.loc[i_ht*3 + 2, :] = ['vzt', f'p{i_ht}', 0, 0, ht]

    # load raw data from pickle
    time_df = pd.read_pickle(path)
    time = np.arange(time_df.shape[0]) * 600. / time_df.shape[0]

    # conturb df
    conturb_df = pd.DataFrame(index=time,
                              columns=con_spat_df.k + '_' + con_spat_df.p_id)
    for i_ht, ht in enumerate(_v52_mast_hts):
        x, y, z = [time_df[f'{c}_{ht:.0f}m'] for c in ['X', 'Y', 'Z']]
        u, v, w = rotate_time_series(x.values, y.values, z.values)
        conturb_df[f'vxt_p{i_ht}'] = -u
        conturb_df[f'vyt_p{i_ht}'] = -v
        conturb_df[f'vzt_p{i_ht}'] = w

    # assemble both to dictionary
    con_data = {'con_spat_df': con_spat_df,
                'con_turb_df': conturb_df}

    return con_data
