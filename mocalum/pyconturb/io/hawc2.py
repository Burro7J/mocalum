"""Input-output module for files related to HAWC2

Notes
-----
Much of this was copied, modified, and/or cleaned from DTU's Wind Energy
Toolbox.


Author
------
Jenni Rinker
rink@dtu.dk
"""
import os

import numpy as np
import pandas as pd


# hard-code parameters from .set file format
_sel_lineno_info = 8  # line no. in sel with info
_sel_lineno_chanstart = 12  # line no. channel start
_sel_col_wds = [13, 44, 55]  # character widths in sel file
_sel_df_cols = ['channel', 'var_desc', 'units', 'notes', 'scale']


def sel_to_df(path, unique=True):
    """Load data from a HAWC2 sel file into a pandas dataframe

    Arguments
    ---------
    path : str
        Path to .sel file
    unique : boolean
        Whether to also create unique channel names

    Returns
    -------
    res_df : pandas.DataFrame
        Pandas dataframe with results from HAWC2
    """

    # load info from sel file
    path = os.path.splitext(path)[0]  # remove ext if given
    sel_path = f'{path}.sel'  # path to sel file
    with open(sel_path, 'r') as sel_fid:
        sel_lines = sel_fid.readlines()
    line_info = sel_lines[_sel_lineno_info].split()
    n_chnls = int(line_info[1])
    fmt = line_info[3]

    # load data from sel file
    sel_df = pd.DataFrame(np.empty((n_chnls, len(_sel_df_cols))),
                          columns=_sel_df_cols)  # dataframe with sel data
    sel_df['scale'] = 1.  # initialize scale factor to one
    for i_line in range(n_chnls):
        sel_line = sel_lines[i_line + _sel_lineno_chanstart]
        sel_df.iloc[i_line, 0] = int(sel_line[:_sel_col_wds[0]].strip())  # chn
        sel_df.iloc[i_line, 1] = sel_line[_sel_col_wds[0]:
                                          _sel_col_wds[1]].strip()  # varble
        sel_df.iloc[i_line, 2] = sel_line[_sel_col_wds[1]:
                                          _sel_col_wds[2]].strip()  # units
        sel_df.iloc[i_line, 3] = sel_line[_sel_col_wds[2]:].strip()  # descptn
        if fmt.lower() == 'binary':  # get scale if binary
            sel_df.iloc[i_line, -1] = float(sel_lines[i_line +
                                                      _sel_lineno_chanstart
                                                      + n_chnls + 2])

    if unique:
        sel_df['unique'] = get_unique_chnl_names(sel_df)

    return sel_df


def get_unique_chnl_names(sel):
    """Unique channel names from hawc2 sel file

    This assumes the user made use of the hashtag's ability to custom name
    their own channels.
    """
    if isinstance(sel, str):
        sel_df = sel_to_df(sel)
    elif isinstance(sel, pd.DataFrame):
        sel_df = sel.copy()
    else:
        raise ValueError(f'Unknown input type {type(sel)}')
    sel_df['unique'] = ''
    sel_df.loc[sel_df.var_desc == 'Time', 'unique'] = 'time'
    for idx in sel_df[sel_df['notes'].str.contains('Free wind speed Vx')].index:
        row = sel_df.loc[idx, :]
        sel_df.loc[idx, 'unique'] = row.notes.split()[-1].replace('wind_', 'vxg_')
    for idx in sel_df[sel_df['notes'].str.contains('Free wind speed Vy')].index:
        row = sel_df.loc[idx, :]
        sel_df.loc[idx, 'unique'] = row.notes.split()[-1].replace('wind_', 'vyg_')
    for idx in sel_df[sel_df['notes'].str.contains('Free wind speed Vz')].index:
        row = sel_df.loc[idx, :]
        sel_df.loc[idx, 'unique'] = row.notes.split()[-1].replace('wind_', 'vzg_')

    return sel_df.unique.values


def get_num_scans(path):
    """get number of scans from sel file
    """
    # load info from sel file
    path = os.path.splitext(path)[0]  # remove ext if given
    sel_path = f'{path}.sel'  # path to sel file
    with open(sel_path, 'r') as sel_fid:
        sel_lines = sel_fid.readlines()
    line_info = sel_lines[_sel_lineno_info].split()
    return int(line_info[0])


def dat_to_df(path, n_scns=None, channel_names=None, sel_df=None,
              unique=True):
    """HAWC2 binary .dat file to pandas dataframe
    """
    # load necessary info if not passed in
    if n_scns is None:  # get number of scans if not passed in
        n_scns = get_num_scans(path)
    if channel_names is None:  # get channel names if not passed in
        channel_names = get_unique_chnl_names(path)
    if sel_df is None:  # get .sel dataframe if not passed in
        sel_df = sel_to_df(path)

    # load data in dat file
    dat_path = os.path.splitext(path)[0] + '.dat'  # path to dat file
    with open(dat_path, 'rb') as dat_fid:
        dat_df = pd.DataFrame(np.zeros((n_scns, len(channel_names))),
                              columns=channel_names)
        for i in range(sel_df.shape[0]):
            dat_fid.seek(i * n_scns * 2, 0)
            dat_df.iloc[:, i] = np.fromfile(dat_fid, 'int16', n_scns) * \
                sel_df.loc[i, 'scale']

    if unique:
        dat_df.columns = get_unique_chnl_names(sel_df)
    return dat_df
