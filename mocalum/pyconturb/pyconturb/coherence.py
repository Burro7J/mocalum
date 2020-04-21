# -*- coding: utf-8 -*-
"""Functions related to definition of coherence models
"""
import itertools

import numpy as np


def get_coh_mat(freq, spat_df, coh_model='iec', **kwargs):
    """Create coherence matrix for given frequencies and coherence model
    """
    if coh_model == 'iec':  # IEC coherence model
        if 'ed' not in kwargs.keys():  # add IEC ed to kwargs if not passed in
            kwargs['ed'] = 3
        coh_mat = get_iec_coh_mat(freq, spat_df, **kwargs)
    elif coh_model == '3d':  # 3D coherence model
        coh_mat = get_3d_coh_mat(freq, spat_df, **kwargs)

    else:  # unknown coherence model
        raise ValueError(f'Coherence model "{coh_model}" not recognized.')

    return coh_mat


def get_iec_coh_mat(freq, spat_df,  **kwargs):
    """Create IEC 61400-1 Ed. 3 coherence matrix for given frequencies
    """
    # preliminaries
    if kwargs['ed'] != 3:  # only allow edition 3
        raise ValueError('Only edition 3 is permitted.')
    if any([k not in kwargs.keys() for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    freq = np.array(freq).reshape(1, -1)
    n_f, n_s = freq.size, spat_df.shape[1]
    # get indices of point-pairs
    ii_jj = [(i, j) for (i, j) in itertools.combinations(np.arange(n_s), 2)]
    ii = np.array([tup[0] for tup in ii_jj])
    jj = np.array([tup[1] for tup in ii_jj])
    # calculate distances and coherences
    xyz = spat_df.loc[['x', 'y', 'z']].values.astype(float)
    coh_mat = np.repeat(np.atleast_3d(np.eye((n_s))), n_f, axis=2)
    r = np.sqrt((xyz[1, ii] - xyz[1, jj])**2 + (xyz[2, ii] - xyz[2, jj])**2)
    # mask to u values and asign values
    mask = ((spat_df.iloc[0, ii].values == 0) & (spat_df.iloc[0, jj].values == 0))
    coh_values = np.exp(-12 *
                        np.sqrt((r[mask].reshape(-1, 1) / kwargs['u_ref'] * freq)**2
                                + (0.12 * r[mask].reshape(-1, 1) / kwargs['l_c'])**2))
    coh_mat[ii[mask], jj[mask], :] = coh_values
    coh_mat[jj[mask], ii[mask]] = np.conj(coh_values)
    return coh_mat


def get_3d_coh_mat(freq, spat_df, **kwargs):
    """Create coherence matrix with 3d coherence for given frequencies
    """
    if any([k not in kwargs.keys() for k in ['u_ref', 'l_c']]):  # check kwargs
        raise ValueError('Missing keyword arguments for IEC coherence model')
    # intermediate variables
    freq = np.array(freq).reshape(1, -1)
    n_f, n_s = freq.size, spat_df.shape[1]
    ii_jj = [(i, j) for (i, j) in itertools.combinations(np.arange(n_s), 2)]
    ii = np.array([tup[0] for tup in ii_jj])
    jj = np.array([tup[1] for tup in ii_jj])
    xyz = spat_df.loc[['x', 'y', 'z']].values
    coh_mat = np.repeat(np.atleast_3d(np.eye((n_s))), n_f, axis=2)
    r = np.sqrt((xyz[1, ii] - xyz[1, jj])**2 + (xyz[2, ii] - xyz[2, jj])**2)
    # loop through the three components
    for (k, lc_scale) in [(0, 1), (1, 2.7 / 8.1), (2, 0.66 / 8.1)]:
        l_c = kwargs['l_c'] * lc_scale
        mask = ((spat_df.iloc[0, ii].values == k) & (spat_df.iloc[0, jj].values == k))
        coh_values = np.exp(-12 *
                            np.sqrt((r[mask].reshape(-1, 1) / kwargs['u_ref'] * freq)**2
                                    + (0.12 * r[mask].reshape(-1, 1) / l_c)**2))
        coh_mat[ii[mask], jj[mask], :] = coh_values
        coh_mat[jj[mask], ii[mask]] = np.conj(coh_values)
    return coh_mat
