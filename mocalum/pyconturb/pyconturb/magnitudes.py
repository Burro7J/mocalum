# -*- coding: utf-8 -*-
"""Functions related to definitions of spectral models
"""
import numpy as np

from pyconturb.spectral_models import get_spec_values
from pyconturb.sig_models import get_sig_values
from pyconturb._utils import get_freq


def get_magnitudes(spat_df, spec_func, sig_func, **kwargs):
    """"""
    t, freq = get_freq(**kwargs)
    spc_arr = get_spec_values(freq, spat_df, spec_func, **kwargs)
    mags_arr = spc_to_mag(spat_df, spc_arr, sig_func, **kwargs)
    return mags_arr


def spc_to_mag(spat_df, spc_arr, sig_func, **kwargs):
    """Convert spectral array to magnitudes (sets DC component to 0)
    """
    # get unscaled magnitudes
    t, freq = get_freq(**kwargs)
    n_t, df = t.size, freq[1]
    mags = np.sqrt(spc_arr * df / 2)  # (nf, nsp)
    mags[0, :] = 0.  # mean is zero to make math easier
    # scale to get the correct ti
    std_theo = get_sig_values(spat_df, sig_func, **kwargs)  # (n_sp,)
    if n_t == 2:
        std_now = np.sqrt(n_t/(n_t-1) * np.sum(np.abs(mags)**2, axis=0))
    else:
        std_now = np.sqrt(n_t/(n_t-1)
                          * (2*np.sum(np.abs(mags)**2, axis=0)
                             - ((n_t+1) % 2)*np.abs(mags[-1, :])**2))
    alpha = std_theo / std_now
    return (alpha * mags).astype(float)  # (nf, nsp)
