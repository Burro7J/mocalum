# -*- coding: utf-8 -*-
"""Create an unconstrained or constrained turbulence box.

The main function function, ``gen_turb`` can be used both with and without
constraining data. Please see the Examples section in the documentation
to see how to use it.
"""
import warnings

import numpy as np
import pandas as pd

from pyconturb.coherence import get_coh_mat
from pyconturb.core import TimeConstraint
from pyconturb.magnitudes import get_magnitudes
from pyconturb.sig_models import iec_sig, data_sig
from pyconturb.spectral_models import kaimal_spectrum, data_spectrum
from pyconturb.wind_profiles import get_wsp_values, power_profile, data_profile
from pyconturb._utils import combine_spat_con, _spat_rownames, _DEF_KWARGS, clean_turb


def gen_turb(spat_df, T=600, dt=1, con_tc=None, coh_model='iec',
             wsp_func=None, sig_func=None, spec_func=None,
             interp_data='none', seed=None, nf_chunk=1, verbose=False, **kwargs):
    """Generate a turbulence box (constrained or unconstrained).

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have rows `[k, x, y, z]`,
        and each of the `n_sp` columns corresponds to a different spatial location and
        turbine component (u, v or w).
    T : float, optional
        Total length of time to simulate in seconds. Default is 600.
    dt : float, optional
        Time step for generated turbulence in seconds. Default is 1.
    con_tc : pyconturb TimeConstraint, optional
        Optional constraining data for the simulation. The TimeConstraint object is built
        into PyConTurb; see documentation for more details.
    coh_model : str, optional
        Spatial coherence model specifier. Default is IEC 61400-1.
    wsp_func : function, optional
        Function to specify spatial variation of mean wind speed. See details
        in `Mean wind speed profiles` section.
    sig_func : function, optional
        Function to specify spatial variation of turbulence standard deviation.
        See details in `Turbulence standard deviation` section.
    spec_func : function, optional
        Function to specify spatial variation of turbulence power spectrum. See
        details in `Turbulence spectra` section.
    interp_data : str, optional
        Interpolate mean wind speed, standard deviation, and/or power spectra profile
        functions from provided constraint data. Possible options are ``'none'`` (use
        provided/default profile functions), ``'all'`` (interpolate all profile,
        functions), or a list containing and combination of ``'wsp'``, ``'sig'`` and
        ``'spec'`` (interpolate the wind speed, standard deviation and/or power spectra).
        Default is IEC 61400-1 profiles (i.e., no interpolation).
    seed : int, optional
        Optional random seed for turbulence generation. Use the same seed and
        settings to regenerate the same turbulence box.
    nf_chunk : int, optional
        Number of frequencies in a chunk of analysis. Increasing this number may speed
        up computation but may result in more (or too much) memory used. Smaller grids
        may benefit from larger values for ``nf_chunk``. Default is 1.
    verbose : bool, optional
        Print extra information during turbulence generation. Default is False.
    **kwargs
        Optional keyword arguments to be fed into the
        spectral/turbulence/profile/etc. models.

    Returns
    -------
    turb_df : pandas.DataFrame
        Generated turbulence box. Each row corresponds to a time step and each
        column corresponds to a point/component in ``spat_df``.
    """
    if verbose:
        print('Beginning turbulence simulation...')
    # if con_data passed in, throw deprecation warning
    if ('con_data' in kwargs) and (con_tc is None):  # don't use con_data please
        warnings.warn('The con_data option is deprecated and will be removed in future' +
                      ' versions. Please see the documentation for how to specify' +
                      ' time constraints.',
                      DeprecationWarning, stacklevel=2)
        con_tc = TimeConstraint().from_con_data(kwargs['con_data'])
    # if asked to interpret but no data, throw warning
    if (((interp_data == 'all') or isinstance(interp_data, list)) and (con_tc is None)):
        raise ValueError('If "interp_data" is not "none", constraints must be given!')

    # add T, dt, con_tc to kwargs
    kwargs = {**_DEF_KWARGS, **kwargs, 'T': T, 'dt': dt, 'con_tc': con_tc}
    wsp_func, sig_func, spec_func = assign_profile_functions(wsp_func, sig_func,
                                                             spec_func, interp_data)

    # assign/create constraining data
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))  # no. time steps
    t = np.arange(n_t) * kwargs['dt']  # time array
    if con_tc is None:  # create empty constraint data if not passed in
        constrained, n_d = False, 0  # no. of constraints
        con_tc = TimeConstraint(index=_spat_rownames)
    else:
        constrained = True
        n_d = con_tc.shape[1]  # no. of constraints
        if not np.array_equal(con_tc.get_time().index, t):
            raise ValueError('TimeConstraint time does not match requested T, dt!')

    # combine data and sim spat_dfs
    all_spat_df = combine_spat_con(spat_df, con_tc)  # all sim points
    n_s = all_spat_df.shape[1]  # no. of total points to simulate

    one_point = False
    if n_s == 1:  # only one point, skip coherence
        one_point = True

    # intermediate variables
    n_f = n_t // 2 + 1  # no. freqs
    freq = np.arange(n_f) / kwargs['T']  # frequency array

    # get magnitudes of points to simulate. (nf, nsim). con_tc in kwargs.
    sim_mags = get_magnitudes(all_spat_df.iloc[:, n_d:], spec_func, sig_func, **kwargs)

    if constrained:
        conturb_fft = np.fft.rfft(con_tc.get_time().values, axis=0) / n_t  # constr fft
        con_mags = np.abs(conturb_fft)  # mags of constraints
        all_mags = np.concatenate((con_mags, sim_mags), axis=1)  # con and sim
    else:
        all_mags = sim_mags  # just sim

    # get uncorrelated phasors for simulation
    np.random.seed(seed=seed)  # initialize random number generator
    sim_unc_pha = np.exp(1j * 2*np.pi * np.random.rand(n_f, n_s - n_d))
    if not (n_t % 2):  # if even time steps, last phase must be 0 or pi for real sig
        sim_unc_pha[-1, :] = np.exp(1j * np.round(np.real(sim_unc_pha[-1, :])) * np.pi)

    # no coherence if one point
    if one_point:
        turb_fft = all_mags * sim_unc_pha

    # if more than one point, correlate everything
    else:
        turb_fft = np.zeros((n_f, n_s), dtype=complex)
        n_chunks = int(np.ceil(freq.size / nf_chunk))

        # loop through frequencies
        for i_f in range(1, freq.size):
            i_chunk = i_f // nf_chunk  # calculate chunk number
            if (i_f - 1) % nf_chunk == 0:  # genr cohrnc chunk when needed
                if verbose:
                    print(f'  Processing chunk {i_chunk + 1} / {n_chunks}')
                all_coh_mat = get_coh_mat(freq[i_chunk * nf_chunk:
                                               (i_chunk + 1) * nf_chunk],
                                          all_spat_df, coh_model=coh_model,
                                          **kwargs)

            # assemble "sigma" matrix, which is coh matrix times mag arrays
            coh_mat = all_coh_mat[:, :, i_f % nf_chunk]
            sigma = np.einsum('i,j->ij', all_mags[i_f, :],
                              all_mags[i_f, :]) * coh_mat

            # get cholesky decomposition of sigma matrix
            cor_mat = np.linalg.cholesky(sigma)

            # if constraints, assign data unc_pha
            if constrained:
                dat_unc_pha = np.linalg.solve(cor_mat[:n_d, :n_d], conturb_fft[i_f, :])
            else:
                dat_unc_pha = []
            unc_pha = np.concatenate((dat_unc_pha, sim_unc_pha[i_f, :]))
            cor_pha = cor_mat @ unc_pha

            # calculate and save correlated Fourier components
            turb_fft[i_f, :] = cor_pha

        del all_coh_mat  # free up memory

    # convert to time domain and pandas dataframe
    turb_arr = np.fft.irfft(turb_fft, axis=0, n=n_t) * n_t
    turb_df = pd.DataFrame(turb_arr, columns=all_spat_df.columns, index=t)

    # return just the desired simulation points
    turb_df = clean_turb(spat_df, all_spat_df, turb_df)

    # add in mean wind speed according to specified profile
    wsp_profile = get_wsp_values(spat_df, wsp_func, **kwargs)
    turb_df[:] += wsp_profile

    if verbose:
        print('Turbulence generation complete.')

    return turb_df


def assign_profile_functions(wsp_func, sig_func, spec_func, interp_data):
    """Assign profile functions based on user inputs"""
    # assign iec profile functions as default if nothing was passed in
    prof_funcs = [wsp_func, sig_func, spec_func]  # given inputs
    iec_funcs = [power_profile, iec_sig,  kaimal_spectrum]  # iec default functions
    prof_funcs = [tup[tup[0] is None] for tup in zip(prof_funcs, iec_funcs)]  # overwrite
    # change interp_data to list format if 'all' or 'none' passed in
    if interp_data == 'none': interp_data = []  # no profs interpolated
    elif interp_data == 'all': interp_data = ['wsp', 'sig', 'spec']  # all profs interp'd
    elif not isinstance(interp_data, list):  # bad input
        raise ValueError('"interp_data" must be either "all", "none", or a list!')
    # assign the interp_data profiles IF no custom function was passed in
    for prof in interp_data:
        if (prof == 'wsp') and (wsp_func is None): prof_funcs[0] = data_profile
        elif (prof == 'sig') and (sig_func is None): prof_funcs[1] = data_sig
        elif (prof == 'spec') and (spec_func is None): prof_funcs[2] = data_spectrum
        else: raise ValueError(f'Unknown profile type "{prof}"!')  # bad input
    return prof_funcs
