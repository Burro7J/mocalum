# -*- coding: utf-8 -*-
"""Command-line script for converting V52 met mast file to HAWC2 turbulence box

Usage
-----
$ python metmast_to_h2turb.py ny nz sy sz zc sim_type path_mmast path_turbdir
(see comments in script for what these mean)
"""
import sys

import numpy as np

from pyconturb.core.simulation import gen_turb
from pyconturb.core.helpers import df_to_hawc2, gen_spat_grid
from pyconturb.io.data_conversion import v52_pickle_to_condata


if __name__ == '__main__':

    inp_args = sys.argv  # get list of input arguments

    # default parameters for optional inputs
    seed = None  # random seed
    turb_name = 'turb_'  # prefix for turbulence box name when saved
    mem_gb = 0.20  # amount of memory to use in gigabytes
    verbose = False  # print out arguments during simulation [-]
    T_sim = 600  # length of time to simulatione [s]
    dt_sim = 0.10  # desired simulation time step [s]
    i_hub = 2  # index of hub height measurement [-]
    scale = True  # scale spectra to ensure correct std dev [-]

    # load values from command line
    ny = int(inp_args[1])  # no. points in y direction
    nz = int(inp_args[2])  # no. points in z direction
    sy = float(inp_args[3])  # lateral box size [m]
    sz = float(inp_args[4])  # vertical box size [m]
    zc = float(inp_args[5])  # height of box center [m]
    sim_type = inp_args[6]  # unconstr or constr simltn ('unc' or 'con')
    path_metmast = inp_args[7]  # path to met mast pickle file
    path_turb = inp_args[8]  # path to direct to save turb (no trailing \)

    # calculate coherence length scale from box center
    l_c = 8.1 * (42 * (zc > 60) + 0.7 * zc * (zc <= 60))

    # create spatial dataframe for pyconturb
    y = np.linspace(-sy/2, sy/2, ny)
    z = np.linspace(-sz/2, sz/2, nz) + zc
    spat_df = gen_spat_grid(y, z)

    # calculate hub-height wind speed for both unc and con cases
    # (note this is only used for coherenc calculations -- the mean wind speed
    #   profile is defined in the htc file so the turb box is zero-mean)
    con_data = v52_pickle_to_condata(path_metmast)
    con_spat_df = con_data['con_spat_df']
    con_turb_df = con_data['con_turb_df']
    v_hub = -np.mean(con_turb_df[f'vxt_p{i_hub:.0f}'].values)

    # calculate reference turbulence intensity
    sig_u = np.std(con_turb_df[f'vxt_p{i_hub:.0f}'].values)
    i_ref = sig_u / (0.75 * v_hub + 5.6)  # eq. 11 in IEC 61400-1

    # assign keyword arguments
    kwargs = {'v_hub': v_hub, 'ed': 3, 'l_c': l_c, 'z_hub': zc,
              'T': T_sim, 'dt': dt_sim, 'i_ref': i_ref}

    # assign values for unconstrained case
    if sim_type == 'unc':
        con_data = None
        coh_model, spc_model, wsp_model = 'iec', 'kaimal', 'none'

    # assign values for constrained case
    elif sim_type == 'con':
        kwargs['method'] = 'z_interp'  # how to get spectral values @ other hts
        coh_model, spc_model, wsp_model = 'iec', 'data', 'none'

        # resample constraint data to match dt_sim and T_sim
        t_sim = np.arange(0, T_sim, dt_sim)  # new time vector
        con_turb_df = con_turb_df.reindex(index=t_sim)  # add nans
        con_turb_df = con_turb_df.interpolate(
                method='linear').fillna(method='bfill')  # interpolate nans
        con_turb_df = con_turb_df.loc[t_sim]  # pull out only sim times

    # throw error if unrecognized input
    else:
        raise ValueError(f'Unrecognized simulation type "{sim_type}"')

    # simulate turbulence
    turb_df = gen_turb(spat_df, con_data=con_data,
                       coh_model=coh_model, spc_model=spc_model,
                       wsp_model=wsp_model, scale=scale,
                       seed=seed, mem_gb=mem_gb, verbose=verbose,
                       **kwargs)

    # save in hawc2 format
    df_to_hawc2(turb_df, spat_df, path_turb,
                prefix=turb_name)
