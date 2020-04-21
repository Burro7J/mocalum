# -*- coding: utf-8 -*-
"""generate the constraining turbulence used in the example

Author
------
Jenni Rinker
rink@dtu.dk
"""
import os

from ksec3d.core.simulation import gen_turb
from ksec3d.core.helpers import gen_spat_grid


# execute this block if file is run as script
if __name__ == '__main__':

    # inputs to generate constraining turbulence
    y, z = 0, [15, 31, 50, 85, 110, 131]  # heights mimic a met mast
    kwargs = {'v_hub': 10, 'i_ref': 0.14, 'ed': 3, 'l_c': 340.2,
              'z_hub': 70, 'T': 100, 'dt': 0.5}  # arguments for iec turb sim
    coh_model = 'iec'  # IEC 61400-1 Ed. 3 coherence model
    spc_model = 'kaimal'  # kaimal spectrum
    wsp_model = 'iec'  # IEC 61400-1 Ed. 3 mean wind speed profile (power law)
    seed = 3003  # set the random seed so we can reproduce results

    con_spat_df = gen_spat_grid(y, z)  # generate the grid of points to sim
    spat_csv = os.path.join('.', 'conturb_spat.csv')
    con_spat_df.to_csv(spat_csv, index=False)

    con_turb_df = gen_turb(con_spat_df, coh_model=coh_model,
                           spc_model=spc_model, wsp_model=wsp_model, seed=seed,
                           **kwargs).reset_index()  # simulate the turbulence

    turb_csv = os.path.join('.', 'conturb.csv')
    con_turb_df.rename(columns={'index': 'time'}).to_csv(turb_csv,
                                                         index=False)
