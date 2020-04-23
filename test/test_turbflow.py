import numpy as np
import xarray as xr
import mocalum as mc

koshava_xyz = [0, 0, 0]
no_scans = 10000
# wind field setup
ref_height = 100      # power law reference height
meas_height = 100
shear_exponent = 0.2  # power law shear exponent
wind_speed = 15       # wind speed at referenec height
wind_dir = 340        # wind direction
w = 0

# beam steering setup
distance = 1000   # meter
elevation = np.degrees(np.arcsin(meas_height / distance))
angular_res = 1   # degree
azimuth_mid = 0   # central azimuth angle
sector_size = 90  # degree
scan_speed = 1    # degree.s^-1
max_speed = 50    # degree.s^-1
max_acc = 100     # degree.s^-2

# Uncertainty terms
no_sim = 10000 # number of simulations
corr_coef = 0   # correlation coefficient
mu = 0         # we assume no systematic uncertainty
azim_std = 0.1 # degrees
elev_std = 0.1 # degrees
dis_std = 10   # meters
rad_std = 0.1  # m.s-1  In [23]:

tmp = mc.Mocalum()
# tmp.x_res = tmp.y_res = 10

tmp.set_ivap_probing(koshava_xyz, sector_size, azimuth_mid, angular_res, elevation, distance,
                    no_scans, scan_speed, max_speed,max_acc)

tmp.gen_unc_contributors(corr_coef)


tmp.x_res = tmp.y_res = 25
tmp.z_res = 5
tmp.gen_turb_ffield(wind_speed, wind_dir, w, ref_height, shear_exponent)
