import numpy as np
import xarray as xr
from .persistance import data
from .utils import move2time, spher2cart, get_plaw_uvw, project2los, ivap_rc
from .samples import gen_unc
from tqdm import tqdm


class Mocalum:

    def __init__(self):
        self.data = data # this is an instance of Data() from persistance.py
        self.u = None
        self.x_res = 10
        self.y_res = 10
        self.z_res = 1

    def set_meas_cfg(self):
        pass

    def _calc_xyz(self):
        x,y,z = spher2cart(self.data.probing.az  + self.data.probing.unc_az,
                           self.data.probing.el  + self.data.probing.unc_el,
                           self.data.probing.rng + self.data.probing.unc_rng)

        x.values+=self.data.meas_cfg['lidar_pos'][0]
        y.values+=self.data.meas_cfg['lidar_pos'][1]
        z.values+=self.data.meas_cfg['lidar_pos'][2]

        self.data._add_xyz(x,y,z)

    def _cr8_ffield_bbox(self):

        x_coord= np.array([self.data.probing.x.min(), self.data.probing.x.max()])
        y_coord= np.array([self.data.probing.y.min(), self.data.probing.y.max()])
        z_coord= np.array([self.data.probing.z.min(), self.data.probing.z.max()])
        time_steps = self.data.probing.time.values

        # create/updated flow field bounding box config dict
        self.data._cr8_bbox_dict(x_coord, y_coord, z_coord,
                                 self.x_res, self.y_res, self.z_res, time_steps)



    def set_ivap_probing(self, lidar_pos, sector_size, azimuth_mid, angular_res,
                         elevation, rng, no_scans = 1,
                         scan_speed=1, max_speed=50, max_acc=100):

        # prevent users stupidity raise errors!
        if angular_res <= 0:
            raise ValueError('Angular resolution must be > 0!')
        if no_scans <= 0:
            raise ValueError('Number of scans must be > 0!')
        if type(no_scans) is not int:
            raise ValueError('Number of scans must be int!')

        az = np.arange(azimuth_mid-sector_size/2,
                       azimuth_mid+sector_size/2 + angular_res,
                       angular_res, dtype=float)
        rng = np.full(len(az), rng, dtype=float)
        el = np.full(len(az), elevation, dtype=float)
        no_los = len(az)
        sweep_time = move2time(sector_size, max_acc, max_speed)
        scan_time = sector_size*scan_speed

        # create measurement configuration dictionary
        self.data._cr8_meas_cfg(lidar_pos, 'IVAP', az, el, rng, no_los, no_scans,
                                scan_speed, sector_size, sweep_time,
                                max_speed, max_acc)

        # tile probing coordinates to match no of scans
        az = np.tile(az, no_scans)
        rng = np.tile(rng, no_scans)
        el = np.tile(el, no_scans)

        # creating time dim
        time = np.arange(0, sector_size*scan_speed + angular_res*scan_speed,
                        angular_res*scan_speed)
        time = np.tile(time, no_scans)
        to_add = np.repeat(scan_time + sweep_time, no_scans*no_los)
        multip = np.repeat(np.arange(0,no_scans), no_los)
        time = time + multip*to_add

        # create probing dataset later to be populated with probing unc
        self.data._cr8_probing_ds(az, el, rng, time)

        # adding xyz (these will not have uncertainties)
        self._calc_xyz()

        # create flow field bounding box dict
        self._cr8_ffield_bbox()

    def gen_unc_contributors(self, corr_coef=0,
                             unc_cfg={'unc_az':{'mu':0, 'std':0.1},
                                      'unc_el':{'mu':0, 'std':0.1},
                                      'unc_rng':{'mu':0, 'std':10},
                                      'unc_est':{'mu':0, 'std':0.1}}):

        # store information in unc_cfg
        self.data.unc_cfg.update(unc_cfg)
        self.data.unc_cfg.update({'corr_coef':corr_coef})

        # sample uncertainty contribution considering normal distribution
        # and add them to probing xr.DataSet
        for unc_term, cfg in unc_cfg.items():
            samples = gen_unc(np.full(self.data.meas_cfg['no_los'], cfg['mu']),
                              np.full(self.data.meas_cfg['no_los'], cfg['std']),
                              corr_coef, self.data.meas_cfg['no_scans'])
            self.data._add_unc(unc_term, samples.flatten())

        # update x,y,z coordinates of measurement points
        self._calc_xyz()

        # update flow field bounding box dict
        self._cr8_ffield_bbox()


    def gen_plaw_ffield(self,ws=10, wdir=180, w=0, href=100, alpha=0.2):
        """Generates flow field assuming power law

        Parameters
        ----------
        ws : int, optional
            wind speed, by default 10
        wdir : int, optional
            wind direction, by default 180
        w : int, optional
            vertical wind speed, by default 0
        href : int, optional
            reference height, by default 100
        alpha : float, optional
            shear expoenent, by default 0.2
        """

        no_dim = 3
        fmodel_cfg= {'flow_model':'power_law',
                     'wind_speed':ws,
                     'upward_velocity':w,
                     'wind_from_direction':wdir,
                     'reference_height':href,
                     'shear_expornent':alpha,
                     }
        self.data._cr8_fmodel_cfg(fmodel_cfg)
        self.data._cr8_empty_ffield_ds(no_dim)

        hmeas = self.data.ffield.z.values
        u, v, w = get_plaw_uvw(hmeas, href, ws,w, wdir, alpha)
        self.u = u
        self.v = v
        self.data._upd8_ffield_ds(u, v, w, no_dim)


    def calc_los_speed(self):
        """Calcutes projection of wind speed on beam line-of-sight
        """
        if self.data.fmodel_cfg['flow_model'] == 'power_law':
            hmeas = self.data.probing.z.values
            u = self.data.ffield.u.isel(x=0,y=0).interp(z=hmeas)
            v = self.data.ffield.v.isel(x=0,y=0).interp(z=hmeas)
            w = self.data.ffield.w.isel(x=0,y=0).interp(z=hmeas)

        else:
            # slowest possible way of pulling u,v,w
            # since it currently steps through each time stamp
            # and interpolates data
            tme = self.data.probing.time.values
            u = np.empty(len(tme))
            v = np.empty(len(tme))
            w = np.empty(len(tme))

            for i,t in enumerate(tqdm(tme, desc='Projecting LOS')):
                x = self.data.probing.x.sel(time = t).values
                y = self.data.probing.y.sel(time = t).values
                z = self.data.probing.z.sel(time = t).values
                u[i] = self.data.ffield.u.sel(time=t).interp(x=x, y=y, z=z)
                v[i] = self.data.ffield.v.sel(time=t).interp(x=x, y=y, z=z)
                w[i] = self.data.ffield.w.sel(time=t).interp(x=x, y=y, z=z)

        los = project2los(u,v,w,
                          self.data.probing.az.values +
                          self.data.probing.unc_az.values,
                          self.data.probing.el.values +
                          self.data.probing.unc_el.values)
        los += self.data.probing.unc_est.values

        self.data._cr8_los_ds(los)

    def reconstruct_wind(self, rc_method = 'IVAP'):
        """Reconstructs wind speed according to the selected retrieval method

        Parameters
        ----------
        rc_method : str, optional
            Retrieval method, by default 'IVAP'
        """

        if rc_method == 'IVAP':
            vrad = np.asarray(np.split(self.data.los.vrad.values,
                                    self.data.los.no_scans.values))
            azm = np.asarray(np.split(self.data.los.az.values,
                                    self.data.los.no_scans.values))
            u, v, ws = ivap_rc(vrad, azm, 1)

            self.data._cr8_rc_wind_ds(u,v,ws)
        else:

            print('Unsupported wind reconstruction method')
