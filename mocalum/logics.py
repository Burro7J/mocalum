import numpy as np
from numpy.linalg import inv as inv
import xarray as xr
from .persistance import data
from .utils import move2time, spher2cart, get_plaw_uvw, project2los, ivap_rc, _rot_matrix
from .utils import sliding_window_slicing, bbox_pts_from_array, bbox_pts_from_cfg
from .utils import calc_mean_step
from .samples import gen_unc
from tqdm import tqdm


# turbulence box generation tools
from pyconturb.wind_profiles import power_profile
from pyconturb import gen_turb, gen_spat_grid

class Mocalum:

    def __init__(self):
        self.data = data # this is an instance of Data() from persistance.py
        self.x_res = 25
        self.y_res = 25
        self.z_res = 5
        self.turbbox_time = 600 # seconds = 10 min
        self.t_res = None
        self.tmp_bb = None

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
        # TODO: Maybe add rot matrix which does not do anything to coordinates
        CRS = {'x':'Absolute coordinate, coresponds to Easting in m',
               'y':'Absolute coordinate, coresponds to Northing in m',
               'z':'Absolute coordinate, coresponds to height above sea level in m',
               'rot_matrix':None}

        x_coord= np.array([self.data.probing.x.min(), self.data.probing.x.max()])
        y_coord= np.array([self.data.probing.y.min(), self.data.probing.y.max()])
        z_coord= np.array([self.data.probing.z.min(), self.data.probing.z.max()])
        t_coord = self.data.probing.time.values
        t_res = calc_mean_step(t_coord)
        self.t_res = t_res

        # create/updated flow field bounding box config dict
        self.data._cr8_bbox_dict(CRS,
                                 x_coord, y_coord, z_coord, t_coord,
                                 0,0,0,0,
                                 self.x_res, self.y_res, self.z_res, t_res)



    def _cr8_turbbbox(self):
        avg_azimuth = self.data.meas_cfg['az'].mean()
        ws = self.data.fmodel_cfg['wind_speed']
        wdir = self.data.fmodel_cfg['wind_from_direction']
        R_az = _rot_matrix(-avg_azimuth) #2D rot_matrix
        R_tb = _rot_matrix(wdir)
        CRS = {'x':'Relative coordinate, use inv(rot_matrix) to convert to abs',
               'y':'Relative coordinate, use inv(rot_matrix) to convert to abs',
               'z':'Absolute coordinate, coresponds to height above sea level',
               'rot_matrix':R_tb}
        # get probing coordinates
        beam_xyz = self._get_prob_cords()

        # reproject probing coordinates to coordinate system where
        # y axis is aligned with the mean azimuth direction
        beam_xy= beam_xyz[:,(0,1)].dot(R_az)

        # create bounding box around the beam positions while rotating
        # the bounding box points back to absolute coordinate system
        bbox_pts = bbox_pts_from_array(beam_xy).dot(inv(R_az))

        # reproject those points now to turbulence box coordinate system
        # in which x axis is aligned with the mean wind direction:
        bbox_pts = bbox_pts.dot(R_tb)

        # Calculating time
        t_res = self.x_res / ws
        self.t_res = t_res
        t_coord = np.arange(0, self.turbbox_time + t_res, t_res)

        self.data._cr8_bbox_dict(CRS,
                                 bbox_pts[:,0], bbox_pts[:,1], beam_xyz[:,2],
                                 t_coord,
                                 0, 0, 0, 0,
                                 self.x_res, self.y_res, self.z_res, t_res)

    def gen_turb_ffield(self,ws=10, wdir=180, w=0, href=100, alpha=0.2):
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
        fmodel_cfg= {'flow_model':'power_law',
                     'wind_speed':ws,
                     'upward_velocity':w,
                     'wind_from_direction':wdir,
                     'reference_height':href,
                     'shear_expornent':alpha,
                     }
        self.data._cr8_fmodel_cfg(fmodel_cfg)
        self._cr8_turbbbox()
        # self.data._cr8_empty_tfield_ds()

        _, y, z, _ = self.data._get_ffield_coords()
        spat_df = gen_spat_grid(y, z)


        turb_df = gen_turb(spat_df, T=self.turbbox_time + self.data.ffield_bbox_cfg['t']['res'],
                           dt=self.data.ffield_bbox_cfg['t']['res'],
                           wsp_func = power_profile,
                           u_ref=ws, z_ref=href, alpha=alpha)

        self.data._cr8_3d_tfield_ds(turb_df)

        # self.data._upd8_tfield_ds(turb_df)
        self._to_4D_ds()


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

        fmodel_cfg= {'flow_model':'power_law',
                     'wind_speed':ws,
                     'upward_velocity':w,
                     'wind_from_direction':wdir,
                     'reference_height':href,
                     'shear_exponent':alpha,
                     }
        self.data._cr8_fmodel_cfg(fmodel_cfg)

        z_coord= np.arange(self.data.ffield_bbox_cfg['z']['min'],
                           self.data.ffield_bbox_cfg['z']['max'],
                           self.data.ffield_bbox_cfg['z']['res'])


        u, v, w = get_plaw_uvw(z_coord, href, ws, w, wdir, alpha)

        self.data._cr8_plfield_ds(u, v, w)

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

    def _get_prob_cords(self):

        x = self.data.probing.x.values
        y = self.data.probing.y.values
        z = self.data.probing.z.values

        return np.array([x,y,z]).transpose()


    def _to_4D_ds(self):
        ws = self.data.fmodel_cfg['wind_speed']
        bbox_pts = bbox_pts_from_cfg(self.data.ffield_bbox_cfg)

        x_start_pos = bbox_pts[:,0].min()
        x_len = abs(bbox_pts[:,0].max() - bbox_pts[:,0].min())
        x_res = ws * self.t_res
        no_items = int(np.ceil(x_len / x_res)) + 1

        u_3d = self.data.ffield.u.values.transpose()
        v_3d = self.data.ffield.v.values.transpose()
        w_3d = self.data.ffield.w.values.transpose()

        u_4d = sliding_window_slicing(u_3d, no_items, item_type=1).transpose(0,3,2,1)
        v_4d = sliding_window_slicing(v_3d, no_items, item_type=1).transpose(0,3,2,1)
        w_4d = sliding_window_slicing(w_3d, no_items, item_type=1).transpose(0,3,2,1)

        t = np.arange(0, u_4d.shape[0]*self.t_res, self.t_res)
        x_coord = np.arange(0, no_items*x_res, x_res) + x_start_pos

        self.data._cr8_4d_tfield_ds(u_4d, v_4d, w_4d, x_coord, t)