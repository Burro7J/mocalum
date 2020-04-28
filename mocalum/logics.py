import numpy as np
from numpy.linalg import inv as inv
import xarray as xr
from .persistance import data
from .utils import move2time, spher2cart, get_plaw_uvw, project2los, ivap_rc, _rot_matrix
from .utils import sliding_window_slicing, bbox_pts_from_array, bbox_pts_from_cfg
from .utils import calc_mean_step, safe_execute
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
        self.tmp_los = None

    @staticmethod
    def _is_lidar_pos(lidar_pos):
        """
        Validates if the lidar position is according to definition

        Parameters
        ----------
        lidar_pos : ndarray
            nD array containing data with `float` or `int` type
            corresponding to x, y and z coordinates of a lidar.
            nD array data are expressed in meters.

        Returns
        -------
            True / False
        """

        rules = [True, True, True]

        exec("try: rules[0] = type(lidar_pos).__module__ == np.__name__\nexcept: rules[0]=False")
        exec("try: rules[1] = len(lidar_pos.shape) == 1\nexcept: rules[1]=False")
        exec("try: rules[1] = lidar_pos.shape[0] == 3\nexcept: rules[2]=False")


        messages = ["Not numpy array",
                    "Lidar position must be one dimensional numpy array",
                    "Lidar position array must contain three elements"]
        if all(rules):
            return True
        else:
            for i, rule in enumerate(rules):
                if rule == False:
                    print(messages[i])
            return False

    def add_lidar(self, id, lidar_pos, **kwargs):
        """Adds a lidar instance to the measurement config

        Lidars can be add one at time. Currently only the instrument position
        in UTM coordinate system is supported.

        Parameters
        ----------
        instrument_id : str, required
            String which identifies instrument in the instrument dictionary.
        lidar_pos : numpy, required
            1D array containing data with `float` or `int` type corresponding
            to Northing, Easting and Height coordinates of the instrument.
            1D array data are expressed in meters.

        Other Parameters
        -----------------
        unc_est : float, optional
            Uncertainty in estimating radial velocity from Doppler spectra.
            Unless provided, (default) value is set to 0.1 m/s.
        unc_rng : float, optional
            Uncertainty in detecting range at which atmosphere is probed.
            Unless provided, (default) value is set to 1 m.
        unc_az : float, optional
            Uncertainty in the beam steering for the azimuth angle.
            Unless provided, (default) value is set to 0.1 deg.
        unc_el : float, optional
            Uncertainty in the beam steering for the elevation angle.
            Unless provided, (default) value is set to 0.1 deg.
        corr_coef : float, optional
            Correlation coffecient of inputs
            Default 0, uncorrelated inputs
            Max 1, 100% correlated inputs

        Raises
        ------
        InappropriatePosition
            If the provided position of instrument is not properly provided.


        """
        if not(self._is_lidar_pos(lidar_pos)):
            raise ValueError("Lidar position is not properly provided")


        unc_elements = ['unc_est', 'unc_az','unc_el',
                        'unc_rng', 'corr_coef']

        lidar_dict = {id:{'position': lidar_pos,
                          'uncertainty':{
                              'unc_az':{'mu':0,'std':0.1,'units':'deg'},
                              'unc_el':{'mu':0, 'std':0.1, 'units':'deg'},
                              'unc_rng':{'mu':0, 'std':10, 'units':'m'},
                              'unc_est':{'mu':0, 'std':0.1, 'units':'m.s^-1'},
                              'corr_coef':0},
                          'config': {},
                                        }
                        }

        if kwargs != None:
            for element in unc_elements:
                if element in kwargs:
                    lidar_dict['uncertainty']['element'] = kwargs['element']


        self.data.meas_cfg.update(lidar_dict)

        # if self.verbos:
        #     print('Instrument \'' + instrument_id + '\' of category \'' +
        #         category +'\' added to the instrument dictionary, ' +
        #         'which now contains ' + str(len(self.instruments)) +
        #         ' instrument(s).')

    def _calc_xyz(self, lidar_id):
        probing_ds = self.data.probing[lidar_id]
        x,y,z = spher2cart(probing_ds.az  + probing_ds.unc_az,
                           probing_ds.el  + probing_ds.unc_el,
                           probing_ds.rng  + probing_ds.unc_rng)

        x.values+=self.data.meas_cfg[lidar_id]['position'][0]
        y.values+=self.data.meas_cfg[lidar_id]['position'][1]
        z.values+=self.data.meas_cfg[lidar_id]['position'][2]
        self.x = x
        self.data._add_xyz(lidar_id,x,y,z)

    def _cr8_ffield_bbox(self, lidar_id):
        # TODO: Maybe add rot matrix which does not do anything to coordinates
        CRS = {'x':'Absolute coordinate, coresponds to Easting in m',
               'y':'Absolute coordinate, coresponds to Northing in m',
               'z':'Absolute coordinate, coresponds to height above sea level in m',
               'rot_matrix':_rot_matrix(90)}

        x_coord= np.array([self.data.probing[lidar_id].x.min() - self.x_res,
                           self.data.probing[lidar_id].x.max() + self.x_res])
        y_coord= np.array([self.data.probing[lidar_id].y.min() - self.y_res,
                           self.data.probing[lidar_id].y.max() + self.y_res])
        z_coord= np.array([self.data.probing[lidar_id].z.min() - self.z_res,
                           self.data.probing[lidar_id].z.max() + self.z_res])
        t_coord = self.data.probing[lidar_id].time.values
        t_res = calc_mean_step(t_coord)
        self.t_res = t_res

        # create/updated flow field bounding box config dict
        self.data._cr8_bbox_dict(lidar_id, CRS,
                                 x_coord, y_coord, z_coord, t_coord,
                                 0,0,0,0,
                                 self.x_res, self.y_res, self.z_res, t_res)



    def _cr8_turbbbox(self, lidar_id):
        # needs to find an average azimuth between n lidar
        try:
            if type(lidar_id) == str:
                avg_azimuth = self.data.meas_cfg[lidar_id]['config']['az'].mean()
                beam_xyz = self._get_prob_cords(lidar_id)
                name_bbox = 'turb_for_' + lidar_id

            elif type(lidar_id) == list or type(lidar_id) == np.ndarray:
                name_bbox = 'turb_for_'
                for i, id in enumerate(lidar_id):
                    if i == 0:
                        beam_xyz =self._get_prob_cords(id)
                        aa = self.data.meas_cfg[id]['config']['az'].mean()
                        name_bbox += id
                    else:
                        name_bbox += '_and_'
                        name_bbox += id
                        beam_xyz = np.append(beam_xyz,
                                             self._get_prob_cords(id),
                                             axis = 0)
                        aa = np.append(aa, self.data.meas_cfg[id]['config']['az'].mean())
                avg_azimuth = aa.mean()
        except:
            raise ValueError('lidar id(s) does not exist !')

        ws = self.data.fmodel_cfg['wind_speed']
        wdir = self.data.fmodel_cfg['wind_from_direction']
        R_az = _rot_matrix(-avg_azimuth) #2D rot_matrix
        R_tb = _rot_matrix(wdir)
        CRS = {'x':'Relative coordinate, use inv(rot_matrix) to convert to abs',
               'y':'Relative coordinate, use inv(rot_matrix) to convert to abs',
               'z':'Absolute coordinate, coresponds to height above sea level',
               'rot_matrix':R_tb}
        # get probing coordinates
        # beam_xyz = self._get_prob_cords()

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

        self.data._cr8_bbox_dict(name_bbox, CRS,
                                 bbox_pts[:,0], bbox_pts[:,1], beam_xyz[:,2],
                                 t_coord,
                                 0, 0, 0, 0,
                                 self.x_res, self.y_res, self.z_res, t_res)
        return name_bbox

    def gen_turb_ffield(self,lidar_id,
                        ws=10, wdir=180, w=0, href=100, alpha=0.2):
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
        fmodel_cfg= {'flow_model':'PyConTurb',
                     'wind_speed':ws,
                     'upward_velocity':w,
                     'wind_from_direction':wdir,
                     'reference_height':href,
                     'shear_expornent':alpha,
                     }
        self.data._cr8_fmodel_cfg(fmodel_cfg)
        name_bbox = self._cr8_turbbbox(lidar_id)
        # self.data._cr8_empty_tfield_ds()

        _, y, z, _ = self.data._get_ffield_coords(name_bbox)
        spat_df = gen_spat_grid(y, z)
        T_tot = self.turbbox_time
        T_res = self.data.ffield_bbox_cfg[name_bbox]['t']['res']


        turb_df = gen_turb(spat_df, T=T_tot + T_res,
                           dt=T_res,
                           wsp_func = power_profile,
                           u_ref=ws, z_ref=href, alpha=alpha)

        self.data._cr8_3d_tfield_ds(name_bbox, turb_df)

        # self.data._upd8_tfield_ds(turb_df)
        self._to_4D_ds(name_bbox)


    def set_ivap_probing(self, lidar_id, sector_size, azimuth_mid, angular_res,
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

        # lidar_pos = self.data.meas_cfg[lidar_id]['position']

        # create measurement configuration dictionary
        self.data._upd8_meas_cfg(lidar_id, 'IVAP', az, el, rng, no_los, no_scans,
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
        self.data._cr8_probing_ds(lidar_id, az, el, rng, time)

        # calculate uncertainties

        # adding xyz (these will not have uncertainties)
        self._calc_xyz(lidar_id)

        # # create flow field bounding box dict
        self._cr8_ffield_bbox(lidar_id)

    def gen_unc_contributors(self, lidar_id, unc_cfg = None):

        no_los = self.data.meas_cfg[lidar_id]['config']['no_los']
        no_scans = self.data.meas_cfg[lidar_id]['config']['no_scans']
        if unc_cfg == None:
            corr_coef = self.data.meas_cfg[lidar_id]['uncertainty']['corr_coef']
            unc_cfg = self.data.meas_cfg[lidar_id]['uncertainty'].copy()
            del unc_cfg['corr_coef']
        else:
            corr_coef = unc_cfg['corr_coef']
            unc_cfg = unc_cfg.copy()
            del unc_cfg['corr_coef']




        # sample uncertainty contribution considering normal distribution
        # and add them to probing xr.DataSet
        for unc_term, cfg in unc_cfg.items():
            samples = gen_unc(np.full(no_los, cfg['mu']),
                              np.full(no_los, cfg['std']),
                              corr_coef, no_scans)
            self.data._add_unc(lidar_id, unc_term, samples.flatten())

        # update x,y,z coordinates of measurement points
        self._calc_xyz(lidar_id)

        # # update flow field bounding box dict
        self._cr8_ffield_bbox(lidar_id)


    # def gen_unc_contributors_old(self, corr_coef=0,
    #                          unc_cfg={'unc_az':{'mu':0, 'std':0.1},
    #                                   'unc_el':{'mu':0, 'std':0.1},
    #                                   'unc_rng':{'mu':0, 'std':10},
    #                                   'unc_est':{'mu':0, 'std':0.1}}):

    #     # store information in unc_cfg
    #     self.data.unc_cfg.update(unc_cfg)
    #     self.data.unc_cfg.update({'corr_coef':corr_coef})

    #     # sample uncertainty contribution considering normal distribution
    #     # and add them to probing xr.DataSet
    #     for unc_term, cfg in unc_cfg.items():
    #         samples = gen_unc(np.full(self.data.meas_cfg['no_los'], cfg['mu']),
    #                           np.full(self.data.meas_cfg['no_los'], cfg['std']),
    #                           corr_coef, self.data.meas_cfg['no_scans'])
    #         self.data._add_unc(unc_term, samples.flatten())

    #     # update x,y,z coordinates of measurement points
    #     self._calc_xyz()

    #     # update flow field bounding box dict
    #     self._cr8_ffield_bbox()

    def gen_plaw_ffield(self,lidar_id,
                        ws=10, wdir=180, w=0, href=100, alpha=0.2):
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


        try:
            if type(lidar_id) == str:
                z_coord= np.arange(self.data.ffield_bbox_cfg[lidar_id]['z']['min'],
                                   self.data.ffield_bbox_cfg[lidar_id]['z']['max'],
                                   self.data.ffield_bbox_cfg[lidar_id]['z']['res'])
            elif type(lidar_id) == list or type(lidar_id) == np.ndarray:
                z_min = np.empty(len(lidar_id))
                z_max = np.empty(len(lidar_id))
                z_res = np.empty(len(lidar_id))
                for i,id in enumerate(lidar_id):
                    z_min[i]= self.data.ffield_bbox_cfg[id]['z']['min']
                    z_max[i]= self.data.ffield_bbox_cfg[id]['z']['max']
                    z_res[i]= self.data.ffield_bbox_cfg[id]['z']['res']

                z_coord = np.arange(z_min.min() - z_res.min(),
                                    z_max.max() + z_res.min(), z_res.min())
        except:
            raise ValueError('lidar id(s) does not exist !')




        u, v, w = get_plaw_uvw(z_coord, href, ws, w, wdir, alpha)

        self.data._cr8_plfield_ds(lidar_id, u, v, w)

    def calc_los_speed(self, lidar_id):
        """Calcutes projection of wind speed on beam line-of-sight
        """

        # needs if lidars exist before proceeding forward


        if self.data.fmodel_cfg['flow_model'] == 'power_law':
            hmeas = self.data.probing.z.values
            u = self.data.ffield.u.isel(x=0,y=0).interp(z=hmeas)
            v = self.data.ffield.v.isel(x=0,y=0).interp(z=hmeas)
            w = self.data.ffield.w.isel(x=0,y=0).interp(z=hmeas)

        elif self.data.fmodel_cfg['flow_model'] == 'PyConTurb':
            # slowest possible way of pulling u,v,w
            # since it currently steps through each time stamp
            # and interpolates data
            time_prob = self.data.probing.time.values
            time_tbox = self.data.ffield.time.values
            R_tb = self.data.ffield_bbox_cfg['lidar']['CRS']['rot_matrix']

            # Normalize time to re-feed the turbulence
            time_norm = np.mod(time_prob, time_tbox.max())
            u = np.empty(len(time_norm))
            v = np.empty(len(time_norm))
            w = np.empty(len(time_norm))

            for i,t in enumerate(tqdm(time_norm, desc='Projecting LOS')):
                x = self.data.probing.x.isel(time = i).values
                y = self.data.probing.y.isel(time = i).values
                z = self.data.probing.z.isel(time = i).values

                xy = np.array([x,y]).dot(R_tb)

                u[i] = self.data.ffield.u.interp(time=t, x=xy[0], y=xy[1], z=z).values
                v[i] = self.data.ffield.v.interp(time=t, x=xy[0], y=xy[1], z=z).values
                w[i] = self.data.ffield.w.interp(time=t, x=xy[0], y=xy[1], z=z).values

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

    def _get_prob_cords(self,lidar_id):

        x = self.data.probing[lidar_id].x.values
        y = self.data.probing[lidar_id].y.values
        z = self.data.probing[lidar_id].z.values

        return np.array([x,y,z]).transpose()


    def _to_4D_ds(self, id):
        ws = self.data.fmodel_cfg['wind_speed']
        bbox_pts = bbox_pts_from_cfg(self.data.ffield_bbox_cfg[id])

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

        self.data._cr8_4d_tfield_ds(id, u_4d, v_4d, w_4d, x_coord, t)
