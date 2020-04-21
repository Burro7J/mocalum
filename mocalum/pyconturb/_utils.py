# -*- coding: utf-8 -*-
"""utility functions
"""
import os

import numpy as np
import pandas as pd
import scipy.interpolate as sciint


_spat_rownames = ['k', 'x', 'y', 'z']  # row names of spatial df
_DEF_KWARGS = {'u_ref': 0, 'z_ref': 90, 'alpha': 0.2, 'turb_class': 'A',
               'l_c': 340.2}  # lc for coherence ***NOTE THESE OVERWRITE FUNCTION DEFS
_HAWC2_BIN_FMT = '<f'  # HAWC2 binary turbulence datatype
_HAWC2_TURB_COOR = {'u': -1, 'v': -1, 'w': 1}  # hawc2 turb xyz to uvw


def clean_turb(spat_df, all_spat_df, turb_df, decimals=10):
    """Remove the columns we don't return and rename the rest correctly. Will check only
    to decimals places when removing duplicates (data unchanged)."""
    # drop the columns that aren't in spat_df (use isclose for float comparisons)
    drop_cols = all_spat_df.apply(lambda col:
                                  not np.all(np.isclose(col.values, spat_df.values.T),
                                             axis=1).sum()).values
    turb_df.drop(turb_df.columns[drop_cols], axis=1, inplace=True)
    all_spat_df.drop(all_spat_df.columns[drop_cols], axis=1, inplace=True)
    # get unique locs in spat_df (need rounding logic here for dropping float duplicates)
    spat_xyz = spat_df.loc[['x', 'y', 'z']]
    spat_xyz = spat_xyz.T.loc[~spat_xyz.T.apply(np.round,
                                                args=[decimals]).duplicated()].T
    # for each column in all_spat_df, find the correct name using spat_df and rename it
    for colname in all_spat_df:
        col = all_spat_df[colname]
        k, x, y, z = col.values
        pid = np.all(np.isclose(np.array([x, y, z]), spat_xyz.values.T),
                     axis=1).argmax()  # use np.isclose for float comparisons
        new_name = f'{"uvw"[int(k)]}_p{pid}'
        turb_df.rename(columns={colname: new_name}, inplace=True)
    # order according to spat_df
    col_names = []
    for colname in spat_df:
        col = spat_df[colname]
        k, x, y, z = col.values
        pid = np.all(np.isclose(np.array([x, y, z]), spat_xyz.values.T),
                     axis=1).argmax()  # use np.isclose for float comparisons
        col_names.append(f'{"uvw"[int(k)]}_p{pid}')
    return turb_df[col_names]


def combine_spat_con(spat_df, con_tc, drop_duplicates=True, decimals=10):
    """Add constraining points in TimeConstraint to spat_df. NOTE constraints must come
    first or gen_turb will break. ALSO keep the data rows! Need them for corr. ALSO this
    function catches duplicates by first rounding to decimals places.
    """
    con_spat_df = con_tc.get_spat().add_suffix('_con')
    if (set(spat_df.columns).intersection(set(con_spat_df.columns)) and (con_tc.size)):
        raise ValueError('Prohibited spat_df/con_tc column names! No column in spat_df' +
                         ' may have the same name as one on con_tc with "_con" ' +
                         'appended.')
    comb_df = pd.concat((con_spat_df, spat_df), axis=1)
    if drop_duplicates:
        # need to round before dropping to prevent machine-precision floats being "uniq"
        comb_df = comb_df.T.loc[~comb_df.T.apply(np.round,
                                                 args=[decimals]).duplicated()].T
    return comb_df


def df_to_h2turb(turb_df, spat_df, path, prefix=''):
    """ksec3d-style turbulence dataframe to binary files for hawc2

    Notes
    -----
    * The turbulence must have been generated on a y-z grid.
    * The naming convention must be 'u_p0', 'v_p0, 'w_p0', 'u_p1', etc.,
       where the point indices proceed vertically along z before horizontally
       along y.
    """
    nx = turb_df.shape[0]  # turbulence dimensions for reshaping
    ny = len(set(spat_df.loc['y'].values))
    nz = len(set(spat_df.loc['z'].values))
    # make and save binary files for all three components
    for c in 'uvw':
        arr = turb_df.filter(regex=f'{c}_', axis=1).values.reshape((nx, ny, nz))
        bin_path = os.path.join(path, f'{prefix}{c}.bin')
        with open(bin_path, 'wb') as bin_fid:
            arr.astype(np.dtype(_HAWC2_BIN_FMT)).tofile(bin_fid)
    return


def gen_spat_grid(y, z, comps=[0, 1, 2]):
    """Generate spat_df (all turbulent components and grid defined by x and z)

    Notes
    -----
    0=u is downwind, 2=w is vertical and 1=v is lateral (right-handed
    coordinate system).
    """
    ys, zs = np.meshgrid(y, z)  # make a meshgrid
    ks = np.array(comps, dtype=int)  # sanitizing
    xs = np.zeros_like(ys)  # all points in a plane
    col_names = [f'{"uvw"[k]}_p{ip}' for ip in range(xs.size) for k in ks]
    spat_arr = np.c_[np.tile(comps, xs.size),
                     np.repeat(np.c_[xs.T.ravel(), ys.T.ravel(), zs.T.ravel()],
                               ks.size, axis=0)].T  # create array using numpy
    return pd.DataFrame(spat_arr, index=_spat_rownames, columns=col_names)


def get_freq(**kwargs):
    """get frequency array"""
    n_t = int(np.ceil(kwargs['T'] / kwargs['dt']))
    t = np.arange(n_t) * kwargs['dt']
    n_f = n_t // 2 + 1
    freq = np.arange(n_f) / kwargs['T']
    return t, freq


def h2turb_to_arr(spat_df, path):
    """raw-load a hawc2 turbulent binary file to numeric array"""
    ny, nz = pd.unique(spat_df.loc['y']).size, pd.unique(spat_df.loc['z']).size
    bin_arr = np.fromfile(path, dtype=np.dtype(_HAWC2_BIN_FMT))
    nx = bin_arr.size // (ny * nz)
    if (nx * ny * nz) != bin_arr.size:
        raise ValueError('Binary file size does not match spat_df!')
    bin_arr.shape = (nx, ny, nz)
    return bin_arr


def h2turb_to_df(spat_df, path, prefix=''):
    """load a hawc2 binary file into a pandas datafram with transform to uvw"""
    turb_df = pd.DataFrame()
    for c in 'uvw':
        comp_path = os.path.join(path, f'{prefix}{c}.bin')
        arr = h2turb_to_arr(spat_df, comp_path)
        nx, ny, nz = arr.shape
        comp_df = pd.DataFrame(arr.reshape(nx, ny*nz)).add_prefix(f'{c}_p')
        turb_df = turb_df.join(comp_df, how='outer')
    turb_df = turb_df[[f'{c}_p{i}' for i in range(2) for c in 'uvw']]
    return turb_df


def is_line(points):
    """see if the points given in points fall along a single line"""
    # first rotate by first pair to remove issue with vertical lines
    theta = np.arctan2((points[1, 1] - points[0, 1]), (points[1, 0] - points[0, 0]))
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot_points = points @ R
    # then do least-squares fit
    a = np.c_[rot_points[:, 0], np.ones(rot_points.shape[0])]
    b = rot_points[:, 1]
    x = np.linalg.lstsq(a, b, rcond=0)[0]
    if np.isclose(0., np.linalg.norm(a@x-b)):
        return True
    else:
        return False


def interpolator(points, values, xi):
    """Interpolate points in D dimensions.

    This function is based upon scipy's griddata function, but it has been expanded to
    handle single points and points that fall in a line. It is used in certain profile
    functions for the wind speed, turbulence standard deviation, and spectrum for
    interpolating the values calculated from ``con_tc`` to the new requested points.
    See examples of its usage in the Interpolator example.

    Parameters
    ----------
    points : ndarray of floats, shape (n, D)
        Data point coordinates. Can either be an array of
        shape (n, D), or a tuple of `D` arrays, each with shape `n`.
    values : ndarray of float or complex, shape (n,)
        Data values.
    xi : 2-D ndarray of float or tuple of 1-D array, shape (M, D)
        Points at which to interpolate data.

    Returns
    -------
    intp_val : ndarray
        Array of interpolated values, shape M.
    """
    ndim = 2  # function assumes we only have 2 interpolating dimensions (y and z)
    san_vars = [points, xi]  # variables we need to sanitize
    for i in range(len(san_vars)):
        if isinstance(san_vars[i], tuple):  # if it's a tuple, array it
            san_vars[i] = np.atleast_2d(san_vars[i])  # returns an ndim x np array
        elif isinstance(san_vars[i], (np.ndarray, list)):  # allow a list to be passed in
            san_vars[i] = np.atleast_2d(san_vars[i]).T  # returns an ndim x np array
        else:
            raise ValueError(f'Inputs points and xi can only be a tuple, list or numpy '
                             'array!')
        san_vars[i] = san_vars[i].T  # griddata takes np x ndim, so transpose
    points, xi = san_vars  # reassing points and xi
    # if one point, just return the values
    if (points.size == ndim):
        return sciint.griddata(points, values, xi, method='nearest')
    # if it's a line, use numpy's 1d interpolation
    if is_line(points):  # points lie in a line
        theta = np.arctan2((points[1, 1] - points[0, 1]), (points[1, 0] - points[0, 0]))
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rot_points, rot_xi = points @ R, xi @ R
        intp_val = np.interp(rot_xi[:, 0], rot_points[:, 0], values)
    else:  # more than 1 pt, not in a line (so 3+ points)
        intp_val = sciint.griddata(points, values, xi)  # 1st call griddata w/linear
        outside = np.isnan(intp_val)  # points outside grid will be nans
        out_intp = sciint.griddata(points, values, xi[outside],
                                   method='nearest')  # reeval outside grid with nearest
        intp_val[outside] = out_intp
    return intp_val


def make_hawc2_input(turb_dir, spat_df, **kwargs):
    """return strings for the hawc2 input files
    """
    # string of center position
    z_ref = kwargs['z_ref']
    str_cntr_pos0 = '  center_pos0             0.0 0.0 ' + \
        f'{-z_ref:.1f} ; hub height\n'

    # string for mann model block
    T, dt = kwargs['T'], kwargs['dt']
    y, z = set(spat_df.loc['y'].values), set(spat_df.loc['z'].values)
    n_x, du = int(np.ceil(T / dt)), dt * kwargs['u_ref']
    n_y, dv = len(y), (max(y) - min(y)) / (len(y) - 1)
    n_z, dw = len(z), (max(z) - min(z)) / (len(z) - 1)
    str_mann = '  begin mann ;\n' + \
               f'    filename_u {turb_dir}/u.bin ; \n' + \
               f'    filename_v {turb_dir}/v.bin ; \n' + \
               f'    filename_w {turb_dir}/w.bin ; \n' + \
               f'    box_dim_u {n_x:.0f} {du:.1f} ; \n' + \
               f'    box_dim_v {n_y:.0f} {dv:.1f} ; \n' + \
               f'    box_dim_w {n_z:.0f} {dw:.1f} ; \n' + \
               f'    dont_scale 1 ; \n' + \
               '  end mann '

    # string for output
    pts_df = spat_df.loc[['x', 'y', 'z'], spat_df.loc['k'] == 0]
    str_output = ''
    for col in pts_df.columns:
        x_p = -pts_df.loc['y', col]
        y_p = -pts_df.loc['x', col]
        z_p = -pts_df.loc['z', col]
        i_p = int(col.split('p')[-1])
        str_output += f'  wind free_wind 1 {x_p:.1f} {y_p:.1f} {z_p:.1f}' + \
            f' # wind_p{i_p} ; \n'

    return str_cntr_pos0, str_mann, str_output


def rotate_time_series(ux, uy, uz):
    """Yaw and pitch time series so v- and w-directions have zero mean

        Args:
            ux (numpy array): array of x-sonic velocity
            uy (numpy array): array of y-sonic velocity
            uz (numpy array): array of z-sonic velocity

        Returns:
            x_rot (numpy array): [n_t x 3] array of rotated data (yaw+pitch)
            x_yaw (numpy array): [n_t x 3] array of rotated data (yaw)
    """

    # return all NaNs if any component is all nan values
    if all(np.isnan(ux)) * all(np.isnan(uy)) * all(np.isnan(uz)):
        u = np.zeros(ux.shape)
        v = np.zeros(uy.shape)
        w = np.zeros(uz.shape)
        u[:] = np.nan
        v[:] = np.nan
        w[:] = np.nan

    # if at least one data point in all three components
    else:

        # combine velocities into array
        x_raw = np.concatenate((ux.reshape(ux.size, 1),
                                uy.reshape(ux.size, 1),
                                uz.reshape(ux.size, 1)), axis=1)

        # interpolate out any NaN values
        for i_comp in range(x_raw.shape[1]):
            x = x_raw[:, i_comp]
            idcs_all = np.arange(x.size)
            idcs_notnan = np.logical_not(np.isnan(x))
            x_raw[:, i_comp] = np.interp(idcs_all, idcs_all[idcs_notnan], x[idcs_notnan])

        # rotate through yaw angle
        theta = np.arctan(np.nanmean(x_raw[:, 1]) / np.nanmean(x_raw[:, 0]))
        A_yaw = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0],
                          [0, 0, 1]])
        x_yaw = x_raw @ A_yaw

        # rotate through pitch angle
        phi = np.arctan(np.nanmean(x_yaw[:, 2]) / np.nanmean(x_yaw[:, 0]))
        A_pitch = np.array([[np.cos(phi), 0, -np.sin(phi)],
                            [0, 1, 0],
                            [np.sin(phi), 0, np.cos(phi)]])
        x_rot = x_yaw @ A_pitch

        # if u is negative, we need to rotate 180 degrees around the y asix
        if x_rot[:, 0].sum() < 0:
            alpha = np.pi
            A_rot = np.array([[np.cos(alpha), 0, -np.sin(alpha)],
                              [0, 1, 0],
                              [np.sin(alpha), 0, np.cos(alpha)]])
            x_rot = x_rot @ A_rot

        # define rotated velocities
        u = x_rot[:, 0]
        v = x_rot[:, 1]
        w = x_rot[:, 2]

    return u, v, w
