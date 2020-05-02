
import numpy as np
import xarray as xr
from math import degrees, atan2
# from .persistance import data


def generate_beam_coords(lidar_pos, meas_pt_pos):
    """
    Generates beam steering coordinates in spherical coordinate system.

    Parameters
    ----------
    lidar_pos : ndarray
        1D array containing data with `float` or `int` type corresponding to
        Northing, Easting and Height coordinates of a lidar.
        Coordinates unit is meter.
    meas_pt_pos : ndarray
        nD array containing data with `float` or `int` type corresponding to
        Northing, Easting and Height coordinates of a measurement point(s).
        Coordinates unit is meter.

    Returns
    -------
    beam_coords : ndarray
        nD array containing beam steering coordinates for given measurement points.
        Coordinates have following structure [azimuth, elevation, range].
        Azimuth and elevation angles are given in degree.
        Range unit is meter.
    """
    # testing if  meas_pt has single or multiple measurement points
    if len(meas_pt_pos.shape) == 2:
        x_array = meas_pt_pos[:, 0]
        y_array = meas_pt_pos[:, 1]
        z_array = meas_pt_pos[:, 2]
    else:
        x_array = np.array([meas_pt_pos[0]])
        y_array = np.array([meas_pt_pos[1]])
        z_array = np.array([meas_pt_pos[2]])


    # calculating difference between lidar_pos and meas_pt_pos coordiantes
    dif_xyz = np.array([lidar_pos[0] - x_array, lidar_pos[1] - y_array, lidar_pos[2] - z_array])

    # distance between lidar and measurement point in space
    distance_3D = np.sum(dif_xyz**2,axis=0)**(1./2)

    # distance between lidar and measurement point in a horizontal plane
    distance_2D = np.sum(np.abs([dif_xyz[0],dif_xyz[1]])**2,axis=0)**(1./2)

    # in radians
    azimuth = np.arctan2(x_array-lidar_pos[0], y_array-lidar_pos[1])
    # conversion to metrological convention
    azimuth = (360 + azimuth * (180 / np.pi)) % 360

    # in radians
    elevation = np.arccos(distance_2D / distance_3D)
    # conversion to metrological convention
    elevation = np.sign(z_array - lidar_pos[2]) * (elevation * (180 / np.pi))

    beam_coord = np.transpose(np.array([azimuth, elevation, distance_3D]))

    return beam_coord

def spher2cart(azimuth, elevation, radius):
    """Converts spherical coordinates to Cartesian coordinates

    Parameters
    ----------
    azimuth : numpy
        Array containing azimuth angles
    elevation : numpy
        Array containing elevation angles
    radius : numpy
        Array containing radius angles

    Returns
    -------
    x : numpy
        Array containing x coordinate values.
    y : numpy
        Array containing y coordinate values.
    z : numpy
        Array containing z coordinate values.

    Raises
    ------
    TypeError
        If dimensions of inputs are not the same
    """

    azimuth = 90 - azimuth # converts to 'Cartesian' angle
    azimuth = np.radians(azimuth) # converts to radians
    elevation = np.radians(elevation)

    try:
        x = radius * np.cos(azimuth) * np.cos(elevation)
        y = radius * np.sin(azimuth) * np.cos(elevation)
        z = radius * np.sin(elevation)
        return x, y, z
    except:
        raise TypeError('Dimensions of inputs are not the same!')


def trajectory2displacement(lidar_pos, trajectory, rollover = True):
    """
    Converts trajectory described in a Cartesian coordinate system to
    angular positions of the lidar scanner heads.

    Parameters
    ----------
    lidar_pos : ndarray
        nD array containing the lidar position in a Cartesian
        coordinate system.
    trajectory : ndarray
        nD array containing trajectory points in a Cartesian
        coordinateys system.
    rollover : boolean
        Indicates whether the lidar motion controller has
        a rollover capability.
        The default value is set to True.

    Returns
    -------
    angles_start : ndarray
        nD array containing the starting position of the scanner head.
    angles_stop : ndarray
        nD array containing the ending position of the scanner head.
    angular_displacement : ndarray
        nD array containing angular displacements that the motion system
        needs to perform when moving from one to another trajectory point.
    """

    NO_DIGITS = 2
    angles_start = generate_beam_coords(lidar_pos, np.roll(trajectory, 1, axis = 0))[:, (0,1)]
    angles_stop = generate_beam_coords(lidar_pos, np.roll(trajectory, 0, axis = 0))[:, (0,1)]

    angular_displacement = abs(angles_start - angles_stop)


    ind_1 = np.where((angular_displacement[:, 0] > 180) &
                     (abs(360 - angular_displacement[:, 0]) <= 180))

    ind_2 = np.where((angular_displacement[:, 0] > 180) &
                     (abs(360 - angular_displacement[:, 0]) > 180))

    angular_displacement[:, 0][ind_1] = 360 - angular_displacement[:, 0][ind_1]
    angular_displacement[:, 0][ind_2] = 360 + angular_displacement[:, 0][ind_2]


    ind_1 = np.where((angular_displacement[:, 1] > 180) &
                     (abs(360 - angular_displacement[:, 1]) <= 180))
    ind_2 = np.where((angular_displacement[:, 1] > 180) &
                     (abs(360 - angular_displacement[:, 1]) > 180))
    angular_displacement[:, 1][ind_1] = 360 - angular_displacement[:, 1][ind_1]
    angular_displacement[:, 1][ind_2] = 360 + angular_displacement[:, 1][ind_2]
    return np.round(angles_start, NO_DIGITS), \
           np.round(angles_stop, NO_DIGITS), \
           np.abs(angular_displacement)



def displacement2time(displacement, Amax, Vmax):
    """
    Calculates minimum move time to perform predefined angular motions.

    Parameters
    ----------
    displacement : ndarray
        nD array containing angular displacements
        that a motion system needs to perform.
        The displacements unit depends on the units of Amax and Vmax.
        Typically the unit is in degrees.
    Amax : float
        Maximum permited acceleration of the motion system.
        Typically the unit for Amax is degrees/s^2 .
    Vmax : float
        Maximum permited velocity of the motion system (aka rated speed).
        Typically the unit for Vmas is degree/s .

    Returns
    -------
    move_time : ndarray
        nD array containing minimum move time to perform the diplacements
        The unit of move_time elements depends on the input parameters.
        Typically the unit is s (seconds).

    Notes
    -----
    Equations used to calculate move time are based on [1],
    assuming an infinite jerk.

    References
    ----------
    .. [1] Peters R. D.: Ideal Lift Kinematics: Complete Equations for
        Plotting Optimum Motion  Elevator Technology 6,
        Proceedings of ELEVCON’95, 1995.
    """

    move_time = np.empty((len(displacement),), dtype=float)

    # find indexes for which the scanner head
    # will reach maximum velocity (i.e. rated speed)
    index_a = np.where(displacement > (Vmax**2) / Amax)

    # find indexes for which the scanner head
    # will not reach maximum velocity (i.e. rated speed)
    index_b = np.where(displacement <= (Vmax**2) / Amax)

    move_time[index_a] = displacement[index_a] / Vmax + Vmax / Amax
    move_time[index_b] = 2 * np.sqrt(displacement[index_b] / Amax)


    return move_time


def project2los(u,v,w, azimuth, elevation, ignore_elevation = True):
    """Projects wind vector to line-of-sight

    Parameters
    ----------
    u : numpy
        u component of the wind
    v : numpy
        v component of the wind
    w : numpy
        upward velocity of the wind
    azimuth : numpy
        azimuth angle of the beam
    elevation : numpy
        elevation angle of the beam
    ignore_elevation : bool, optional
        To assume horizontal beam or not, by default True

    Returns
    -------
    los : numpy
        Projected wind vector to LOS, i.e. radial/los wind speed
    Raises
    ------
    TypeError
        If dimensions of inputs are not the same
    """

    # handles both single values as well arrays
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    try:
        if ignore_elevation:
            los = u * np.sin(azimuth) + v * np.cos(azimuth)
        else:
            los = u * np.sin(azimuth) * np.cos(elevation) + \
                v * np.cos(azimuth) * np.cos(elevation) + \
                w * np.sin(elevation)

        return los
    except:
        raise TypeError('Dimensions of inputs are not the same!')



def ivap_rc(los, azimuth, ax = 0):
    """Calculates u and v components by using IVAP algo on set of LOS speed
    measurements acquired using PPI scans.

    Parameters
    ----------
    los : numpy
        Array containing LOS speed measurements
    azimuth : numpy
        Array containing azimuth angles corresponding to LOS speed measurements
    ax : int
        Along which axes should calculation take place, by default 0

    Returns
    -------
    u : numpy
        Array of reconstracted u component of the wind
    v : numpy
        Array of reconstracted v component of the wind
    wind_speed: numpy
        Array of reconstracted horizontal wind speed
    """
    azimuth = np.radians(azimuth)

    A = ((np.sin(azimuth))**2).sum(axis=ax)
    B = ((np.cos(azimuth))**2).sum(axis=ax)
    C = (np.sin(azimuth)*np.cos(azimuth)).sum(axis=ax)
    D = A*B - C**2
    E = (los*np.cos(azimuth)).sum(axis=ax)
    F = (los*np.sin(azimuth)).sum(axis=ax)

    v = (E*A - F*C) / D
    u = (F*B - E*C) / D
    wind_speed = np.sqrt(u**2 + v**2)

    return u, v, wind_speed

def dd_rc_single(los, azimuth):
    """Calculates u and v components by using IVAP algo on set of LOS speed
    measurements acquired using PPI scans.

    Parameters
    ----------
    los : numpy
        Array containing LOS speed measurements
    azimuth : numpy
        Array containing azimuth angles corresponding to LOS speed measurements

    Returns
    -------
    u : numpy
        Array of reconstracted u component of the wind
    v : numpy
        Array of reconstracted v component of the wind
    wind_speed: numpy
        Array of reconstracted horizontal wind speed
    """

    azimuth = np.radians(azimuth)

    R_ws = np.array([[np.sin(azimuth[0]), np.cos(azimuth[0])],
                     [np.sin(azimuth[1]), np.cos(azimuth[1])]])

    u, v = np.dot(np.linalg.inv(R_ws),los)
    wind_speed = np.sqrt(u**2 + v**2)

    wind_direction = (90 - degrees(atan2(-v, -u))) % 360

    return u, v, wind_speed, wind_direction



def td_rc_single(los, azimuth, elevation):
    """Calculates u and v components by using IVAP algo on set of LOS speed
    measurements acquired using PPI scans.

    Parameters
    ----------
    los : numpy
        Array containing LOS speed measurements
    azimuth : numpy
        Array containing azimuth angles corresponding to LOS speed measurements

    Returns
    -------
    u : numpy
        Array of reconstracted u component of the wind
    v : numpy
        Array of reconstracted v component of the wind
    wind_speed: numpy
        Array of reconstracted horizontal wind speed
    """
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    R_ws = np.array([[np.cos(elevation[0]) * np.sin(azimuth[0]), np.cos(elevation[0]) * np.cos(azimuth[0]), np.sin(elevation[0])],
                     [np.cos(elevation[1]) * np.sin(azimuth[1]), np.cos(elevation[1]) * np.cos(azimuth[1]), np.sin(elevation[1])],
                     [np.cos(elevation[2]) * np.sin(azimuth[2]), np.cos(elevation[2]) * np.cos(azimuth[2]), np.sin(elevation[2])]])

    u, v, w = np.dot(np.linalg.inv(R_ws),los)
    wind_speed = np.sqrt(u**2 + v**2)
    wind_direction = (90 - degrees(atan2(-v, -u))) % 360

    return u,v,w, wind_speed, wind_direction


def dd_rc_array(los, azimuth, elevation, rc_type = 1):

    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    if rc_type == 0:
        R = np.array([[np.sin(azimuth[0]), np.sin(azimuth[1])],
                      [np.cos(azimuth[0]), np.cos(azimuth[1])]])
    else:
        R = np.array([[np.sin(azimuth[0])*np.cos(elevation[0]),np.sin(azimuth[1])*np.cos(elevation[1])],
                      [np.cos(azimuth[0])*np.cos(elevation[0]),np.cos(azimuth[1])*np.cos(elevation[1])]])

    inv_R = np.linalg.inv(R.T)

    uvw_array = np.einsum('ijk,ik->ij', inv_R,los)

    V_h_array = np.sqrt(uvw_array[:,0]**2 + uvw_array[:,1]**2)

    wind_dir_array = (90 - np.arctan2(-uvw_array[:,1],-uvw_array[:,0])* (180 / np.pi)) % 360
    return V_h_array, wind_dir_array


def td_rc_array(los, azimuth, elevation, rc_type = 1):

    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)

    if rc_type == 0:
        R = np.array([[np.sin(azimuth[0]), np.sin(azimuth[1]), np.sin(azimuth[2])],
                      [np.cos(azimuth[0]), np.cos(azimuth[1]), np.cos(azimuth[2])],
                      [np.sin(elevation[0]), np.sin(elevation[1]), np.sin(elevation[2])]])
    else:
        R = np.array([[np.sin(azimuth[0])*np.cos(elevation[0]),np.sin(azimuth[1])*np.cos(elevation[1]),np.sin(azimuth[2])*np.cos(elevation[2])],
                      [np.cos(azimuth[0])*np.cos(elevation[0]),np.cos(azimuth[1])*np.cos(elevation[1]),np.cos(azimuth[2])*np.cos(elevation[2])],
                      [np.sin(elevation[0]),                   np.sin(elevation[1]),                   np.sin(elevation[2])                   ]])

    inv_R = np.linalg.inv(R.T)

    uvw_array = np.einsum('ijk,ik->ij', inv_R,los)

    V_h_array = np.sqrt(uvw_array[:,0]**2 + uvw_array[:,1]**2)

    wind_dir_array = (90 - np.arctan2(-uvw_array[:,1],-uvw_array[:,0])* (180 / np.pi)) % 360
    return V_h_array, wind_dir_array


def move2time(displacement, Amax, Vmax):
    """
    Calculates minimum move time to perform predefined angular motions.
    """

    max_acc_move = (Vmax**2) / Amax

    if displacement > max_acc_move:
        return displacement/Vmax + Vmax/Amax
    else:
        return 2*np.sqrt(displacement/Amax)



def add_xyz(ds, lidar_pos):
    x,y,z = spher2cart(ds.az +ds.unc_az,
                       ds.el +ds.unc_el,
                       ds.dis+ds.unc_dis)
    ds.x.values = x.values + lidar_pos[0]
    ds.y.values = y.values + lidar_pos[1]
    ds.z.values = z.values + lidar_pos[2]

    return ds

def get_plaw_uvw(height, ref_height=100, wind_speed=10,
                 w=0, wind_dir=180, shear_exponent=0.2):
    u = - wind_speed * np.sin(np.radians(wind_dir))
    v = - wind_speed * np.cos(np.radians(wind_dir))
    gain = (height / ref_height)**shear_exponent
    u_new = gain * u
    v_new = gain * v
    w_new = np.full(len(u_new), w)

    return u_new, v_new, w_new


def sliding_window_slicing(a, no_items, item_type=0):
    """This method perfoms sliding window slicing of numpy arrays

    Parameters
    ----------
    a : numpy
        An array to be slided in subarrays
    no_items : int
        Number of sliced arrays or elements in sliced arrays
    item_type: int
        Indicates if no_items is number of sliced arrays (item_type=0) or
        number of elements in sliced array (item_type=1), by default 0

    Return
    ------
    numpy
        Sliced numpy array
    """
    if item_type == 0:
        no_slices = no_items
        no_elements = len(a) + 1 - no_slices
        if no_elements <=0:
            raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))
    else:
        no_elements = no_items
        no_slices = len(a) - no_elements + 1
        if no_slices <=0:
            raise ValueError('Sliding slicing not possible, no_items is larger than ' + str(len(a)))

    subarray_shape = a.shape[1:]
    shape_cfg = (no_slices, no_elements) + subarray_shape
    strides_cfg = (a.strides[0],) + a.strides
    as_strided = np.lib.stride_tricks.as_strided #shorthand
    return as_strided(a, shape=shape_cfg, strides=strides_cfg)


def _rot_matrix(wind_dir):

    azimuth = wind_dir - 90  # it is revers to permutate origina x&y axis
    azimuth = np.radians(azimuth) # converts to radians
    c, s = np.cos(azimuth), np.sin(azimuth)
    R = np.array([[c, s], [-s, c]])
    return R


def bbox_pts_from_array(a):

    bbox_pts = np.full((4,2), np.nan)

    x_min = a[:,0].min()
    y_min = a[:,1].min()
    x_max = a[:,0].max()
    y_max = a[:,1].max()
    bbox_pts[0] = np.array([x_min,y_min])
    bbox_pts[1] = np.array([x_min,y_max])
    bbox_pts[2] = np.array([x_max,y_max])
    bbox_pts[3] = np.array([x_max,y_min])

    return bbox_pts

def bbox_pts_from_cfg(cfg):

    bbox_pts = np.full((4,2), np.nan)
    bbox_pts[0] = np.array([cfg['x']['min'],
                            cfg['y']['min']])
    bbox_pts[1] = np.array([cfg['x']['min'],
                            cfg['y']['max']])
    bbox_pts[2] = np.array([cfg['x']['max'],
                            cfg['y']['max']])
    bbox_pts[3] = np.array([cfg['x']['max'],
                            cfg['y']['min']])
    return bbox_pts

def calc_mean_step(a):
    """
    Calculates mean step between consecutive elements in 1D array

    Parameters
    ----------
    a : numpy
        1D array

    Returns
    -------
    float
        Average step between consecutive elements
    """

    a.sort()
    steps = np.abs(np.roll(a,1) - a)[1:]
    return steps.mean()


def safe_execute(default, exception, function, *args):
    """
    from: https://stackoverflow.com/questions/36671077/one-line-exception-handling
    """
    try:
        return function(*args)
    except exception:
        return default