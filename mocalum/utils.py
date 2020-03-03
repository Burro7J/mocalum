
import numpy as np
from math import degrees, atan2


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



def wind_vector_to_los(u,v,w, azimuth, elevation, ignore_elevation = True):
    """[summary]

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
        [description], by default True

    Returns
    -------
    los : numpy
        radial velocity or LOS speed
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




def ivap(los, azimuth):
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

    A = ((np.sin(azimuth))**2).sum()
    B = ((np.cos(azimuth))**2).sum()
    C = (np.sin(azimuth)*np.cos(azimuth)).sum()
    D = A*B - C**2
    E = (los*np.cos(azimuth)).sum()
    F = (los*np.sin(azimuth)).sum()

    v = (E*A - F*C) / D
    u = (F*B - E*C) / D
    wind_speed = np.sqrt(u**2 + v**2)

    return u, v, wind_speed


def dual_Doppler_rc(los, azimuth):
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



def triple_Doppler_rc(los, azimuth, elevation):
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