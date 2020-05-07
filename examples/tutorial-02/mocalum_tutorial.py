import matplotlib.pyplot as plt
from matplotlib.pyplot import Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from numpy.linalg import inv as inv

def plot_sd_scan_setup(lidar_id, mc_obj):
    """
    Plots 2D geometry of single lidar scan

    Parameters
    ----------
    lidar_id : str
        Id of lidar
    mc_obj : mocalum
        Instance of mocalum class
    """

    no_los = mc_obj.data.meas_cfg[lidar_id]['config']['no_los']
    meas_pts= mc_obj._get_prob_cords(lidar_id)[:no_los]

    lidar_pos = mc_obj.data.meas_cfg[lidar_id]['position']

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.grid()
    plt.scatter(meas_pts[:,0], meas_pts[:,1], c="blue",
                label='measurements',zorder=10)

    for i,pt in enumerate(meas_pts):
        if i==0:
            plt.plot([lidar_pos[0],pt[0]],[lidar_pos[1],pt[1]],
                     c='black',alpha=0.4, label='beam')
        else:
            plt.plot([lidar_pos[0],pt[0]],[lidar_pos[1],pt[1]],
                     c='black',alpha=0.4)


    plt.scatter(lidar_pos[0],lidar_pos[1], c="green", label=lidar_id,zorder=150)
    ax.set_aspect('equal')
    plt.legend(loc="upper left")
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')

    plt.show()

def plot_md_scan_setup(lidar_ids, mc_obj):
    """
    Plots 2D geometry of multi-lidar scan

    Parameters
    ----------
    lidar_ids: list
        List of strings corresponding to lidar ids
    mc_obj : mocalum
        Instance of mocalum class
    """


    no_los = mc_obj.data.meas_cfg[lidar_ids[0]]['config']['no_los']
    meas_pts= mc_obj._get_prob_cords(lidar_ids[0])[:no_los]

    lidar_pos = []
    for id in lidar_ids:
        lidar_pos += [mc_obj.data.meas_cfg[id]['position']]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.grid()
    plt.scatter(meas_pts[:,0], meas_pts[:,1], c="blue",
                label='measurements',zorder=10)
    colors = ["green", "orange", "purple"]
    for j,id in enumerate(lidar_ids):
        plt.scatter(lidar_pos[j][0],
                    lidar_pos[j][1], c=colors[j], label=id,zorder=150)
        for i,pt in enumerate(meas_pts):
            if i==0 and j==0:
                plt.plot([lidar_pos[j][0],pt[0]],[lidar_pos[j][1],pt[1]],
                        c='black',alpha=0.4, label='beam')
            else:
                plt.plot([lidar_pos[j][0],pt[0]],[lidar_pos[j][1],pt[1]],
                        c='black',alpha=0.4)

    ax.set_aspect('equal')
    plt.legend(loc="lower right")
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')

    plt.show()


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

def bbox_pts_from_cfg(cfg):
    """
    Creates 2D bounding box points from bounding box config dictionary

    Parameters
    ----------
    cfg : dict
        Bounding box dictionary

    Returns
    -------
    numpy
        Numpy array of shape (2,2) corresponding to 4 corners of 2D bounding box
    """

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


def plot_bbox(mc_obj):
    """
    Plots 2D geometry of lidar scan and flow field box

    Parameters
    ----------
    mc_obj : mocalum
        Instance of mocalum class
    """

    flow_id = mc_obj.data.ffield.generator


    lidar_ids = mc_obj.data.bbox_ffield[flow_id]['linked_lidars']


    no_los = mc_obj.data.meas_cfg[lidar_ids[0]]['config']['no_los']
    meas_pts= mc_obj._get_prob_cords(lidar_ids[0])[:no_los]

    bbox_pts = bbox_pts_from_cfg(mc_obj.data.bbox_ffield[flow_id])
    diag = np.abs(bbox_pts[0]-bbox_pts[2]).max()
    R_tb = mc_obj.data.bbox_ffield[flow_id]['CRS']['rot_matrix']
    min_bbox_pts = bbox_pts.dot(inv(R_tb))
    bbox_c = min_bbox_pts.mean(axis = 0)

    wind_dir = mc_obj.data.fmodel_cfg['wind_from_direction']

    flowbox = Polygon(min_bbox_pts,alpha=0.4, color='grey', label="flow field bbox")

    wind_dir_pt = spher2cart(wind_dir,0,diag/2)[:2]

    lidar_pos = []
    for id in lidar_ids:
        lidar_pos += [mc_obj.data.meas_cfg[id]['position']]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.grid()


    plt.arrow(bbox_c[0], bbox_c[1],-wind_dir_pt[0],-wind_dir_pt[1],
              width=8,color="red", label='wind',zorder=50)
    plt.scatter(meas_pts[:,0], meas_pts[:,1], c="blue",
                label='measurements',zorder=10)
    colors = ["green", "orange", "purple"]
    for j,id in enumerate(lidar_ids):
        plt.scatter(lidar_pos[j][0],
                    lidar_pos[j][1], c=colors[j], label=id,zorder=150)
        for i,pt in enumerate(meas_pts):
            if i==0 and j==0:
                plt.plot([lidar_pos[j][0],pt[0]],[lidar_pos[j][1],pt[1]],
                        c='black',alpha=0.4, label='beam')
            else:
                plt.plot([lidar_pos[j][0],pt[0]],[lidar_pos[j][1],pt[1]],
                        c='black',alpha=0.4)
    ax.add_patch(flowbox)
    ax.set_aspect('equal')
    plt.legend(loc="lower right")
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')

    plt.show()


def plot_ffield(mc_obj):
    """
    Plots 2D geometry of lidar scan and flow field box

    Parameters
    ----------
    mc_obj : mocalum
        Instance of mocalum class
    """

    flow_id = mc_obj.data.ffield.generator


    lidar_ids = mc_obj.data.bbox_ffield[flow_id]['linked_lidars']


    no_los = mc_obj.data.meas_cfg[lidar_ids[0]]['config']['no_los']
    meas_pts= mc_obj._get_prob_cords(lidar_ids[0])[:no_los]

    bbox_pts = bbox_pts_from_cfg(mc_obj.data.bbox_ffield[flow_id])
    diag = np.abs(bbox_pts[0]-bbox_pts[2]).max()
    R_tb = mc_obj.data.bbox_ffield[flow_id]['CRS']['rot_matrix']
    min_bbox_pts = bbox_pts.dot(inv(R_tb))
    bbox_c = min_bbox_pts.mean(axis = 0)

    wind_dir = mc_obj.data.fmodel_cfg['wind_from_direction']

    flowbox = Polygon(min_bbox_pts,alpha=0.4, color='grey', label="flow field bbox")

    wind_dir_pt = spher2cart(wind_dir,0,diag/2)[:2]

    lidar_pos = []
    for id in lidar_ids:
        lidar_pos += [mc_obj.data.meas_cfg[id]['position']]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.grid()

    mc_obj.data.ffield.u.isel(z=0, time=0).plot.pcolormesh('Easting', 'Northing',ax=ax,cmap='Greys')


    plt.arrow(bbox_c[0], bbox_c[1],-wind_dir_pt[0],-wind_dir_pt[1],
              width=8,color="red", label='wind',zorder=50)
    plt.scatter(meas_pts[:,0], meas_pts[:,1], c="blue",
                label='measurements',zorder=10)
    colors = ["green", "orange", "purple"]
    for j,id in enumerate(lidar_ids):
        plt.scatter(lidar_pos[j][0],
                    lidar_pos[j][1], c=colors[j], label=id,zorder=150)
        for i,pt in enumerate(meas_pts):
            if i==0 and j==0:
                plt.plot([lidar_pos[j][0],pt[0]],[lidar_pos[j][1],pt[1]],
                        c='black',alpha=0.4, label='beam')
            else:
                plt.plot([lidar_pos[j][0],pt[0]],[lidar_pos[j][1],pt[1]],
                        c='black',alpha=0.4)
    ax.add_patch(flowbox)
    ax.set_aspect('equal')
    plt.legend(loc="lower right")
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')

    plt.show()