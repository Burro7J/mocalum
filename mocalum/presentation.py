import matplotlib.pyplot as plt
from matplotlib.pyplot import Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from .utils import spher2cart, bbox_pts_from_array, bbox_pts_from_cfg
from .logics import _rot_matrix
import numpy as np
from numpy.linalg import inv as inv


def plot_mocalum_setup(mc_obj):
    """
    Plots 2D geometry of lidar scan and flow field box

    Parameters
    ----------
    mc_obj : mocalum
        Instance of mocalum class
    """
    avg_azimuth = mc_obj.data.meas_cfg['az'].mean()
    no_los = mc_obj.data.meas_cfg['no_los']
    arc_cords= mc_obj._get_prob_cords()[:no_los]

    arc_bbox = bbox_pts_from_array(mc_obj._get_prob_cords()[:,(0,1)].dot(_rot_matrix(avg_azimuth)))
    arc_bbox = arc_bbox.dot(np.linalg.inv(_rot_matrix(avg_azimuth)))

    x_len = arc_bbox[:,0].max()-arc_bbox[:,0].min()
    y_len = arc_bbox[:,1].max()-arc_bbox[:,0].min()
    diagonal = ((x_len)**2+(y_len)**2)**(.5)


    lidar_pos = mc_obj.data.meas_cfg['lidar_pos']
    wind_dir = mc_obj.data.fmodel_cfg['wind_from_direction']
    R_tb = mc_obj.data.ffield_bbox_cfg['CRS']['rot_matrix']

    bbox_pts = bbox_pts_from_cfg(mc_obj.data.ffield_bbox_cfg)
    min_bbox_pts = bbox_pts.dot(inv(R_tb))
    bbox_c = min_bbox_pts.mean(axis = 0)


    wind_dir_pt = spher2cart(wind_dir,0,diagonal/4)[:2]


    flowbox = Polygon(min_bbox_pts,alpha=0.4, color='grey', label="flow field bbox")
    arcbox = Polygon(arc_bbox,alpha=0.4, color='blue', label="measurements bbox")

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.grid()
    # plt.scatter(bbox_pts[:,0],bbox_pts[:,1], c="black")

    plt.scatter(arc_cords[:,0], arc_cords[:,1], c="blue", label='measurements'
                ,zorder=10)
    plt.arrow(bbox_c[0], bbox_c[1],-wind_dir_pt[0],-wind_dir_pt[1],
              width=8,color="red", label='wind',zorder=50)
    for i,pt in enumerate(arc_cords):
        if i==0:
            plt.plot([lidar_pos[0],pt[0]],[lidar_pos[1],pt[1]], c='black',alpha=0.4, label='beam')
        else:
            plt.plot([lidar_pos[0],pt[0]],[lidar_pos[1],pt[1]], c='black',alpha=0.4)
    # plt.scatter(tbox_pts[:,0], tbox_pts[:,1], c="red")


    plt.scatter(lidar_pos[0],lidar_pos[1], c="green", label='lidar',zorder=150)
    ax.add_patch(flowbox)
    ax.add_patch(arcbox)
    ax.set_aspect('equal')
    plt.legend(loc="upper left")

    plt.show()