import matplotlib.pyplot as plt
from matplotlib.pyplot import Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from .utils import spher2cart
import numpy as np

def plot_mocalum_setup(mc_obj):
    """
    Plots 2D geometry of lidar scan and flow field box

    Parameters
    ----------
    mc_obj : mocalum
        Instance of mocalum class
    """
    bbox_pts = mc_obj._get_bbox_pts(mc_obj.data.ffield_bbox_cfg)
    bbox_c = bbox_pts.mean(axis = 0)
    arc_cords= mc_obj._get_prob_cords()
    diagonal = mc_obj._get_diagonal(mc_obj.data.ffield_bbox_cfg)


    lidar_pos = mc_obj.data.meas_cfg['lidar_pos']
    wind_dir = mc_obj.data.fmodel_cfg['wind_from_direction']
    wind_dir_pt = spher2cart(wind_dir,0,diagonal/4)[:2]

    tbox_pts = (mc_obj._get_bbox_pts(mc_obj.data.turb_bbox_cfg)
                + np.array([mc_obj.data.turb_bbox_cfg['x']['offset'],
                            mc_obj.data.turb_bbox_cfg['y']['offset']]))

    tbox_pts = tbox_pts.dot(np.linalg.inv(mc_obj.data.turb_bbox_cfg['CSR']['rot_matrix']))


    bbox = Polygon(bbox_pts,alpha=0.4, color='grey', label="flow field bbox")
    tbox = Polygon(tbox_pts,alpha=0.1, color='red', label="turb field bbox")

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.grid()
    # plt.scatter(bbox_pts[:,0],bbox_pts[:,1], c="black")

    plt.scatter(arc_cords[:,0], arc_cords[:,1], c="blue", label='arc pts'
                ,zorder=10)
    plt.arrow(bbox_c[0], bbox_c[1],-wind_dir_pt[0],-wind_dir_pt[1],
              width=8,color="red", label='wind',zorder=50)
    for pt in arc_cords:
        plt.plot([lidar_pos[0],pt[0]],[lidar_pos[1],pt[1]], c='black',alpha=0.4)
    # plt.scatter(tbox_pts[:,0], tbox_pts[:,1], c="red")


    plt.scatter(lidar_pos[0],lidar_pos[1], c="green", label='lidar',zorder=150)



    ax.add_patch(tbox)
    ax.add_patch(bbox)

    ax.set_aspect('equal')
    plt.legend(loc="lower left")

    plt.show()