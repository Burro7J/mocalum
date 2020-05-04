
import matplotlib.pyplot as plt

def plot_scan_setup(lidar_id, mc_obj):
    """
    Plots 2D geometry of lidar scan and flow field box

    Parameters
    ----------
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