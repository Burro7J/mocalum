
import matplotlib.pyplot as plt

def plot_scan_setup(lidar_ids, mc_obj):
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