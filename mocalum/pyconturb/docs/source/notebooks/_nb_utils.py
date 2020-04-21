"""Utility functions for notebook examples"""
import matplotlib.pyplot as plt


def plot_slice(spat_df, turb_df, comp='u', val='mean',
               interpolation='none', ax=None):
    """Plot a time or statistic slice of turbulence box (grid only).
    """
    y = spat_df.loc['y'].unique()
    z = spat_df.loc['z'].unique()
    if comp not in ['u', 'v', 'w']:
        raise ValueError('Component must be "u", "v" or "w"!')
    if isinstance(val, str):
        plot_grid = (turb_df.filter(regex=comp).describe().loc[val]
                     .values.reshape(y.size, z.size).T)
    elif isinstance(val, (int, float)):
        if val not in turb_df.index:
            raise ValueError('Requested slice not in index!')
        plot_grid = turb_df.filter(regex=comp).loc[val].values.reshape(y.size, z.size)
    if ax is None:
        fig, ax = plt.subplots()
    plt.imshow(plot_grid,  # imshow requires nz-ny slice
               origin='lower',  # smallest y-z in lower left, not upper left
               extent=[y[0], y[-1], z[0], z[-1]],  # lateral and vertical limits
               interpolation=interpolation)  # image smoothing
    plt.colorbar()
    return ax


def plot_interp(yp, zp, valp, y, z, val, fmt='%.2f'):
    """a useful plotting function for the interpolator examples"""
    plt.imshow(val.reshape(y.shape), origin='lower',
               extent=[y.min(), y.max(), z.min(), z.max()], cmap='Greys')
    plt.colorbar(format=fmt)
    plt.scatter(yp, zp, s=100, c=valp, edgecolors='0.8', cmap='Greens', label='intp. locs')
    plt.colorbar(format=fmt)