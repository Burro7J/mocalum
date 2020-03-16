from .utils import add_xyz
import numpy as np
import xarray as xr

def gen_unc(mu, std, corr_coef=0, no_samples = 10000):
    """Generate correlated random samples for inputs.

    Parameters
    ----------
    mu : numpy
        Array of mean values of inputs
    std : numpy
        Array of standard deviations if inputs
    corr_coef : float
        Correlation coffecient of inputs
        Default 0, uncorrelated inputs
        Max 1, 100% correlated inputs
    no_samples : int, optional
        Number of generated samamples, by default 10000

    Returns
    -------
    samples : numpy
        Array of correlated random samples.
        Array dimension is (len(mu), no_samples)

    Raises
    ------
    DimensionError
        If dimensions of mu and std are not the same
    """

    if len(mu)!= len(std):
        raise DimensionError('Length of mu and std arrays are not the same!')

    cov_matrix = np.full((len(mu), len(mu)), corr_coef)
    np.fill_diagonal(cov_matrix, 1)

    # Generate the random samples.
    samples = std*np.random.multivariate_normal(mu, cov_matrix, no_samples)

    return samples


# def gen_unc(no_los, no_sim, unc_cfg={'azimuth':{'mu':0, 'std':0.1},
#                                'elevation':{'mu':0, 'std':0.1},
#                                'range':{'mu':0, 'std':10},
#                                'estimation':{'mu':0, 'std':0.1},
#                                'corr_coef':0}):

#     unc = {'unc_az': gen_corr_samples(np.full(no_los, unc_desc['azimuth']['mu']),
#                                     np.full(no_los, unc_desc['azimuth']['std']),
#                                     unc_cfg['corr_coef'], no_sim),
#            'unc_el': gen_corr_samples(np.full(no_los, unc_desc['elevation']['mu']),
#                                     np.full(no_los, unc_desc['elevation']['std']),
#                                     unc_cfg['corr_coef'], no_sim),
#            'unc_rng':gen_corr_samples(np.full(no_los, unc_desc['range']['mu']),
#                                     np.full(no_los, unc_desc['range']['std']),
#                                     unc_cfg['corr_coef'], no_sim),
#            'unc_est':gen_corr_samples(np.full(no_los, unc_desc['estimation']['mu']),
#                                      np.full(no_los, unc_desc['estimation']['std']),
#                                      unc_cfg['corr_coef'], no_sim)}

#     return unc

# def mc_config_ivap(ds, lidar_pos, corr_coef=0, unc_desc={'azimuth':{'mu':0, 'std':0.1},
#                                               'elevation':{'mu':0, 'std':0.1},
#                                               'distance':{'mu':0, 'std':10},
#                                               'los':{'mu':0, 'std':0.1}}):

#     no_sim = int(ds.no_scans.values)
#     no_los = int(ds.no_los.values)
#     unc_dict = gen_unc(no_los, no_sim, corr_coef, unc_desc)
#     ds.unc_az.values = unc_dict['unc_az'].flatten()
#     ds.unc_el.values = unc_dict['unc_el'].flatten()
#     ds.unc_dis.values = unc_dict['unc_dis'].flatten()
#     ds.unc_los.values = unc_dict['unc_los'].flatten()

#     ds = add_xyz(ds, lidar_pos)

#     return ds