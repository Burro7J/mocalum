import numpy as np

def gen_correlated_samples(mu, std, corr_coef=0, no_samples = 10000):
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


