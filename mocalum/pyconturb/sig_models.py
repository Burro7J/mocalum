# -*- coding: utf-8 -*-
"""Define how the turbulence standard deviation varies with k, y and z.

You can either use the built-in models (see below) or create your own function
to model the spatial variation of the turbulence standard deviation.
"""
import numpy as np

from pyconturb._utils import _DEF_KWARGS, interpolator


def get_sig_values(spat_df, sig_func, **kwargs):
    """Turbulence standard deviation for points/components in ``spat_df``.

    The ``sig_func`` must be a function of the form::

        sig_values = sig_func(k, y, z, **kwargs)

    where k, y and z can be floats, np.arrays or pandas.Series. You can use
    the functions built into PyConTurb (see below) or define your own custom
    function. The output is assumed to be in m/s.

    Parameters
    ----------
    spat_df : pandas.DataFrame
        Spatial information on the points to simulate. Must have columns
        ``[k, p_id, x, y, z]``, and each of the ``n_sp`` rows corresponds
        to a different spatial location and turbuine component (u, v or
        w).
    sig_func : function
        Function to map k, y and z to the turbulence standard deviation in m/s.
    **kwargs
        Keyword arguments to pass into ``sig_func``.

    Returns
    -------
    sig_values : np.array
        [m/s] Turbulence standard deviation(s) for the given spatial locations(s)
        /component(s). Dimension is ``(n_sp,)``.
    """
    return sig_func(spat_df.loc['k'], spat_df.loc['y'], spat_df.loc['z'], **kwargs)


def data_sig(k, y, z, con_tc=None, **kwargs):
    """Turbulence standard deviation interpolated from a TimeConstraint object.

    See the Examples and/or Reference Guide for details on the interpolator logic or for
    how to construct a TimeConstraint object. Note that this function uses the
    biased estimator for the standard deviation (i.e., NumPy's default ``np.std``).

    Parameters
    ----------
    k : array-like
        [-] Integer indicator of the turbulence component. 0=u, 1=v, 2=w.
    y : array-like
        [m] Location of point(s) in the lateral direction. Can be int/float,
        np.array or pandas.Series.
    z : array-like
        [m] Location of point(s) in the vertical direction. Can be int/float,
        np.array or pandas.Series.
    con_tc : pyconturb.TimeConstraint
        [-] Constraint object. Must have correct format; see documentation on
        PyConTurb's TimeConstraint object for more details.
    **kwargs
        Unused (optional) keyword arguments.

    Returns
    -------
    sig_values : np.array
        [m/s] Turbulence standard deviation(s) at the specified location(s). Dimension
        is ``(n_sp,)``.
    """
    if con_tc is None:
        raise ValueError('No data provided!')
    k, y, z = np.array(k, dtype=int), np.array(y), np.array(z)
    out_array = np.empty_like(y, dtype=float)
    for kval in np.unique(k):  # loop over passed-in components
        out_mask = (k == kval)
        con_mask = (con_tc.loc['k'] == kval).values
        ypts = con_tc.iloc[2, con_mask].values.astype(float)
        zpts = con_tc.iloc[3, con_mask].values.astype(float)
        vals = np.std(con_tc.iloc[4:, con_mask], axis=0).astype(float)
        out_array[out_mask] = interpolator((ypts, zpts), vals, (y[out_mask], z[out_mask]))
    return out_array


def iec_sig(k, y, z, turb_class=_DEF_KWARGS['turb_class'], **kwargs):
    """Turbulence standard deviation as specified in IEC 61400-1 Ed. 3.

    Parameters
    ----------
    k : array-like
        [-] Integer indicator of the turbulence component. 0=u, 1=v, 2=w.
    y : array-like
        [m] Location of point(s) in the lateral direction. Can be int/float,
        np.array or pandas.Series.
    z : array-like
        [m] Location of point(s) in the vertical direction. Can be int/float,
        np.array or pandas.Series.
    turb_class : str, optional
        [-] Turbulence class.
    **kwargs
        Unused (optional) keyword arguments.

    Returns
    -------
    sig_values : np.array
        [m/s] Turbulence standard deviation(s) at the specified location(s). Dimension
        is ``(n_sp,)``.
    """
    kwargs = {**{'turb_class': turb_class}, **kwargs}  # add dflts if not given
    assert kwargs['turb_class'].lower() in 'abc', 'Invalid or no turbulence class!'
    i_ref = {'a': 0.16, 'b': 0.14, 'c': 0.12}[kwargs['turb_class'].lower()]
    sig1 = i_ref * (0.75 * kwargs['u_ref'] + 5.6)  # std dev in u
    sig_k = sig1 * np.asarray(1.0 * (k == 0) + 0.8 * (k == 1) + 0.5 * (k == 2))
    return np.array(sig_k, dtype=float)
