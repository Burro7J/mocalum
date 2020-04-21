# -*- coding: utf-8 -*-
"""Base classes used in PyConTurb
"""
import numpy as np
import pandas as pd
from pandas import DataFrame


class TimeConstraint(DataFrame):
    """DataFrame-style object that specfies time constraints for simulation.

    The TimeConstraint object is essentially a pandas.DataFrame with a particular
    structure and some added useful methods. The first four rows correspond to ``k``,
    ``x``, ``y`` and ``z``. The remaining rows are time steps. Each column corresponds to
    a different location/turbulence component. The index must be of the form
    ``['k', 'x', 'y', 'z', <float array of time steps>]``. The columns need not have
    names.

    The object can be pickled/saved using same methods as pandas. To reload, load a
    properly-formatted pandas.DataFrame using the usual methods and pass it into the
    object: e.g.,``TimeConstraint(pd.read_csv(path))``.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame.
        Dict can contain Series, arrays, constants, or list-like objects.
    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided
    columns : Index or array-like
        Column labels to use for resulting frame. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer
    copy : boolean, default False
        Copy data from inputs. Only affects DataFrame / 2d ndarray input
    """
    def get_spat(self):
        """Return a DataFrame with the spatial information"""
        return self[self.index.map(lambda x: type(x) is str)]

    def get_time(self):
        """Return a DataFrame with the spatial information"""
        return self[self.index.map(lambda x: type(x) is not str)]

    def get_T(self):
        """Return the total time of the constraining data"""
        t = self.get_time().index
        return t[-1] + t[1]

    def from_con_data(self, con_data=None, con_spat_df=None, con_turb_df=None):
        """Create TimeConstraint from old-style con_data dictionary"""
        # assing/check inputs
        if (con_data is None) and ((con_spat_df is None) or (con_turb_df is None)):
            raise ValueError('con_data was not given, so both con_spat_df and ' +
                             'con_turb_df must be given!')
        elif (con_data is not None):
            con_spat_df = con_data['con_spat_df']
            con_turb_df = con_data['con_turb_df']
        # check correct sizes
        if con_turb_df.shape[1] != con_spat_df.shape[0]:
            raise ValueError('No. rows in con_spat_df do not match columns in ' +
                             'con_turb_df!')
        try:
            con_spat_df = con_spat_df.T.drop('p_id').T
        except KeyError:
            pass
        con_df = pd.DataFrame(np.r_[con_spat_df.T, con_turb_df],
                              columns=con_turb_df.columns,
                              index=con_spat_df.columns.to_list()
                              + con_turb_df.index.to_list())
        self.__init__(con_df)
        return self
