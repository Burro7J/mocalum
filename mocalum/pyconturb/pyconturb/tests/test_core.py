# -*- coding: utf-8 -*-
"""Test functions in core.py
"""
import numpy as np
import pandas as pd

from pyconturb import TimeConstraint


def test_timecon_get_spat():
    """verify we get correct spatial frame"""
    # given
    con_tc = TimeConstraint(np.atleast_2d([0, 0, 0, 119, 0, 1]).T,
                            index=['k', 'x', 'y', 'z', 0, 1], columns=['u_p0'])
    theo_time = con_tc.loc[['k', 'x', 'y', 'z']]  # pulls out last two rows
    # when
    time_vec = con_tc.get_spat()
    # then
    np.testing.assert_array_equal(theo_time, time_vec)


def test_timecon_get_time():
    """verify we get correct time frame"""
    # given
    con_tc = TimeConstraint(np.atleast_2d([0, 0, 0, 119, 0, 1]).T,
                            index=['k', 'x', 'y', 'z', 0, 1], columns=['u_p0'])
    theo_time = con_tc.loc[[0, 1]]  # pulls out last two rows
    # when
    time_vec = con_tc.get_time()
    # then
    np.testing.assert_array_equal(theo_time, time_vec)


def test_timecon_get_T():
    """verify we get correct time frame"""
    # given
    con_tc = TimeConstraint(np.atleast_2d([0, 0, 0, 119, 0, 1]).T,
                            index=['k', 'x', 'y', 'z', 0, 1], columns=['u_p0'])
    theo_T = 2  # pulls out last two rows
    # when
    T = con_tc.get_T()
    # then
    np.testing.assert_array_equal(theo_T, T)


def test_timecon_from_condata():
    """verify correct converstion from old con_data format"""
    # given
    con_spat_df = pd.DataFrame([[0, 0, 0, 0, 119]],
                               columns=['k', 'p_id', 'x', 'y', 'z'])
    con_turb_df = pd.DataFrame([[0], [1]], index=[0, 1], columns=['u_p0'])
    con_data = {'con_spat_df': con_spat_df, 'con_turb_df': con_turb_df}
    theo_tc = TimeConstraint(np.atleast_2d([0, 0, 0, 119, 0, 1]).T,
                             index=['k', 'x', 'y', 'z', 0, 1], columns=['u_p0'])
    # when
    con_tc1 = TimeConstraint().from_con_data(con_data)
    con_tc2 = TimeConstraint().from_con_data(con_spat_df=con_spat_df, con_turb_df=con_turb_df)
    # then
    pd.testing.assert_frame_equal(theo_tc, con_tc1, check_dtype=False)
    pd.testing.assert_frame_equal(theo_tc, con_tc2, check_dtype=False)


if __name__ == '__main__':
    test_timecon_get_spat()
    test_timecon_get_time()
    test_timecon_get_T()
    test_timecon_from_condata()
