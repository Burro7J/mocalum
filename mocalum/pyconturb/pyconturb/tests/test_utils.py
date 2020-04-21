# -*- coding: utf-8 -*-
"""test util functions
"""
import os

import numpy as np
import pandas as pd
import pytest

from pyconturb import TimeConstraint
import pyconturb._utils as utils


_spat_rownames = utils._spat_rownames


def test_clean_turb():
    """verify correct columns are removed and others are renamed (also handle floats)"""
    # given
    spat_df = pd.DataFrame([[0, 1], [0, 0], [0, 0], [70, 70. + 1e-13]],
                           index=['k', 'x', 'y', 'z'], columns=['u_p0', 'v_p0'], dtype=float)
    all_spat_df = pd.DataFrame([[0, 1, 0], [0, 0, 0], [0, 0, 0], [70. + 1e-13, 70, 80]],
                               index=['k', 'x', 'y', 'z'],
                               columns=['u_p0', 'v_p0_con', 'u_p0_con'], dtype=float)
    turb_df = pd.DataFrame([[0, 1, 2], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
                           index=np.arange(4)/4,
                           columns=['u_p0', 'v_p0_con', 'u_p0_con'], dtype=float)
    theo_df = pd.DataFrame([[0, 1], [2, 3], [3, 4], [4, 5]],
                           index=np.arange(4)/4, columns=['u_p0', 'v_p0'])
    # when
    utils.clean_turb(spat_df, all_spat_df, turb_df)
    # then
    pd.testing.assert_frame_equal(turb_df, theo_df, check_dtype=False)
    pd.testing.assert_index_equal(turb_df.index, theo_df.index)  # also check indexes


def test_combine_spat_con_empty():
    """verify we can combine empty/nonempty spat_df and con_tc"""
    # given
    empt_df = pd.DataFrame(index=_spat_rownames)
    df_sp = pd.DataFrame([[0], [0], [0], [60]], index=_spat_rownames, columns=['u_p0'])
    df_cn = pd.DataFrame([[0], [0], [0], [70]], index=_spat_rownames, columns=['u_p0'])
    theo_dfs = [empt_df,  # empty+empt=empt
                df_sp,  # empt con, full spat=spat
                df_cn.rename(index=str, columns={'u_p0': 'u_p0_con'}),  # emp spt, ful cn
                pd.concat((df_cn.rename(index=str, columns={'u_p0': 'u_p0_con'}), df_sp),
                          axis=1)]  # full con, full spat=con/spat
    for i in range(4):  # 4 diff tests
        spat_df = [empt_df, df_sp][i % 2]  # spat_df empty on 0, 2
        con_tc = TimeConstraint([empt_df, df_cn][i // 2])  # con_tc empty on 0, 1
        # when
        comb_df = utils.combine_spat_con(spat_df, con_tc)
        # then
        pd.testing.assert_frame_equal(comb_df, theo_dfs[i])


def test_combine_spat_con_nonunique():
    """should raise an error when we try to combine badly named columns"""
    # given
    spat_df = pd.DataFrame([[0], [0], [0], [60]], index=_spat_rownames, columns=['u_p0_con'])
    con_tc = TimeConstraint([[0], [0], [0], [70]], index=_spat_rownames, columns=['u_p0'])
    # when and then
    with pytest.raises(ValueError):
        utils.combine_spat_con(spat_df, con_tc)


def test_combine_spat_con_tcinspat():
    """some columns in TimeConstraint are in spat_df and float precision"""
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [50, 60.]],
                           index=_spat_rownames, columns=['u_p0', 'u_p1'])
    con_tc = TimeConstraint([[0, 0], [0, 0], [0, 0], [60 + 1e-13, 70]],
                            index=_spat_rownames, columns=['u_p0', 'u_p1'])
    theo_df = pd.DataFrame([[0, 0, 0], [0, 0, 0], [0, 0, 0], [60, 70, 50]],
                           index=_spat_rownames, columns=['u_p0_con', 'u_p1_con', 'u_p0'])
    # when
    comb_df = utils.combine_spat_con(spat_df, con_tc)
    # then
    pd.testing.assert_frame_equal(theo_df, comb_df, check_dtype=False)


def test_pctdf_to_h2turb():
    """save PyConTurb dataframe as binary file and load again"""
    # given
    path = '.'
    spat_df = utils.gen_spat_grid(0, [50, 70])
    turb_df = pd.DataFrame(np.random.rand(100, 6),
                           columns=[f'{c}_p{i}' for i in range(2) for c in 'uvw'])
    # when
    utils.df_to_h2turb(turb_df, spat_df, '.')
    test_df = utils.h2turb_to_df(spat_df, path)
    [os.remove(os.path.join('.', f'{c}.bin')) for c in 'uvw']
    # then
    pd.testing.assert_frame_equal(turb_df, test_df, check_dtype=False)


def test_gen_spat_grid():
    """verify column names and entries of spat grid
    """
    # given
    y, z, comps = 0, 0, [0, 2]
    theo_df = pd.DataFrame(np.zeros((4, 2)),  index=_spat_rownames,
                           columns=['u_p0', 'w_p0'])
    theo_df.iloc[0, :] = [0, 2]
    # when
    spat_df = utils.gen_spat_grid(y, z, comps=comps)
    # then
    pd.testing.assert_frame_equal(theo_df, spat_df, check_dtype=False)


def test_get_freq_values():
    """verify correct output of get_freq"""
    # given
    kwargs = {'T': 41, 'dt': 2}
    t_theo = np.arange(0, 42, 2)
    f_theo = np.arange(0, 11) / 41
    # when
    t_out, f_out = utils.get_freq(**kwargs)
    # then
    np.testing.assert_almost_equal(t_theo, t_out)
    np.testing.assert_almost_equal(f_theo, f_out)


def test_make_hawc2_input():
    """verify correct strings for hawc2 input"""
    # given
    turb_dir = '.'
    spat_df = utils.gen_spat_grid([-10, 10], [109, 129])
    kwargs = {'z_ref': 119, 'T': 600, 'dt': 1, 'u_ref': 10}
    str_cntr_theo = '  center_pos0             0.0 0.0 -119.0 ; hub height\n'
    str_mann_theo = ('  begin mann ;\n    filename_u ./u.bin ; \n'
                     + '    filename_v ./v.bin ; \n    filename_w ./w.bin ; \n'
                     + '    box_dim_u 600 10.0 ; \n    box_dim_v 2 20.0 ; \n'
                     + '    box_dim_w 2 20.0 ; \n    dont_scale 1 ; \n  end mann ')
    str_output_theo1 = '  wind free_wind 1 10.0 0.0 -109.0 # wind_p0 ; '
    str_output_theo2 = '  wind free_wind 1 10.0 0.0 -129.0 # wind_p1 ; '
    # when
    str_cntr, str_mann, str_output = \
        utils.make_hawc2_input(turb_dir, spat_df, **kwargs)
    # then
    assert(str_cntr == str_cntr_theo)
    assert(str_mann == str_mann_theo)
    assert(str_output.split('\n')[0] == str_output_theo1)
    assert(str_output.split('\n')[1] == str_output_theo2)


def test_rotate_time_series():
    """verify time series rotation"""
    # given
    xyzs = [[np.array([np.nan]), ] * 3,  # all nans
            [np.ones([1]), np.ones([1]), np.ones([1])],  # equal in all three
            [-np.ones([1]), np.ones([1]), np.ones([1])]]  # negative in x
    uvws = [xyzs[0],
            [np.sqrt(3) * np.ones([1]), np.zeros([1]), np.zeros([1])],
            [np.sqrt(3) * np.ones([1]), np.zeros([1]), np.zeros([1])]]
    # when
    for xyz, uvw_theo in zip(xyzs, uvws):
        uvw = utils.rotate_time_series(*xyz)
        # then
        np.testing.assert_almost_equal(uvw[0], uvw_theo[0])


def test_interpolator():
    """verify interpolator works for different cases: 1) a single points, 2) 2-pt vert
        line, 3) 2-pt horiz line, 4) 2-pt diag line, 5) 3-pt diag line, 6) triangle of
        points. All cases tests with both arrays in and tuples in.
    """
    # given
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
    values = np.array([0, 0.5, 1])
    xi = np.array([[-0.5, -0.5], [0.5, 0.5], [1.5, 1.5]])
    cases = (['1pt', [0], [0], [0], [0]],  # 1 point
             ['vline', [0, 1], [0, 2], [0, 1, 2], [0, 1, 2]],  # vert line
             ['hline', [0, 2], [0, 2], [0, 1, 2], [0, 1, 2]],  # horiz line
             ['dline2', [0, 3], [0, 2], [0, 1, 2], [0, 1, 2]],  # diag line, 2 pts
             ['dline3', [0, 4, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]],  # diag, 3 pts
             ['triang', [0, 2, 3], [0, 1, 2], [0, 1, 2], [0, 1, 2]])  # triangle
    for case in cases:
        name, i_pts, i_vals, i_xi, i_out = case
        # when
        out_arr = utils.interpolator(points[i_pts], values[i_vals], xi[i_xi])  # array in
        out_tup = utils.interpolator((points[i_pts, 0], points[i_pts, 1]),
                                     values[i_vals], (xi[i_xi, 0], xi[i_xi, 1]))  # tuple
        # then
        np.testing.assert_almost_equal(out_arr, values[i_out])
        np.testing.assert_almost_equal(out_tup, values[i_out])


def test_interpolator_badinput():
    """verify interpolator throws error with bad input
    """
    # given
    points, values, xi = 'abc'
    # when and then
    with pytest.raises(ValueError):
        utils.interpolator(points, values, xi)


if __name__ == '__main__':
    test_clean_turb()
    test_combine_spat_con_empty()
    test_combine_spat_con_nonunique()
    test_combine_spat_con_tcinspat()
    test_pctdf_to_h2turb()
    test_gen_spat_grid()
    test_get_freq_values()
    test_make_hawc2_input()
    test_rotate_time_series()
    test_interpolator()
    test_interpolator_badinput()
