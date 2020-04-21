# -*- coding: utf-8 -*-
"""Test functions in coherence.py

Author
------
Jenni Rinker
rink@dtu.dk
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from pyconturb.simulation import gen_turb
from pyconturb.coherence import get_coh_mat
from pyconturb._utils import gen_spat_grid, _spat_rownames


def test_main_default():
    """Check the default value for get_coherence
    """
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 1]], index=_spat_rownames, columns=['u_p0', 'u_p1'])
    freq = 1
    kwargs = {'u_ref': 1, 'l_c': 1}
    coh_theory = np.array([[1, 5.637379774e-6],
                           [5.637379774e-6, 1]])
    # when
    coh = get_coh_mat(freq, spat_df, **kwargs)[:, :, 0]
    # then
    np.testing.assert_allclose(coh, coh_theory)


def test_main_badcohmodel():
    """Should raise an error if a wrong coherence model is passed in
    """
    # given
    spat_df = pd.DataFrame([[0], [0], [0], [0]], index=_spat_rownames, columns=['u_p0'])
    freq = 1
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(spat_df, freq, coh_model='garbage')


def test_iec_badedition():
    """IEC coherence should raise an error if any edn other than 3 is given
    """
    # given
    spat_df = pd.DataFrame([[0], [0], [0], [0]], index=_spat_rownames, columns=['u_p0'])
    freq, coh_model = 1, 'iec'
    kwargs = {'ed': 4, 'u_ref': 12, 'l_c': 340.2}
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(spat_df, freq, **kwargs)


def test_iec_missingkwargs():
    """IEC coherence should raise an error if missing parameter(s)
    """
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 1]], index=_spat_rownames, columns=['u_p0', 'u_p1'])
    freq, kwargs, coh_model = 1, {'ed': 3, 'u_ref': 12}, 'iec'
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(freq, spat_df, coh_model=coh_model, **kwargs)


def test_3d_missingkwargs():
    """IEC coherence should raise an error if missing parameter(s)
    """
    # given
    spat_df = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 1]], index=_spat_rownames, columns=['u_p0', 'u_p1'])
    freq, kwargs, coh_model = 1, {'ed': 3, 'u_ref': 12}, '3d'
    # when & then
    with pytest.raises(ValueError):
        get_coh_mat(freq, spat_df, coh_model=coh_model, **kwargs)


def test_iec_value():
    """Verify that the value of IEC coherence matches theory
    """
    # 1: same comp, 2: diff comp
    for comp2, coh2 in [(0, 0.0479231144), (1, 0)]:
        # given
        spat_df = pd.DataFrame([[0, comp2], [0, 0], [0, 0], [0, 1]],
                               index=_spat_rownames, columns=['u_p0', 'x_p1'])
        freq = 0.5
        kwargs = {'ed': 3, 'u_ref': 2, 'l_c': 3, 'coh_model': 'iec'}
        coh_theory = np.array([[1., coh2], [coh2, 1.]])
        # when
        coh = get_coh_mat(freq, spat_df, **kwargs)[:, :, 0]
        # then
        np.testing.assert_allclose(coh, coh_theory)


def test_3d_value():
    """Verify that the value of 3d coherence matches theory"""
    # 1: same comp, 2: diff comp
    for comp, coh2 in [(0, 0.0479231144), (1, 0.0358754554), (2, 0.0013457414)]:
        # given
        spat_df = pd.DataFrame([[comp, comp], [0, 0], [0, 0], [0, 1]],
                               index=_spat_rownames, columns=['x_p0', 'y_p1'])
        freq = 0.5
        kwargs = {'u_ref': 2, 'l_c': 3, 'coh_model': '3d'}
        coh_theory = np.array([[1., coh2], [coh2, 1.]])
        # when
        coh = get_coh_mat(freq, spat_df, **kwargs)[:, :, 0]
        # then
        np.testing.assert_allclose(coh, coh_theory, atol=1e-6)


@pytest.mark.slow  # mark this as a slow test
@pytest.mark.skipci  # don't run in CI
def test_verify_iec_sim_coherence():
    """check that the simulated box has the right coherence
    """
    # given
    y, z = [0], [70, 80]
    spat_df = gen_spat_grid(y, z)
    kwargs = {'u_ref': 10, 'turb_class': 'B', 'l_c': 340.2, 'z_ref': 70,
              'T': 300, 'dt': 100}
    coh_model = 'iec'
    n_real = 1000  # number of realizations in ensemble
    coh_thresh = 0.12  # coherence threshold
    # get theoretical coherence
    idcs = np.triu_indices(spat_df.shape[1], k=1)
    coh_theo = get_coh_mat(1 / kwargs['T'], spat_df, coh_model=coh_model,
                           **kwargs)[idcs].flatten()
    # when
    ii_jj = [(i, j) for (i, j) in
             itertools.combinations(np.arange(spat_df.shape[1]), 2)]  # pairwise indices
    ii, jj = [tup[0] for tup in ii_jj], [tup[1] for tup in ii_jj]
    turb_ens = np.empty((int(np.ceil(kwargs['T']/kwargs['dt'])),
                         3 * len(y) * len(z), n_real))
    for i_real in range(n_real):
        turb_ens[:, :, i_real] = gen_turb(spat_df, coh_model=coh_model, **kwargs)
    turb_fft = np.fft.rfft(turb_ens, axis=0)
    x_ii, x_jj = turb_fft[1, ii, :], turb_fft[1, jj, :]
    coh = np.mean((x_ii * np.conj(x_jj)) /
                  (np.sqrt(x_ii * np.conj(x_ii)) *
                   np.sqrt(x_jj * np.conj(x_jj))),
                  axis=-1)
    max_coh_diff = np.abs(coh - coh_theo).max()
    # then
    assert max_coh_diff < coh_thresh


if __name__ == '__main__':
    test_main_default()
    test_main_badcohmodel()
    test_iec_badedition()
    test_iec_missingkwargs()
    test_3d_missingkwargs()
    test_iec_value()
    test_3d_value()
