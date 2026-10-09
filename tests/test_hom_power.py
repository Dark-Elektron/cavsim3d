"""HOM power: the power a beam leaves in each port mode (get_hom_power).

Validates:
  - energy: with lossless walls and every propagating port mode matched, the
    power the ports take is the power the beam loses, Re(Z_par) |I|^2 / 2
  - the reduced model and the joined copies give the full-order powers
  - the spectral lines of a bunch train
"""
import tempfile

import numpy as np
import pytest

from cavsim3d.analysis import bunch_train_spectrum
from cavsim3d.core.em_project import EMProject

from tests.test_beam import CELL_BEAM, CellChain

# 2.8-3.2 GHz: TE10 of the 60 x 40 mm pipe is the only propagating mode
CFG = dict(fmin=2.8, fmax=3.2, nsamples=2, nportmodes=1, order=3, solver_type="direct")


def _cell(tmp_path, name, n=1, maxh=0.009, **cfg):
    p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
    p.geometry = CellChain(n=n, maxh=maxh)
    p.add_beam('beam', **CELL_BEAM)
    p.fds.solve(**dict(CFG, **cfg))
    return p


def test_ports_take_the_power_the_beam_loses(tmp_path):
    fom = _cell(tmp_path, "cell").fds.fom
    I = np.array([0.3, 0.7])                                  # A, one line per frequency
    P = fom.get_hom_power(I)
    loss = 0.5 * fom.beam_impedance().real * I ** 2
    lines = sum(P['P_lines'].values())
    np.testing.assert_allclose(lines, loss, rtol=2.5e-2)       # measured 0.7 / 1.1 %
    assert set(P['P_mode']) == {'1(1)', '2(1)'} and set(P['P_port']) == {'1', '2'}
    assert P['P_total'] == pytest.approx(sum(P['P_port'].values()), rel=1e-12)
    assert P['P_port']['1'] == pytest.approx(P['P_mode']['1(1)'], rel=1e-12)
    # a function of the frequencies works the same way
    Q = fom.get_hom_power(lambda f: np.where(f < 3e9, 0.3, 0.7))
    assert Q['P_total'] == pytest.approx(P['P_total'], rel=1e-12)
    with pytest.raises(ValueError, match="one amplitude per frequency"):
        fom.get_hom_power(np.ones(3))


def test_reduced_and_joined_models_give_the_same_power(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the sections' scratch projects
    p = _cell(tmp_path, "train", maxh=0.014, nsamples=9, fmin=2.6, fmax=3.4)
    f = np.array([2.85, 3.05])                                 # between the snapshots
    rom = p.fds.fom.reduce(tol=1e-8)
    rom.solve(frequencies=f, store_snapshots=False)
    q = _cell(tmp_path, "fine", maxh=0.014, fmin=2.85, fmax=3.05, store_snapshots=False)
    I = np.ones(2)
    P_rom, P_fom = rom.get_hom_power(I), q.fds.fom.get_hom_power(I)
    for lab in P_fom['P_mode']:
        assert P_rom['P_mode'][lab] == pytest.approx(P_fom['P_mode'][lab], rel=1e-4)

    chain = EMProject(name="chain", base_dir=str(tmp_path), overwrite=True)
    chain.add("cell", CellChain(n=1, maxh=0.014), n=2)
    chain.add_beam('beam', **CELL_BEAM)
    chain.fds.solve(**dict(CFG, nsamples=9, fmin=2.6, fmax=3.4))
    fom_join = chain.fds.foms.concatenate()
    concat = chain.fds.foms.reduce(tol=1e-8).concatenate()
    concat.solve(frequencies=fom_join.frequencies / 1e9, store_snapshots=False)
    I9 = np.ones(len(fom_join.frequencies))
    P_join, P_red = fom_join.get_hom_power(I9), concat.get_hom_power(I9)
    for lab in P_join['P_mode']:
        np.testing.assert_allclose(P_red['P_lines'][lab], P_join['P_lines'][lab],
                                   rtol=0, atol=1e-4 * P_join['P_total'])
    # saved with S~ and read back with it
    from cavsim3d.solvers import beam as bm
    from cavsim3d.solvers.concatenation import ConcatenatedSystem
    folder, _ = concat._results_dir()
    zref = bm.load_tilde(folder / "s_tilde" / "s_tilde.h5")['zref']
    np.testing.assert_allclose(zref, concat._beam['zref'])
    loaded = ConcatenatedSystem.load(folder)
    np.testing.assert_allclose(loaded._beam['zref'], zref)


def test_bunch_train_spectrum():
    q, tb, st = 1e-9, 25e-9, 30e-12                            # 1 nC every 25 ns
    f, I = bunch_train_spectrum(q, tb, sigma_t=st, fmax=1.0, fmin=0.1)
    np.testing.assert_allclose(f, np.arange(3, 26) * 0.04)     # GHz: the 40 MHz harmonics
    np.testing.assert_allclose(I, 2 * q / tb * np.exp(-0.5 * (2 * np.pi * f * 1e9 * st) ** 2))
    f0, I0 = bunch_train_spectrum(q, tb, fmax=0.1)
    np.testing.assert_allclose(I0, 2 * q / tb)                 # a point bunch: flat
    assert f0[0] == pytest.approx(0.04)                        # no DC line
    with pytest.raises(ValueError, match="fmax"):
        bunch_train_spectrum(q, tb)
