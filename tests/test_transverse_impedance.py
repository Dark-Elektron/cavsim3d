"""Transverse beam impedance from beams at +-d (transverse_impedance).

Validates:
  - add_transverse_beams() places two beams per transverse plane
  - near the TM110 pair of a pillbox (open ports, below the pipe's cut-off),
    Z_perp has the poles of the two modes with the residues their transverse
    R/Q gives, j (w / c) (R/Q_t) w_i / (4 (w_i - w)), plane by plane
"""
import numpy as np
import pytest

from cavsim3d.core.constants import c0
from cavsim3d.core.em_project import EMProject

CHI11 = 3.831706


def test_beams_placed_per_plane(tmp_path):
    p = EMProject(name="beams", base_dir=str(tmp_path), overwrite=True)
    beams = p.add_transverse_beams(0.004, x=0.001)
    assert [b.name for b in beams] == ['dipole_x+', 'dipole_x-', 'dipole_y+', 'dipole_y-']
    assert [b.point for b in beams] == [(0.005, 0.0, 0.0), (-0.003, 0.0, 0.0),
                                        (0.001, 0.004, 0.0), (0.001, -0.004, 0.0)]
    with pytest.raises(ValueError, match="offset"):
        p.add_transverse_beams(0.0)


def test_poles_of_the_dipole_pair(tmp_path):
    p = EMProject(name="pillbox", base_dir=str(tmp_path), overwrite=True)
    p.create_primitive('pillbox', name='cav', n_cells=1, dims=[100, 100, 30, 0, 100],
                       beampipe='both')
    p.generate_mesh(maxh=0.025, curve_order=4)
    p.add_transverse_beams(0.006)
    cfg = dict(nportmodes=1, order=3, solver_type='direct')
    p.fds.solve(fmin=1.78, fmax=1.80, nsamples=2, **cfg)
    fom = p.fds.fom
    f = fom.get_resonant_frequencies()
    pair = [int(i) for i in np.argsort(np.abs(f - CHI11 * c0 / (2 * np.pi * 0.1)))[:2]]
    rq = {i: fom.get_figures_of_merit(i) for i in pair}
    f0 = float(np.mean(f[pair]))

    p.fds.solve(fmin=f0 * (1 - 1e-3) / 1e9, fmax=f0 * (1 + 1e-3) / 1e9, nsamples=2, **cfg)
    fq = p.fds.fom.frequencies
    for u in "xy":
        zt = p.fds.fom.transverse_impedance(u, ports='open')
        poles = sum(1j * 2 * np.pi * fq / c0 * rq[i][f"R/Q_t_{u} [Ohm]"] * f[i] / (4 * (f[i] - fq))
                    for i in pair)
        # the other modes add a smooth part, of opposite sign relative to the
        # poles on the two sides: the mean of the two ratios is the residue's
        ratio = np.mean(zt.imag / poles.imag)
        assert ratio == pytest.approx(1.0, abs=0.05)              # measured 0.7 / 2.3 %
        assert np.all(np.abs(zt.real) < 0.02 * np.abs(zt.imag))   # lossless: reactive

    with pytest.raises(ValueError, match="plane"):
        p.fds.fom.transverse_impedance('z')
    with pytest.raises(KeyError, match="add_transverse_beams"):
        p.fds.fom.transverse_impedance('x', name='quad')
