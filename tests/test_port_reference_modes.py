"""Reference impedance and labels when ports carry different numbers of modes.

A 50-ohm air coax with TE11 in band (cutoff ~2.9 GHz), 1 mode on port1 and
3 on port2.  Every S-parameter is checked against S rebuilt independently
from the reduced model's A and B, with each port mode referred to its own
wave impedance: S does not depend on the reference as long as Z is scaled
with it.
"""
import json

import h5py
import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.core.persistence import H5Serializer
from cavsim3d.geometry.base import BaseGeometry
from cavsim3d.solvers.base import ParameterConverter
from netgen.occ import Axes, Cylinder, Z as OCC_Z

A_IN, B_OUT, LENGTH = 0.010, 0.023, 0.10
BAND = dict(fmin=1.0, fmax=5.0, nsamples=5)


class _Coax(BaseGeometry):
    def __init__(self):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=0.008)

    def build(self):
        outer = Cylinder(Axes((0, 0, 0), OCC_Z), r=B_OUT, h=LENGTH)
        inner = Cylinder(Axes((0, 0, 0), OCC_Z), r=A_IN, h=LENGTH)
        self.geo = outer - inner
        for f in self.geo.faces:
            f.name = "default"
        self.geo.faces.Min(OCC_Z).name = "port1"
        self.geo.faces.Max(OCC_Z).name = "port2"
        self.geo.mat("vacuum")
        self.bc = "default"


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    base = tmp_path_factory.mktemp("coax")
    proj = EMProject(name="coax", base_dir=str(base))
    proj.geometry = _Coax()
    proj.fds.solve(config=dict(**BAND, order=1, nportmodes={'port1': 1, 'port2': 3},
                               solver_type="direct"))
    rom = proj.fds.fom.reduce(tol=1e-10)
    rom.solve(**BAND)
    return base, proj, rom


def _independent_s(proj, rom):
    ps = proj.fds.port_solver
    order = proj.fds._port_mode_order
    A = np.asarray(rom.A['global'] if isinstance(rom.A, dict) else rom.A)
    B = np.asarray(rom.B['global'] if isinstance(rom.B, dict) else rom.B)
    lam, V = np.linalg.eigh(0.5 * (A + A.conj().T))
    out = []
    for fk in proj.fds.fom.frequencies:
        w = 2 * np.pi * fk
        Z = 1j * w * ((B.T @ V) * (1 / (lam - w ** 2))) @ (V.conj().T @ B)
        out.append(ParameterConverter.z_to_s(
            Z, np.diag([ps.get_port_wave_impedance(p, m, fk) for p, m in order])))
    return np.array(out)


def test_coax_te_modes_use_their_wave_impedance(solved):
    _, proj, _ = solved
    ps = proj.fds.port_solver
    assert ps.port_mode_types['port2'] == {0: 'TEM', 1: 'TE', 2: 'TE'}
    f = 4e9
    z_line = ps.get_port_reference_impedance('port2', 0, f)
    # radii fitted from the mesh: ~1e-4 off
    assert z_line.real == pytest.approx(376.730313668 / (2 * np.pi) * np.log(B_OUT / A_IN), rel=1e-3)
    for m in (1, 2):
        assert ps.get_port_line_impedance('port2', m) is None
        assert ps.get_port_reference_impedance('port2', m, f) == ps.get_port_wave_impedance('port2', m, f)


def test_fom_and_rom_s_match_independent(solved):
    _, proj, rom = solved
    s_ind = _independent_s(proj, rom)
    assert np.abs(proj.fds.fom._S_matrix - s_ind).max() < 1e-8
    assert np.abs(rom._S_matrix - s_ind).max() < 1e-8


def test_rom_labels_and_z_match_fom(solved):
    _, proj, rom = solved
    fom = proj.fds.fom
    keys = [k for k in fom.S_dict if k != 'frequencies']
    assert sorted(keys) == sorted(k for k in rom.S_dict if k != 'frequencies')
    for k in keys:
        np.testing.assert_allclose(rom.S_dict[k], fom.S_dict[k], atol=1e-8)
    np.testing.assert_allclose(rom._Z_matrix, fom._Z_matrix, rtol=1e-6, atol=1e-6)


def test_version1_results_corrected_on_reopen(solved):
    """Results saved when every coax mode was referred to the line impedance."""
    base, proj, _ = solved
    fds = proj.fds
    ps = fds.port_solver
    S_ok, Z_ok = fds.fom._S_matrix.copy(), fds.fom._Z_matrix.copy()
    f = np.asarray(fds.fom.frequencies)
    order = fds._port_mode_order
    proj.save()

    # what the old code wrote: TE modes scaled and referred like the TEM mode
    z_line = ps.get_port_line_impedance('port2', 0)
    legacy = {('port2', 1), ('port2', 2)}
    r = np.array([abs(z_line) / abs(ps.get_port_wave_impedance(p, m, f[0]))
                  if (p, m) in legacy else 1.0 for p, m in order])
    Z_v1 = Z_ok * np.sqrt(np.outer(r, r))[None]
    S_v1 = np.array([ParameterConverter.z_to_s(Z_v1[k], np.diag(
        [z_line if (p, m) in legacy else ps.get_port_reference_impedance(p, m, fk)
         for p, m in order])) for k, fk in enumerate(f)])
    assert np.abs(S_v1 - S_ok).max() > 0.1

    root = base / "coax" / "fds"
    for name, data in (("fom/z/z_global.h5", Z_v1), ("fom/s/s_global.h5", S_v1)):
        with h5py.File(root / name, "w") as fh:
            H5Serializer.save_dataset(fh, "data", data)
    cfg = json.loads((root / "config.json").read_text())
    cfg.pop("z_reference")
    (root / "config.json").write_text(json.dumps(cfg))

    reopened = EMProject(name="coax", base_dir=str(base))
    np.testing.assert_allclose(reopened.fds.fom._S_matrix, S_ok, atol=1e-12)
    np.testing.assert_allclose(reopened.fds.fom._Z_matrix, Z_ok, rtol=1e-12)


def test_rereduced_rom_ignores_older_saved_results(solved):
    """A fresh reduction must not hand back the previous model's saved S."""
    base, proj, _ = solved
    rom_dir = base / "coax" / "fds" / "fom" / "rom"
    with h5py.File(rom_dir / "s" / "s_global.h5", "w") as fh:     # a stale file
        H5Serializer.save_dataset(fh, "data", np.zeros((5, 4, 4), complex))
    rom = proj.fds.fom.reduce(tol=1e-10)
    rom.solve(**BAND)
    assert np.abs(rom._S_matrix).max() > 0.1
