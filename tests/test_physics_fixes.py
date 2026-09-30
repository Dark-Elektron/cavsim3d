"""Regression tests for physics / API behaviour (small models, fast).

Covers: port media (eps_r, mu_r) in the modal impedance, lossy materials through
FOM -> ROM -> concatenation, the port reference frame at joins, resonant modes,
numeric TEM detection, the solve() rerun policy, Touchstone export, assembly
layout, and the analytic cavity references.
"""

import numpy as np
import pytest

import cavsim3d.core.em_project  # noqa: F401  (import order: solvers before rom)
from cavsim3d.core.constants import c0, eps0
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.geometry.assembly import Assembly
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.rom.reduction import ModelOrderReduction

A, B, L = 0.02286, 0.01016, 0.03


def _guide(length=L, maxh=0.006, materials=None):
    geo = RectangularWaveguide(a=A, b=B, L=length, maxh=maxh)
    if materials:
        geo.set_materials(materials)
    return geo


def _solve(geo, fmin, fmax, n=4, order=2, **kw):
    fds = FrequencyDomainSolver(geo, order=order)
    fds.solve(fmin=fmin, fmax=fmax, nsamples=n, solver_type='direct',
              nportmodes=kw.pop('nportmodes', 1), **kw)
    return fds


class TestPortMedium:
    def test_dielectric_filled_guide_is_matched(self):
        """A uniform eps_r-filled guide referenced to its own wave impedance
        must be matched: the cutoff moves with the filling."""
        eps = 2.0
        fc = c0 / (2 * A) / np.sqrt(eps)
        fds = _solve(_guide(materials={'*': {'eps_r': eps}}),
                     1.3 * fc / 1e9, 1.9 * fc / 1e9, order=3)
        S = fds._S_matrix
        assert np.max(np.abs(S[:, 0, 0])) < 0.02
        assert np.allclose(np.abs(S[:, 1, 0]), 1.0, atol=0.02)

    def test_mu_r_enters_the_modal_impedance(self):
        from cavsim3d.solvers.ports import modal_wave_impedance
        f, kc = 10e9, np.pi / A
        z = modal_wave_impedance('TE', kc, f, eps_r=2.0, mu_r=3.0)
        mu, eps = 3.0 * 4e-7 * np.pi, 2.0 * eps0
        gamma = np.sqrt(kc ** 2 - (2 * np.pi * f) ** 2 * mu * eps + 0j)
        assert np.isclose(z, 1j * 2 * np.pi * f * mu / gamma, rtol=1e-9)

    def test_evanescent_te_impedance_is_inductive(self):
        from cavsim3d.solvers.ports import modal_wave_impedance
        z = modal_wave_impedance('TE', np.pi / A, 0.5 * c0 / (2 * A))
        assert abs(z.real) < 1e-9 and z.imag > 0


class TestLossyMaterials:
    EPS, TAND, SIG = 2.0, 0.02, 0.05

    def _analytic_s21(self, f, length):
        w = 2 * np.pi * f
        eps_c = self.EPS * (1 - 1j * self.TAND) - 1j * self.SIG / (w * eps0)
        gamma = np.sqrt((np.pi / A) ** 2 - (w / c0) ** 2 * eps_c + 0j)
        return np.exp(-gamma.real * length)

    def test_fom_rom_and_concat_match_analytic_attenuation(self):
        mat = {'*': {'eps_r': self.EPS, 'tan_delta': self.TAND, 'sigma': self.SIG}}
        length = 0.04
        fds = _solve(_guide(length, 0.005, mat), 8.0, 11.0, n=5, order=3)
        S = fds._S_matrix
        assert fds._fes_global.is_complex
        assert np.allclose(np.abs(S[:, 1, 0]),
                           self._analytic_s21(fds.frequencies, length), rtol=0.01)
        assert np.allclose(S[:, 0, 1], S[:, 1, 0], atol=1e-9)        # reciprocal
        assert np.all(np.abs(S[:, 0, 0]) ** 2 + np.abs(S[:, 1, 0]) ** 2 < 1)

        rom = ModelOrderReduction(fds).reduce(tol=1e-10)
        res = rom.solve(fmin=8.0, fmax=11.0, nsamples=5)
        assert np.max(np.abs(res['S'] - S)) < 1e-8

        asm = Assembly('Z')
        asm.add('h1', _guide(length / 2, 0.005))
        asm.add('h2', _guide(length / 2, 0.005), after='h1')
        asm.build()
        asm.generate_mesh(maxh=0.005)
        asm.set_materials(mat)
        fds2 = _solve(asm, 8.0, 11.0, n=5, order=3, per_domain=True, global_method=None)
        concat = fds2.foms.reduce(tol=1e-10).concatenate()
        assert concat.is_lossy
        rc = concat.solve(fmin=8.0, fmax=11.0, nsamples=5)
        assert np.max(np.abs(rc['S'][:, 1, 0] - S[:, 1, 0])) < 5e-3

    def test_dielectric_loss_alone(self):
        """tan(delta) without conductivity: only the D matrix exists."""
        fds = _solve(_guide(materials={'*': {'eps_r': 1.5, 'tan_delta': 0.01}}), 8.0, 9.0, n=2)
        assert fds.C_global is None and fds.D_global is not None
        S = fds._S_matrix
        assert np.all(np.abs(S[:, 0, 0]) ** 2 + np.abs(S[:, 1, 0]) ** 2 < 1)

    def test_negative_loss_is_rejected(self):
        # rejected when the material is assigned, before any solve
        with pytest.raises(ValueError, match="must be >= 0"):
            _solve(_guide(materials={'*': {'tan_delta': -0.1}}), 8.0, 9.0, n=2)


class TestFieldReconstruction:
    @pytest.mark.parametrize("materials", [None, {'*': {'eps_r': 1.5, 'tan_delta': 0.01}}],
                             ids=["lossless", "lossy"])
    def test_rom_field_holds_the_reduced_solution(self, materials):
        """The ROM field GridFunction is exactly the reconstructed solution:
        real for a lossless model (ports driven on open circuits), complex,
        with its phase, for a lossy one."""
        fds = _solve(_guide(materials=materials), 8.0, 11.0, n=5)
        rom = ModelOrderReduction(fds).reduce(tol=1e-10)
        rom.solve(fmin=8.0, fmax=11.0, nsamples=5)
        x = rom.reconstruct_field(freq_idx=2, excitation_port='port1')
        E = rom._reconstruct_field_gf(2, 'port1')
        assert E.space.is_complex == (materials is not None)
        if materials is None:
            assert not np.any(np.imag(x))
        else:
            assert np.abs(np.imag(x)).max() > 0.1 * np.abs(x).max()
        assert np.allclose(E.vec.FV().NumPy(), x, rtol=0, atol=1e-12 * np.abs(x).max())


class TestPortFrameAtJoins:
    def test_odd_modes_have_the_same_sign_on_both_faces(self):
        from ngsolve import BND
        fds = FrequencyDomainSolver(_guide(maxh=0.004), order=3)
        fds.assemble_matrices(nportmodes=4)
        ps, mesh = fds.port_solver, fds.mesh
        x0, y0 = 0.3 * A, 0.4 * B
        for m in range(4):
            e1 = np.array(ps.port_modes['port1'][m](mesh(x0, y0, 0.0, BND)))
            e2 = np.array(ps.port_modes['port2'][m](mesh(x0, y0, L, BND)))
            assert np.allclose(e1, e2, atol=1e-6 * max(1.0, np.abs(e1).max())), \
                ps.get_mode_name('port1', m)


class TestResonantModes:
    def test_single_solid_matches_pmc_end_cap_spectrum(self):
        from cavsim3d.analytical import RWGAnalytical
        fds = FrequencyDomainSolver(_guide(), order=2)
        fds.assemble_matrices(nportmodes=1)
        f = fds.get_resonant_frequencies(n_modes=4)   # used to raise KeyError
        exact = np.array(list(RWGAnalytical(a=A, b=B, L=L)
                              .all_eigenfrequencies(n_modes=4).values())) * 1e9
        assert np.allclose(f, exact, rtol=0.01)

    def test_closed_pec_box_rules(self):
        from cavsim3d.analytical import RWGAnalytical
        pec = RWGAnalytical(a=A, b=B, L=L).all_eigenfrequencies(n_modes=6, boundary_type='PEC')
        assert 'TE100' not in pec and 'TE101' in pec     # TE needs p >= 1
        assert 'TM110' in pec                            # TM allows p = 0


class TestNumericPortModes:
    def test_coax_has_one_tem_mode(self):
        from netgen.occ import Cylinder, Axes, Z, OCCGeometry
        from ngsolve import Mesh
        from cavsim3d.solvers.ports import PortEigenmodeSolver
        a, b = 1.5e-3, 3.5e-3
        coax = (Cylinder(Axes((0, 0, 0), Z), r=b, h=6e-3)
                - Cylinder(Axes((0, 0, 0), Z), r=a, h=6e-3))
        for f in coax.faces:
            f.name = 'default'
        coax.faces.Min(Z).name = 'port1'
        coax.faces.Max(Z).name = 'port2'
        mesh = Mesh(OCCGeometry(coax).GenerateMesh(maxh=1e-3))
        mesh.Curve(3)
        ps = PortEigenmodeSolver(mesh, order=3, bc='default', mode_source='numeric')
        ps.solve(nmodes=2)
        types = [ps.port_mode_types['port1'][m] for m in range(2)]
        assert types == ['TEM', 'TE']

    @staticmethod
    def _coarse_guide_mesh():
        # 100 x 50 mm guide at maxh 0.06: a port face of a few triangles
        from netgen.occ import Box, Pnt, Z, OCCGeometry
        from ngsolve import Mesh
        box = Box(Pnt(0, 0, 0), Pnt(0.1, 0.05, 0.0667))
        for f in box.faces:
            f.name = 'default'
        box.faces.Min(Z).name = 'port1'
        box.faces.Max(Z).name = 'port2'
        return Mesh(OCCGeometry(box).GenerateMesh(maxh=0.06))

    def test_coarse_port_face(self):
        # the eigensolver used to ask for more modes than the face holds and
        # failed in scipy's eigh ("leading minor ... not positive definite")
        from cavsim3d.solvers.ports import PortEigenmodeSolver
        ps = PortEigenmodeSolver(self._coarse_guide_mesh(), order=2, bc='default',
                                 mode_source='numeric')
        ps.solve(nmodes=2)
        assert ps.port_mode_types['port1'][0] == 'TE'
        assert np.isclose(ps.port_cutoff_kc['port1'][0], np.pi / 0.1, rtol=0.05)   # TE10
        assert len(ps.port_modes['port2']) == 2

    def test_too_few_modes_on_the_port_face_is_an_error(self):
        # order 1 on this face: 3 TE + 3 TM degrees of freedom, so 6 modes at most
        from cavsim3d.solvers.ports import PortEigenmodeSolver
        ps = PortEigenmodeSolver(self._coarse_guide_mesh(), order=1, bc='default',
                                 mode_source='numeric')
        with pytest.raises(ValueError, match="resolves .* numeric port mode"):
            ps.solve(nmodes=8)


class TestRerunPolicy:
    def test_changed_band_recomputes_and_same_band_reuses(self):
        fds = _solve(_guide(), 8.0, 9.0, n=3)
        first = fds._S_matrix.copy()
        fds.solve(fmin=8.0, fmax=9.0, nsamples=3, solver_type='direct')
        assert np.array_equal(fds._S_matrix, first)                  # reused
        fds.solve(fmin=10.0, fmax=11.0, nsamples=3, solver_type='direct')
        assert np.isclose(fds.frequencies[0], 10e9)                  # recomputed
        fds.solve(fmin=8.0, fmax=9.0, nsamples=3, solver_type='direct', rerun=False)
        assert np.isclose(fds.frequencies[0], 10e9)                  # kept stored

    def test_zero_frequency_is_rejected(self):
        with pytest.raises(ValueError, match="fmin must be > 0"):
            _solve(_guide(), 0.0, 9.0, n=2)


class TestExportAndErrors:
    def test_rom_error_against_a_fom_on_another_grid(self):
        # a ROM swept on a finer grid than its FOM used to fail in np.allclose
        fds = _solve(_guide(), 8.0, 9.0, n=3, store_snapshots=True)
        rom = fds.fom.reduce(tol=1e-10)
        rom.solve(fmin=8.0, fmax=9.0, nsamples=7, solver_type='direct')
        errors = rom.compute_error(fds)
        assert all(np.isfinite(e) for e in errors.values())
        assert errors['1(1)2(1)'] < 1e-3                      # S21 of the guide

    def test_touchstone_two_port_column_order(self, tmp_path):
        fds = _solve(_guide(), 8.0, 9.0, n=2)
        S = fds._S_matrix
        # z0=None writes S as solved (and warns that R 50 is only nominal)
        with pytest.warns(UserWarning, match="R 50"):
            fn = fds.export_touchstone(str(tmp_path / "rwg"), z0=None)
        vals = np.array(open(fn).read().splitlines()[-1].split()[1:], float).reshape(-1, 2)
        expect = np.abs([S[-1, 0, 0], S[-1, 1, 0], S[-1, 0, 1], S[-1, 1, 1]])
        assert np.allclose(vals[:, 0], expect)
        fn50 = fds.export_touchstone(str(tmp_path / "rwg50"))     # default: 50 ohm
        assert "# GHz S MA R 50.0" in open(fn50).read()

    def test_compute_error_covers_every_entry(self):
        fds = _solve(_guide(), 8.0, 9.0, n=2)
        err = fds.compute_error(fds)
        assert len(err) == fds._S_matrix.shape[1] ** 2
        assert all(v == 0 for v in err.values())


class TestAssemblyLayout:
    def test_unconnected_components_follow_their_predecessor(self):
        asm = Assembly('Z')
        for k in ('a', 'b', 'c', 'd'):
            asm.add(k, _guide(maxh=0.03))
        asm.compute_layout()
        z = [asm._components[k].transform.translation[2] for k in 'abcd']
        assert np.allclose(z, [0, L, 2 * L, 3 * L], atol=1e-9)

    def test_before_places_the_component_first(self):
        asm = Assembly('Z')
        asm.add('mid', _guide(maxh=0.03))
        asm.add('first', _guide(maxh=0.03), before='mid')
        asm.compute_layout()
        z_first = asm._components['first'].transform.translation[2]
        z_mid = asm._components['mid'].transform.translation[2]
        assert np.isclose(z_mid - z_first, L)


class TestProjectPersistence:
    def test_solve_then_reopen_restores_everything(self, tmp_path):
        from cavsim3d.core.em_project import EMProject
        p = EMProject('rt', base_dir=str(tmp_path), overwrite=True)
        p.geometry = _guide(materials={'*': {'eps_r': 1.5}})
        p.fds.solve(fmin=8.0, fmax=9.0, nsamples=2, order=2, solver_type='direct')
        S = p.fds.fom._S_matrix
        p2 = EMProject('rt', base_dir=str(tmp_path))
        assert p2.geometry is not None
        assert p2.geometry.get_material('vacuum')['eps_r'] == 1.5
        assert np.allclose(p2.fds.fom._S_matrix, S)
        assert p2.fds.port_solver.port_media_eps       # dielectric port restored

    def test_imported_geometry_reopens_from_relative_base_dir(self, tmp_path, monkeypatch):
        from pathlib import Path
        from cavsim3d.core.em_project import EMProject
        step = (Path(__file__).resolve().parents[1] / "docs" / "example_models"
                / "rectangular_waveguide.step")
        monkeypatch.chdir(tmp_path)
        p = EMProject('rel', base_dir='sims', overwrite=True)
        geo = p.import_geometry(step, unit='m', auto_build=False)
        geo.set_materials({'*': {'eps_r': 1.5}})
        geo.build()
        geo.generate_mesh(maxh=0.05)
        p.fds.solve(fmin=2.0, fmax=3.0, nsamples=2, order=1, solver_type='direct')
        p2 = EMProject('rel', base_dir='sims')
        assert p2.geometry is not None          # used to fail: path joined twice
        assert p2.geometry.mesh.ne == geo.mesh.ne
        assert np.allclose(p2.fds.fom._S_matrix, p.fds.fom._S_matrix)

    def test_non_interactive_confirmation_declines(self):
        from cavsim3d.utils.io_utils import get_user_confirmation
        assert get_user_confirmation("delete everything?") is False
