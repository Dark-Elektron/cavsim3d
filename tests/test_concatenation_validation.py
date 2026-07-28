"""Concatenation input validation — port-mode compatibility.

Connected interface ports must carry the same number of port modes.  These
tests cover both a direct (unit) construction and the rectangular-waveguide
pipeline, checking that a mismatch fails with a clear, user-directed message
and that a matching case couples cleanly.
"""

import numpy as np
import pytest

# Import the solver chain first so the rom<->solvers modules finish initialising
# before we pull in ReducedStructure / ConcatenatedSystem (avoids a circular
# import when this is the first cavsim3d module loaded).
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver  # noqa: F401
from cavsim3d.rom.reduction import ModelOrderReduction  # noqa: F401
from cavsim3d.rom.structures import ReducedStructure
from cavsim3d.solvers.concatenation import ConcatenatedSystem


def _struct(domain, port_modes, r=4):
    """A minimal full-order-style ReducedStructure with W=I (no mesh needed)."""
    ncols = sum(len(m) for m in port_modes.values())
    return ReducedStructure(
        Ard=np.eye(r), Brd=np.zeros((r, ncols)),
        ports=list(port_modes.keys()), port_modes=port_modes,
        domain=domain, r=r, n_full=r, is_full_order=True,
        W=np.eye(r), Q_L_inv=np.eye(r),
    )


# ===========================================================================
# Unit: direct structures with a shared interface port
# ===========================================================================

class TestPortModeValidationUnit:

    def test_matching_modes_couple(self):
        # Both sides carry 2 modes on the shared 'junction' port.
        A = _struct("A", {"pA": {0: None}, "junction": {0: None, 1: None}})
        B = _struct("B", {"junction": {0: None, 1: None}, "pB": {0: None}})
        cs = ConcatenatedSystem(structures=[A, B])
        # Should not raise.
        cs.define_connections([((0, "junction"), (1, "junction"))])
        assert cs.n_connections == 1

    def test_mismatched_modes_raise_clear_error(self):
        # A.junction has 2 modes, B.junction has 1 -> must fail.
        A = _struct("A", {"pA": {0: None}, "junction": {0: None, 1: None}})
        B = _struct("B", {"junction": {0: None}, "pB": {0: None}})
        cs = ConcatenatedSystem(structures=[A, B])
        with pytest.raises(ValueError) as ei:
            cs.define_connections([((0, "junction"), (1, "junction"))])
        msg = str(ei.value)
        assert "port modes must match" in msg.lower()
        # Names the offending interfaces and their mode counts.
        assert "'A'.junction" in msg and "'B'.junction" in msg
        assert "2 mode" in msg and "1 mode" in msg


# ===========================================================================
# Integration: rectangular-waveguide sections through the ROM->concat pipeline
# ===========================================================================

class TestPortModeValidationRWG:
    A = 0.1
    L = 0.06
    MAXH = 0.05

    def _reduced_structure(self, nportmodes):
        """Solve one RWG section and return its (single) ReducedStructure."""
        from cavsim3d.geometry.primitives import RectangularWaveguide
        from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
        from cavsim3d.rom.reduction import ModelOrderReduction
        sec = RectangularWaveguide(a=self.A, L=self.L, maxh=self.MAXH)
        solver = FrequencyDomainSolver(sec, order=2)
        solver.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=4,
                                 nportmodes=nportmodes, per_domain=True,
                                 store_snapshots=True, global_method=None))
        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-9)
        return solver, rom.get_all_structures()[0]

    def test_matching_nportmodes_concatenate(self):
        # Two independently reduced RWG sections, both with 1 port mode.
        solA, sA = self._reduced_structure(1)
        _solB, sB = self._reduced_structure(1)
        # Rename ports so the two sections share an interface named 'iface'.
        sA = self._with_ports(sA, {"port1": "left", "port2": "iface"})
        sB = self._with_ports(sB, {"port1": "iface", "port2": "right"})
        cs = ConcatenatedSystem(
            structures=[sA, sB], mesh=sA.mesh, fes=sA.fes,
            port_impedance_func=solA.port_solver.get_port_wave_impedance)
        cs.define_connections([((0, "iface"), (1, "iface"))])
        cs.couple()
        assert cs.n_external_ports == 2

    def test_mismatched_nportmodes_raise(self):
        # One section reduced with 1 mode, the other with 3 -> interface clash.
        solA, sA = self._reduced_structure(1)
        _solB, sB = self._reduced_structure(3)
        sA = self._with_ports(sA, {"port1": "left", "port2": "iface"})
        sB = self._with_ports(sB, {"port1": "iface", "port2": "right"})
        cs = ConcatenatedSystem(
            structures=[sA, sB], mesh=sA.mesh, fes=sA.fes,
            port_impedance_func=solA.port_solver.get_port_wave_impedance)
        with pytest.raises(ValueError) as ei:
            cs.define_connections([((0, "iface"), (1, "iface"))])
        assert "port modes must match" in str(ei.value).lower()

    @staticmethod
    def _with_ports(struct, rename):
        """Return a copy of *struct* with ports renamed via *rename* map."""
        new_ports = [rename.get(p, p) for p in struct.ports]
        new_modes = {rename.get(p, p): m for p, m in struct.port_modes.items()}
        return ReducedStructure(
            Ard=struct.Ard, Brd=struct.Brd, ports=new_ports, port_modes=new_modes,
            domain=struct.domain, r=struct.r, n_full=struct.n_full,
            is_full_order=struct.is_full_order, W=struct.W, Q_L_inv=struct.Q_L_inv,
            fes=struct.fes, mesh=struct.mesh)


# ===========================================================================
# Import / reuse: reload a saved ROM's structures from disk (no live solver)
# ===========================================================================

class TestReducedStructureReuse:
    A = 0.1
    L = 0.06
    MAXH = 0.06

    def _solve_reduce(self):
        from cavsim3d.geometry.primitives import RectangularWaveguide
        sec = RectangularWaveguide(a=self.A, L=self.L, maxh=self.MAXH)
        solver = FrequencyDomainSolver(sec, order=2)
        solver.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1,
                                 per_domain=True, store_snapshots=True,
                                 global_method=None))
        mor = ModelOrderReduction(solver)
        mor.reduce(tol=1e-9)
        return solver, mor

    @staticmethod
    def _two_section(sA, sB, impedance):
        """Couple two sections A.right(iface) -> B.left(iface)."""
        def rn(s, rename, dom):
            return ReducedStructure(
                Ard=s.Ard, Brd=s.Brd,
                ports=[rename.get(p, p) for p in s.ports],
                port_modes={rename.get(p, p): m for p, m in s.port_modes.items()},
                domain=dom, r=s.r, n_full=s.n_full, is_full_order=s.is_full_order,
                W=s.W, Q_L_inv=s.Q_L_inv, fes=s.fes, mesh=s.mesh)
        A = rn(sA, {"port2": "iface"}, "A")
        B = rn(sB, {"port1": "iface"}, "B")
        cs = ConcatenatedSystem(structures=[A, B], mesh=sA.mesh, fes=sA.fes,
                                port_impedance_func=impedance)
        cs.define_connections([((0, "iface"), (1, "iface"))])
        cs.couple()
        return cs

    def test_save_writes_standalone_metadata(self, tmp_path):
        _solver, mor = self._solve_reduce()
        mor.save(str(tmp_path / "roms"))
        assert (tmp_path / "roms" / "structures.json").exists()
        assert (tmp_path / "roms" / "matrices").exists()

    def test_reused_from_disk_matches_live(self, tmp_path):
        from cavsim3d.rom.reduction import load_reduced_structures
        solver, mor = self._solve_reduce()
        mor.save(str(tmp_path / "roms"))
        ref = mor.get_all_structures()[0]

        # Live 2-section concatenation.
        cs_live = self._two_section(ref, ref,
                                    solver.port_solver.get_port_wave_impedance)
        S_live = cs_live.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=8))["S"]

        # Reload structures from disk WITHOUT a solver; impedance rebuilt too.
        structs, impf = load_reduced_structures(str(tmp_path / "roms"),
                                                fes=ref.fes, mesh=ref.mesh)
        assert len(structs) == 1 and impf is not None
        cs_disk = self._two_section(structs[0], structs[0], impf)
        S_disk = cs_disk.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=8))["S"]

        # Disk-reused reduced model reproduces the S-parameters exactly.
        assert np.allclose(S_live, S_disk, atol=1e-9)


# ===========================================================================
# Import an already-run PROJECT and concatenate it (load-if-exists)
# ===========================================================================

class TestProjectImport:
    A = 0.1
    L = 0.06
    MAXH = 0.06

    def _run_project(self, tmp_path, name):
        from cavsim3d.core.em_project import EMProject
        from cavsim3d.geometry.primitives import RectangularWaveguide
        proj = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
        proj.geometry = RectangularWaveguide(a=self.A, L=self.L, maxh=self.MAXH)
        proj.fds.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1))
        proj.fds.fom.reduce(tol=1e-9)       # auto-saves fds/fom/rom
        return tmp_path / name

    def test_reduce_autosaves_importable_rom(self, tmp_path):
        proj_dir = self._run_project(tmp_path, "rwgA")
        assert (proj_dir / "fds" / "fom" / "rom" / "structures.json").exists()

    def test_import_and_concatenate(self, tmp_path):
        from cavsim3d.rom.reduction import import_reduced_structures
        proj_dir = self._run_project(tmp_path, "rwgA")

        # Fresh import from the project folder — no live solver, no recompute.
        structs, impf = import_reduced_structures(str(proj_dir))
        assert len(structs) == 1 and impf is not None

        s = structs[0]

        def rn(src, rename, dom):
            return ReducedStructure(
                Ard=src.Ard, Brd=src.Brd,
                ports=[rename.get(p, p) for p in src.ports],
                port_modes={rename.get(p, p): m for p, m in src.port_modes.items()},
                domain=dom, r=src.r, n_full=src.n_full,
                is_full_order=src.is_full_order, W=src.W, Q_L_inv=src.Q_L_inv)

        cs = ConcatenatedSystem(
            structures=[rn(s, {"port2": "iface"}, "A"),
                        rn(s, {"port1": "iface"}, "B")],
            port_impedance_func=impf)
        cs.define_connections([((0, "iface"), (1, "iface"))])
        cs.couple()
        res = cs.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=6))
        assert cs.n_external_ports == 2
        assert np.all(np.isfinite(res["S"]))
        # matched uniform guide: |S21| ~ 1
        assert np.allclose(np.abs(res["S"][:, 1, 0]), 1.0, atol=0.05)

    def test_import_missing_rom_raises(self, tmp_path):
        from cavsim3d.rom.reduction import import_reduced_structures
        empty = tmp_path / "no_rom"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            import_reduced_structures(str(empty))


# ===========================================================================
# Load-if-exists / run-if-not resolution of a section's reduced model
# ===========================================================================

class TestLoadOrRun:
    A = 0.1
    L = 0.06
    MAXH = 0.06
    CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1)

    def _geo(self):
        from cavsim3d.geometry.primitives import RectangularWaveguide
        return RectangularWaveguide(a=self.A, L=self.L, maxh=self.MAXH)

    def test_run_then_reuse(self, tmp_path):
        from cavsim3d.core.reuse import load_or_run_reduced
        ppath = tmp_path / "section"

        # First call: nothing on disk -> runs FOM+ROM and saves.
        s1, imp1 = load_or_run_reduced(str(ppath), geometry=self._geo(),
                                       fom_config=self.CFG, order=2)
        assert len(s1) == 1 and imp1 is not None
        assert (ppath / "fds" / "fom" / "rom" / "structures.json").exists()

        # Second call: ROM exists -> imports, no geometry required, identical.
        s2, imp2 = load_or_run_reduced(str(ppath))
        assert len(s2) == 1
        assert np.allclose(s1[0].Ard, s2[0].Ard)
        assert np.allclose(s1[0].Brd, s2[0].Brd)

    def test_no_rom_no_geometry_raises(self, tmp_path):
        from cavsim3d.core.reuse import load_or_run_reduced
        with pytest.raises(ValueError):
            load_or_run_reduced(str(tmp_path / "missing"))   # nothing to load or run

    def test_force_recomputes(self, tmp_path):
        from cavsim3d.core.reuse import load_or_run_reduced
        ppath = tmp_path / "section"
        load_or_run_reduced(str(ppath), geometry=self._geo(),
                            fom_config=self.CFG, order=2)
        # force=True must accept (and require) a geometry and recompute.
        s, imp = load_or_run_reduced(str(ppath), geometry=self._geo(),
                                     fom_config=self.CFG, order=2, force=True)
        assert len(s) == 1 and imp is not None


# ===========================================================================
# Interface "physical fit" guards: per-mode fingerprint + training-band overlap
# ===========================================================================

class TestInterfaceFitChecks:
    MAXH = 0.05

    def _reduce(self, a=0.1, L=0.06, fmin=1.8, fmax=2.4):
        """Reduce one RWG section; the structure carries fingerprints + band."""
        from cavsim3d.geometry.primitives import RectangularWaveguide
        sec = RectangularWaveguide(a=a, L=L, maxh=self.MAXH)
        solver = FrequencyDomainSolver(sec, order=2)
        solver.solve(config=dict(fmin=fmin, fmax=fmax, nsamples=4, nportmodes=1,
                                 per_domain=True, store_snapshots=True,
                                 global_method=None))
        mor = ModelOrderReduction(solver)
        mor.reduce(tol=1e-9)
        return solver, mor.get_all_structures()[0]

    @staticmethod
    def _rn(s, rename, dom):
        ns = ReducedStructure(
            Ard=s.Ard, Brd=s.Brd,
            ports=[rename.get(p, p) for p in s.ports],
            port_modes={rename.get(p, p): m for p, m in s.port_modes.items()},
            domain=dom, r=s.r, n_full=s.n_full, is_full_order=s.is_full_order,
            W=s.W, Q_L_inv=s.Q_L_inv, fes=s.fes, mesh=s.mesh)
        fp = getattr(s, "port_fingerprints", None)
        if fp:
            ns.port_fingerprints = {rename.get(p, p): d for p, d in fp.items()}
        ns.training_band = getattr(s, "training_band", None)
        return ns

    def _couple(self, sA, sB, impedance):
        A = self._rn(sA, {"port2": "iface"}, "A")
        B = self._rn(sB, {"port1": "iface"}, "B")
        cs = ConcatenatedSystem(structures=[A, B], port_impedance_func=impedance)
        cs.define_connections([((0, "iface"), (1, "iface"))])   # fingerprint check
        cs.couple()                                             # band check
        return cs

    def test_matching_dims_and_band_couple(self):
        solA, sA = self._reduce(a=0.1, fmin=1.8, fmax=2.4)
        _solB, sB = self._reduce(a=0.1, fmin=1.8, fmax=2.4)
        cs = self._couple(sA, sB, solA.port_solver.get_port_wave_impedance)
        assert cs.n_external_ports == 2

    def test_dimension_mismatch_raises(self):
        # Different width a -> different TE10 cutoff kc -> modes don't correspond.
        solA, sA = self._reduce(a=0.10, fmin=1.8, fmax=2.4)
        _solB, sB = self._reduce(a=0.13, fmin=1.8, fmax=2.4)
        with pytest.raises(ValueError) as ei:
            self._couple(sA, sB, solA.port_solver.get_port_wave_impedance)
        msg = str(ei.value).lower()
        assert "do not correspond" in msg and ("cutoff" in msg or "cross-section" in msg)

    def test_disjoint_training_bands_raise(self):
        # Same geometry but reduced over non-overlapping bands -> cannot couple.
        solA, sA = self._reduce(a=0.1, fmin=1.8, fmax=2.4)
        _solB, sB = self._reduce(a=0.1, fmin=3.0, fmax=3.6)
        with pytest.raises(ValueError) as ei:
            self._couple(sA, sB, solA.port_solver.get_port_wave_impedance)
        assert "disjoint frequency band" in str(ei.value).lower()

    def test_extrapolation_beyond_band_warns(self):
        solA, sA = self._reduce(a=0.1, fmin=1.8, fmax=2.4)
        _solB, sB = self._reduce(a=0.1, fmin=1.8, fmax=2.4)
        cs = self._couple(sA, sB, solA.port_solver.get_port_wave_impedance)
        with pytest.warns(UserWarning, match="training band"):
            cs.solve(config=dict(fmin=1.0, fmax=3.0, nsamples=5))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
