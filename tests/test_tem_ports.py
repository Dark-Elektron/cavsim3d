"""Tests for TEM / coaxial port detection and mode solving.

Uses unit-level tests for geometry fitting and, when the STEP file
``c3794_4hc_1fpc_w_TEM.stp`` is present, integration tests for
end-to-end port eigenmode solving.
"""

import pytest
import numpy as np
from pathlib import Path

from cavsim3d.solvers.ports import (
    PortEigenmodeSolver,
    PortGeometry,
    PortGeometryType,
    AnalyticMode,
)
from cavsim3d.core.constants import Z0

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
STEP_FILE = NOTEBOOKS_DIR / "c3794_4hc_1fpc_w_TEM.stp"


def _file_available() -> bool:
    return STEP_FILE.exists()


skip_no_file = pytest.mark.skipif(
    not _file_available(), reason="c3794_4hc_1fpc_w_TEM.stp not available"
)


# ============================================================
# Unit tests: coaxial geometry fitting
# ============================================================

class TestCoaxialFit:
    """Verify _fit_coaxial detects annular cross-sections."""

    @staticmethod
    def _make_solver():
        solver = object.__new__(PortEigenmodeSolver)
        solver.geometry_tolerance = 0.05
        return solver

    def test_annular_ring_detected(self):
        """An ideal annular ring should be detected as coaxial."""
        solver = self._make_solver()
        a, b = 0.02, 0.05  # inner / outer radius

        area = np.pi * (b**2 - a**2)
        I = np.pi * (b**4 - a**4) / 4
        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])

        geom, error = solver._fit_coaxial(center, normal, t1, t2, area, I, I, 0.0)

        assert geom is not None, "Should detect coaxial geometry"
        assert geom.type == PortGeometryType.COAXIAL
        assert error < 0.01, f"Fit error too large: {error}"
        assert abs(geom.inner_radius - a) < 1e-6
        assert abs(geom.radius - b) < 1e-6

    def test_solid_circle_not_coaxial(self):
        """A solid circular cross-section should NOT be detected as coaxial."""
        solver = self._make_solver()
        R = 0.05
        area = np.pi * R**2
        I = np.pi * R**4 / 4
        center = np.array([0.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])

        geom, error = solver._fit_coaxial(center, normal, t1, t2, area, I, I, 0.0)

        # For a solid circle, a_sq should be ≈ 0 which triggers a_sq <= 0 guard
        # or the error should be very large
        if geom is not None:
            assert error > 0.05, (
                f"Solid circle should have high coaxial error, got {error}"
            )

    def test_rectangle_not_coaxial(self):
        """A rectangular cross-section should NOT be detected as coaxial
        (non-isotropic moments)."""
        solver = self._make_solver()
        a_side, b_side = 0.1, 0.05
        area = a_side * b_side
        I_uu = a_side * b_side**3 / 12
        I_vv = a_side**3 * b_side / 12
        center = np.zeros(3)
        normal = np.array([0.0, 0.0, 1.0])
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])

        geom, error = solver._fit_coaxial(center, normal, t1, t2, area, I_uu, I_vv, 0.0)

        # Non-isotropic ⇒ should be rejected (None or high error)
        if geom is not None:
            assert error > 0.1, (
                f"Rectangle should fail coaxial fit, error={error}"
            )


# ============================================================
# Unit tests: coaxial mode generation
# ============================================================

class TestCoaxialModes:
    """Verify _generate_coaxial_modes produces expected mode list."""

    @staticmethod
    def _make_solver():
        solver = object.__new__(PortEigenmodeSolver)
        return solver

    def test_tem_is_first_mode(self):
        """TEM mode (kc=0) should always be the first generated mode."""
        solver = self._make_solver()
        geom = PortGeometry(
            type=PortGeometryType.COAXIAL,
            center=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            t1=np.array([1.0, 0.0, 0.0]),
            t2=np.array([0.0, 1.0, 0.0]),
            area=np.pi * (0.05**2 - 0.02**2),
            radius=0.05,
            inner_radius=0.02,
        )
        modes = solver._generate_coaxial_modes(geom, nmodes=3)

        assert len(modes) > 0
        assert modes[0].type == 'TEM'
        assert modes[0].kc == 0.0
        assert modes[0].indices == (0, 0)

    def test_higher_modes_have_positive_kc(self):
        """All modes after TEM should have kc > 0."""
        solver = self._make_solver()
        geom = PortGeometry(
            type=PortGeometryType.COAXIAL,
            center=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            t1=np.array([1.0, 0.0, 0.0]),
            t2=np.array([0.0, 1.0, 0.0]),
            area=np.pi * (0.05**2 - 0.02**2),
            radius=0.05,
            inner_radius=0.02,
        )
        modes = solver._generate_coaxial_modes(geom, nmodes=5)

        for mode in modes[1:]:
            assert mode.kc > 0, f"Higher-order mode {mode} should have kc > 0"

    def test_mode_types_present(self):
        """Should generate both TE and TM higher-order modes."""
        solver = self._make_solver()
        geom = PortGeometry(
            type=PortGeometryType.COAXIAL,
            center=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            t1=np.array([1.0, 0.0, 0.0]),
            t2=np.array([0.0, 1.0, 0.0]),
            area=np.pi * (0.05**2 - 0.02**2),
            radius=0.05,
            inner_radius=0.02,
        )
        modes = solver._generate_coaxial_modes(geom, nmodes=10)
        types = {m.type for m in modes}

        assert 'TEM' in types
        assert 'TE' in types
        assert 'TM' in types


# ============================================================
# Unit tests: TEM wave impedance
# ============================================================

class TestTEMImpedance:
    """TEM mode wave impedance should always be Z0."""

    def test_tem_impedance_is_z0(self):
        solver = object.__new__(PortEigenmodeSolver)
        solver.port_cutoff_kc = {'port1': {0: 0.0}}
        solver.port_mode_types = {'port1': {0: 'TEM'}}

        for freq in [1e8, 1e9, 5e9, 10e9]:
            z = solver.get_port_wave_impedance('port1', 0, freq)
            assert abs(z - Z0) < 1e-10, (
                f"TEM impedance at {freq:.0e} Hz: expected Z0={Z0}, got {z}"
            )


# ============================================================
# Unit tests: coaxial mode CF creation
# ============================================================

class TestCoaxialModeCF:
    """Verify _create_coaxial_mode_cf runs without error for each mode type."""

    @staticmethod
    def _make_solver():
        solver = object.__new__(PortEigenmodeSolver)
        return solver

    def test_tem_cf_creation(self):
        """TEM CoefficientFunction should be created without error."""
        solver = self._make_solver()
        geom = PortGeometry(
            type=PortGeometryType.COAXIAL,
            center=np.zeros(3),
            normal=np.array([0.0, 0.0, 1.0]),
            t1=np.array([1.0, 0.0, 0.0]),
            t2=np.array([0.0, 1.0, 0.0]),
            area=np.pi * (0.05**2 - 0.02**2),
            radius=0.05,
            inner_radius=0.02,
        )
        mode = AnalyticMode(type='TEM', indices=(0, 0), kc=0.0, degeneracy=1)
        cf = solver._create_coaxial_mode_cf(mode, geom)
        assert cf is not None
        # Should be a 3-component vector CF
        assert cf.dim == 3


# ============================================================
# Integration tests: end-to-end with STEP file
# ============================================================

class TestCoaxialPortIntegration:
    """Integration tests using c3794_4hc_1fpc_w_TEM.stp."""

    @skip_no_file
    def test_coaxial_port_detected(self):
        """At least one port should be detected as coaxial."""
        from cavsim3d.geometry.importers import OCCImporter

        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        geo.set_materials({
            'hook_top': 'PEC',
            'hook_bottom': 'PEC',
            'ceramic': {'eps_r': 12.0},
        })
        geo.generate_mesh(maxh=0.05)

        solver = PortEigenmodeSolver(geo.mesh, order=2, bc='default')
        ports = [b for b in geo.mesh.GetBoundaries() if 'port' in b.lower()]

        coaxial_ports = []
        for port in ports:
            geom = solver._detect_port_geometry(port)
            if geom.type == PortGeometryType.COAXIAL:
                coaxial_ports.append(port)

        assert len(coaxial_ports) >= 1, (
            f"Expected at least 1 coaxial port, found: "
            f"{[(p, solver._detect_port_geometry(p).type.value) for p in ports]}"
        )

    @skip_no_file
    def test_solve_includes_tem_mode(self):
        """Solving port modes should include a TEM mode on the coaxial port."""
        from cavsim3d.geometry.importers import OCCImporter

        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        geo.set_materials({
            'hook_top': 'PEC',
            'hook_bottom': 'PEC',
            'ceramic': {'eps_r': 12.0},
        })
        geo.generate_mesh(maxh=0.05)

        solver = PortEigenmodeSolver(geo.mesh, order=2, bc='default')
        solver.solve(nmodes=2)

        # Check that at least one port has a TEM mode
        tem_found = False
        for port, types in solver.port_mode_types.items():
            for mode_idx, mtype in types.items():
                if mtype == 'TEM':
                    tem_found = True
                    # Verify kc = 0
                    kc = solver.port_cutoff_kc[port][mode_idx]
                    assert kc == 0.0, f"TEM mode on {port} should have kc=0, got {kc}"
                    break

        assert tem_found, (
            "Expected at least one TEM mode across all ports. "
            f"Mode types: {solver.port_mode_types}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
