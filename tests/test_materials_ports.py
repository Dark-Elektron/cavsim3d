"""Integration tests for material properties, PEC subtraction, and TEM port modes.

Tests use the c3794_1hc_1fpc_ports.stp model which contains:
- Solids: hook_top|lh4, hook_bottom|dh2, ceramic|ceramic, component1|solid1, coupler|solid
- Ports: Ports|port1, Ports|port2
"""

import pytest
import numpy as np
from pathlib import Path

from cavsim3d.geometry.importers import OCCImporter
from cavsim3d.core.constants import Z0

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
STEP_FILE = NOTEBOOKS_DIR / "c3794_1hc_1fpc_ports.stp"


def _file_available() -> bool:
    return STEP_FILE.exists()


skip_no_file = pytest.mark.skipif(
    not _file_available(), reason="c3794_1hc_1fpc_ports.stp not available"
)


# ============================================================
# Test: STEP label extraction
# ============================================================

class TestSTEPLabelExtraction:
    """Verify that BREP names from STEP file are preserved."""

    @skip_no_file
    def test_labels_extracted(self):
        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        solid_names = [s.name for s in geo.geo.solids]
        # Should have meaningful names, not just cell_N
        assert any(not name.startswith('cell_') for name in solid_names), \
            f"Expected STEP labels, got: {solid_names}"


# ============================================================
# Test: Material assignment and PEC subtraction
# ============================================================

class TestMaterialAssignment:
    """Verify set_materials() with PEC and dielectric materials."""

    @skip_no_file
    def test_set_materials_pec_removes_solids(self):
        """PEC solids should be removed from the geometry."""
        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        n_before = len(list(geo.geo.solids))

        geo.set_materials({
            'hook_top': 'PEC',
            'hook_bottom': 'PEC',
            'ceramic': {'eps_r': 12.0},
        })

        n_after = len(list(geo.geo.solids))
        assert n_after < n_before, \
            f"PEC subtraction should remove solids: {n_before} -> {n_after}"

    @skip_no_file
    def test_set_materials_eps_r_alias(self):
        """eps_r alias should be normalised to epsilon_r."""
        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        geo.set_materials({
            'ceramic': {'eps_r': 12.0},
        })

        mat = geo.get_material('ceramic')
        assert mat['eps_r'] == 12.0

    @skip_no_file
    def test_pec_material_returns_pec_flag(self):
        """get_material() for PEC domain should return PEC flag."""
        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        geo.set_materials({
            'hook_top': 'PEC',
        })

        mat = geo.get_material('hook_top')
        assert mat.get('PEC', False) is True


# ============================================================
# Test: eps_r CoefficientFunction in solver
# ============================================================

class TestEpsrCoefficientFunction:
    """Verify that the frequency domain solver builds eps_r CF correctly."""

    @skip_no_file
    def test_build_material_cfs(self):
        """_build_material_cfs should return CFs matching mesh materials."""
        geo = OCCImporter(str(STEP_FILE), unit='mm', maxh=0.05)
        # Subtract PEC solids so only a few non-PEC domains remain,
        # making the 2-port model compatible with the solver's port mapping.
        geo.set_materials({
            'hook_top': 'PEC',
            'hook_bottom': 'PEC',
            'ceramic': {'eps_r': 12.0},
        })
        geo.generate_mesh(maxh=0.05)

        from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
        fds = FrequencyDomainSolver(geo, order=2)
        eps_r_cf, mu_r_cf = fds._build_material_cfs()

        # CFs should be non-None
        assert eps_r_cf is not None
        assert mu_r_cf is not None


# ============================================================
# Test: TEM mode detection
# ============================================================

class TestTEMMode:
    """Verify TEM mode solver in PortEigenmodeSolver."""

    def test_tem_wave_impedance_is_z0(self):
        """TEM wave impedance should be Z0 at any frequency."""
        from cavsim3d.solvers.ports import PortEigenmodeSolver

        # Create a minimal mock to test impedance calculation
        solver = object.__new__(PortEigenmodeSolver)
        solver.port_cutoff_kc = {'test_port': {0: 0.0}}
        solver.port_mode_types = {'test_port': {0: 'TEM'}}

        z = solver.get_port_wave_impedance('test_port', 0, 1e9)
        assert abs(z - Z0) < 1e-10, f"TEM impedance should be Z0={Z0}, got {z}"

        z2 = solver.get_port_wave_impedance('test_port', 0, 5e9)
        assert abs(z2 - Z0) < 1e-10, "TEM impedance should be constant"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
