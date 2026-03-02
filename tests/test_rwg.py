"""Tests for rectangular waveguide analysis."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometry.primitives import RectangularWaveguide
from solvers.frequency_domain import FrequencyDomainSolver
from rom.reduction import ModelOrderReduction
from analytical.rectangular_waveguide import RWGAnalytical


class TestRectangularWaveguide:
    """Test suite for rectangular waveguide simulations."""

    @pytest.fixture
    def rwg_geometry(self):
        """Create standard test geometry."""
        a = 100e-3   # 100 mm width
        L = 200e-3   # 200 mm length
        maxh = 0.04  # Mesh size
        return RectangularWaveguide(a=a, L=L, maxh=maxh)

    @pytest.fixture
    def analytical_solution(self, rwg_geometry):
        """Create analytical solution."""
        return RWGAnalytical(
            a=rwg_geometry.a,
            L=rwg_geometry.L,
            b=rwg_geometry.b
        )

    def test_geometry_creation(self, rwg_geometry):
        """Test geometry is created correctly."""
        assert rwg_geometry.mesh is not None
        assert rwg_geometry.a == 100e-3
        assert rwg_geometry.b == 50e-3  # Default b = a/2
        assert rwg_geometry.L == 200e-3

    def test_cutoff_frequency(self, rwg_geometry, analytical_solution):
        """Test cutoff frequency calculation."""
        fc_geo = rwg_geometry.cutoff_frequency_TE10
        fc_ana = analytical_solution.cutoff_frequency

        assert np.isclose(fc_geo, fc_ana, rtol=1e-10)

        # TE10 cutoff for a=100mm should be ~1.5 GHz
        expected_fc = 299792458 / (2 * 0.1)  # c / (2a)
        assert np.isclose(fc_geo, expected_fc, rtol=1e-10)

    def test_z_parameters_below_cutoff(self, rwg_geometry, analytical_solution):
        """Test Z-parameters below cutoff are purely imaginary."""
        solver = FrequencyDomainSolver(rwg_geometry, order=3)
        solver.assemble_matrices()

        # Frequency below cutoff
        fc = analytical_solution.cutoff_frequency
        freq = 0.5 * fc  # Well below cutoff

        Z_ana = analytical_solution.z_parameters(np.array([freq]))

        # Below cutoff, Z should be purely imaginary (evanescent mode)
        assert np.abs(np.real(Z_ana['Z11'][0])) < np.abs(np.imag(Z_ana['Z11'][0])) * 0.01

    def test_z_parameters_comparison(self, rwg_geometry, analytical_solution):
        """Compare numerical and analytical Z-parameters."""
        solver = FrequencyDomainSolver(rwg_geometry, order=3)

        # Frequency range above cutoff
        fc = analytical_solution.cutoff_frequency
        fmin = 1.1 * fc / 1e9  # GHz
        fmax = 2.0 * fc / 1e9  # GHz

        results = solver.solve(fmin=fmin, fmax=fmax, nsamples=20)
        Z_ana = analytical_solution.z_parameters(solver.frequencies)

        # Compare Z11
        Z11_num = solver.get_param('Z', '1(1)1(1)')
        Z11_ana = Z_ana['Z11']

        relative_error = np.abs(Z11_num - Z11_ana) / np.abs(Z11_ana)
        max_error = np.max(relative_error)

        # Allow 10% error due to mesh coarseness
        assert max_error < 0.1, f"Max relative error: {max_error:.2%}"


class TestROMAccuracy:
    """Test ROM accuracy and convergence."""

    @pytest.fixture
    def setup_rom(self):
        """Setup ROM test case."""
        a = 100e-3
        L = 200e-3
        rwg = RectangularWaveguide(a=a, L=L, maxh=0.04)

        solver = FrequencyDomainSolver(rwg, order=3)
        solver.assemble_matrices()
        solver.solve(fmin=1.5, fmax=3.0, nsamples=50, store_snapshots=True)

        return solver

    def test_svd_decay(self, setup_rom):
        """Test that singular values decay rapidly."""
        solver = setup_rom
        domain = solver.domains[0]

        U, S, Vt = np.linalg.svd(solver.snapshots[domain], full_matrices=False)

        # Singular values should decay
        assert S[0] > S[-1]

        # Check decay rate (should be exponential-ish for smooth solutions)
        decay_ratio = S[10] / S[0] if len(S) > 10 else S[-1] / S[0]
        assert decay_ratio < 0.1, "Singular values not decaying fast enough"

    def test_rom_reduction(self, setup_rom):
        """Test ROM reduction and solution."""
        solver = setup_rom

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        # Check reduction happened
        assert rom.total_reduced_dofs < rom.total_dofs
        assert rom.compression_ratio > 0

        # Solve ROM
        results = rom.solve(fmin=1.5, fmax=3.0, nsamples=100)

        # Results should be well-formed
        assert results['Z'].shape == (100, 2, 2)
        assert np.all(np.isfinite(results['Z']))
        assert np.any(np.abs(results['Z']) > 0)

    def test_rom_vs_fom(self, setup_rom):
        """Test ROM accuracy against FOM."""
        solver = setup_rom

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)
        rom.solve(fmin=1.5, fmax=3.0, nsamples=50)

        # Compute error
        errors = rom.compute_error(solver)

        # All S-parameter errors should be small
        for key, err in errors.items():
            assert err < 0.05, f"ROM error too large for {key}: {err:.2%}"

    def test_rom_eigenvalues(self, setup_rom):
        """Test ROM eigenfrequencies."""
        solver = setup_rom

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        # Get eigenfrequencies
        freqs_rom = rom.get_resonant_frequencies() / 1e9  # GHz
        freqs_fom = solver.get_resonant_frequencies() / 1e9

        # At least some eigenfrequencies should be positive
        assert len(freqs_rom) > 0
        assert len(freqs_fom) > 0

        # First few ROM eigenfrequencies should match FOM
        n_compare = min(5, len(freqs_rom), len(freqs_fom))
        for i in range(n_compare):
            rel_diff = abs(freqs_rom[i] - freqs_fom[i]) / freqs_fom[i]
            assert rel_diff < 0.1, (
                f"Mode {i}: ROM {freqs_rom[i]:.4f} vs FOM {freqs_fom[i]:.4f} "
                f"(diff: {rel_diff:.2%})"
            )

    def test_singular_value_plot(self, setup_rom):
        """Test singular value plot doesn't error."""
        solver = setup_rom

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for tests

        fig, axes = rom.plot_singular_values()
        assert fig is not None


class TestConcatenation:
    """Test ROM structure concatenation."""

    def test_rom_concatenation(self):
        """Test concatenating two independently reduced ROMs."""
        a = 100e-3
        L = 100e-3  # Each cell is 100mm

        # Create two identical structures
        rwg1 = RectangularWaveguide(a=a, L=L, maxh=0.05)
        rwg2 = RectangularWaveguide(a=a, L=L, maxh=0.05)

        solver1 = FrequencyDomainSolver(rwg1, order=2)
        solver2 = FrequencyDomainSolver(rwg2, order=2)

        for solver in [solver1, solver2]:
            solver.assemble_matrices()
            solver.solve(fmin=1.5, fmax=3.0, nsamples=20, store_snapshots=True)

        # Reduce each
        rom1 = ModelOrderReduction(solver1)
        rom1.reduce(tol=1e-6)

        rom2 = ModelOrderReduction(solver2)
        rom2.reduce(tol=1e-6)

        # Concatenate
        concat = rom1.concatenate(
            others=[rom2],
            connections=[((0, 'port2'), (1, 'port1'))]
        )

        # Solve concatenated system
        result = concat.solve(fmin=1.5, fmax=3.0, nsamples=50)

        # Should produce a 2-port system
        assert result['Z'].shape == (50, 2, 2)
        assert np.all(np.isfinite(result['Z']))

        # Compare with analytical for L=200mm (loose tolerance)
        rwg_full = RWGAnalytical(a=a, L=2 * L)
        Z_ana = rwg_full.z_parameters(result['frequencies'])
        Z11_concat = result['Z'][:, 0, 0]
        Z11_ana = Z_ana['Z11']

        # Check that at least the rough magnitude is similar
        mag_concat = np.mean(np.abs(Z11_concat))
        mag_ana = np.mean(np.abs(Z11_ana))
        assert mag_concat > 0, "Concatenated Z11 is zero"
        assert abs(np.log10(mag_concat / mag_ana)) < 1, (
            f"Z11 magnitudes differ by more than 10x: "
            f"concat={mag_concat:.2e}, ana={mag_ana:.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])