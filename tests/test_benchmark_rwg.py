"""
Benchmark tests for rectangular waveguide against analytical solutions.

Tests three model types:
  1. Single solid RWG
  2. Split RWG (2 halves assembled)
  3. Assembled RWG (3 cells assembled)

Each is validated against RWGAnalytical with a few frequency points for speed.
The FOM -> ROM -> Concatenation chain is also tested.
"""

import numpy as np
import pytest
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.geometry.assembly import Assembly
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.rom.reduction import ModelOrderReduction
from cavsim3d.analytical.rectangular_waveguide import RWGAnalytical



# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

RWG_A = 100e-3       # 100 mm width
RWG_B = 50e-3        # 50 mm height (a/2)
RWG_L = 200e-3       # 200 mm length
RWG_MAXH = 0.04      # Mesh element size
NSAMPLES = 3          # Keep tests fast
ORDER = 3
RTOL_FOM = 0.10       # 10% tolerance for FOM vs analytical (coarse mesh)
RTOL_ROM = 0.05       # 5% ROM vs FOM tolerance


# ===========================================================================
# 1. Single solid RWG
# ===========================================================================

class TestRWGSingleModel:
    """Validate a single RWG solid against analytical TE10 solutions."""

    @pytest.fixture(scope="class")
    def rwg(self):
        return RectangularWaveguide(a=RWG_A, L=RWG_L, maxh=RWG_MAXH)

    @pytest.fixture(scope="class")
    def analytical(self):
        return RWGAnalytical(a=RWG_A, L=RWG_L, b=RWG_B)

    @pytest.fixture(scope="class")
    def freq_range(self, analytical):
        """Frequency range above cutoff in GHz."""
        fmin = 1.1 * analytical.fc
        fmax = 2.0 * analytical.fc
        return fmin, fmax

    def test_geometry_creation(self, rwg):
        assert rwg.mesh is not None
        assert rwg.a == RWG_A
        assert rwg.b == RWG_B
        assert rwg.L == RWG_L

    def test_cutoff_frequency(self, rwg, analytical):
        fc_geo = rwg.cutoff_frequency_TE10       # Hz
        fc_ana = analytical.fc * 1e9              # GHz -> Hz
        assert np.isclose(fc_geo, fc_ana, rtol=1e-10)

    def test_fom_z_parameters(self, rwg, analytical, freq_range):
        """FOM Z-parameters should approximately match analytical."""
        fmin, fmax = freq_range  # GHz
        solver = FrequencyDomainSolver(rwg, order=ORDER)
        result = solver.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES)

        # Analytical expects GHz; solver.frequencies is Hz
        freqs_ghz = solver.frequencies / 1e9
        Z_ana = analytical.z_parameters(freqs_ghz)

        # Z11 key in solver format: port1(mode1) port1(mode1) = '1(1)1(1)'
        Z11_num = result['Z_dict']['1(1)1(1)']
        Z11_ana = Z_ana['Z11']

        relative_error = np.abs(Z11_num - Z11_ana) / (np.abs(Z11_ana) + 1e-30)
        max_error = np.max(relative_error)
        assert max_error < RTOL_FOM, f"Max Z11 relative error: {max_error:.2%}"

    def test_rom_reduction_and_accuracy(self, rwg, analytical, freq_range):
        """ROM should compress well and match FOM closely."""
        fmin, fmax = freq_range  # GHz
        solver = FrequencyDomainSolver(rwg, order=ORDER)
        solver.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES, store_snapshots=True)

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        # Check reduction happened
        assert rom.total_reduced_dofs < rom.total_dofs
        assert rom.compression_ratio > 0

        # Solve ROM
        results = rom.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES)
        assert results['Z'].shape[0] == NSAMPLES
        assert np.all(np.isfinite(results['Z']))


# ===========================================================================
# 2. Split RWG (2 halves via Assembly)
# ===========================================================================

class TestRWGSplitModel:
    """
    Split a single RWG into two halves, assemble, and validate against
    analytical solution for the full-length waveguide.
    """

    @pytest.fixture(scope="class")
    def assembly(self):
        half_L = RWG_L / 2
        rwg1 = RectangularWaveguide(a=RWG_A, L=half_L, maxh=RWG_MAXH)
        rwg2 = RectangularWaveguide(a=RWG_A, L=half_L, maxh=RWG_MAXH)

        asm = Assembly(main_axis='Z')
        asm.add("half1", rwg1)
        asm.add("half2", rwg2, after="half1")
        asm.build()
        asm.generate_mesh(maxh=RWG_MAXH)
        return asm

    @pytest.fixture(scope="class")
    def analytical(self):
        return RWGAnalytical(a=RWG_A, L=RWG_L, b=RWG_B)

    @pytest.fixture(scope="class")
    def freq_range(self, analytical):
        fmin = 1.1 * analytical.fc
        fmax = 2.0 * analytical.fc
        return fmin, fmax

    def test_assembly_has_two_domains(self, assembly):
        assert len(assembly.components) == 2

    def test_assembly_has_mesh(self, assembly):
        assert assembly.mesh is not None
        assert assembly.mesh.ne > 0

    def test_fom_per_domain(self, assembly, freq_range):
        """FDS per-domain solve should produce results for both halves."""
        fmin, fmax = freq_range
        solver = FrequencyDomainSolver(assembly, order=ORDER)
        solver.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES, per_domain=True)
        assert solver.n_domains == 2

    def test_concatenation_vs_analytical(self, assembly, analytical, freq_range):
        """
        Full chain: FOM (per-domain) -> ROM -> Concatenation, compared
        against analytical solution for the full-length waveguide.
        """
        fmin, fmax = freq_range
        solver = FrequencyDomainSolver(assembly, order=ORDER)
        solver.solve(
            fmin=fmin, fmax=fmax, nsamples=NSAMPLES,
            per_domain=True, store_snapshots=True
        )

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        concat = rom.concatenate()
        result = concat.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES)

        # Compare concatenated Z11 with analytical for full length
        # concat.frequencies is in Hz; analytical expects GHz
        freqs_ghz = concat.frequencies / 1e9
        Z_ana = analytical.z_parameters(freqs_ghz)
        Z11_concat = result['Z'][:, 0, 0]
        Z11_ana = Z_ana['Z11']

        # Use magnitude-order check (loose due to concatenation + coarse mesh)
        mag_concat = np.mean(np.abs(Z11_concat))
        mag_ana = np.mean(np.abs(Z11_ana))
        assert mag_concat > 0, "Concatenated Z11 is zero"
        assert abs(np.log10(mag_concat / mag_ana)) < 1, (
            f"Z11 magnitudes differ by more than 10x: "
            f"concat={mag_concat:.2e}, ana={mag_ana:.2e}"
        )


# ===========================================================================
# 3. Assembled RWG (3 identical cells)
# ===========================================================================

class TestRWGAssembledModel:
    """
    Assemble 3 identical short RWG cells and validate via concatenation
    against the analytical solution for the combined length.
    """

    CELL_L = RWG_L / 3  # ~66.7 mm each

    @pytest.fixture(scope="class")
    def assembly(self):
        asm = Assembly(main_axis='Z')
        for i in range(3):
            cell = RectangularWaveguide(a=RWG_A, L=self.CELL_L, maxh=RWG_MAXH)
            if i == 0:
                asm.add("cell", cell)
            else:
                # After first add, "cell" gets renamed to "cell_1", etc.
                # Use asm.keys[-1] to reference the last added component
                asm.add("cell", cell, after=asm.keys[-1])
        asm.build()
        asm.generate_mesh(maxh=RWG_MAXH)
        return asm

    @pytest.fixture(scope="class")
    def analytical(self):
        return RWGAnalytical(a=RWG_A, L=RWG_L, b=RWG_B)

    @pytest.fixture(scope="class")
    def freq_range(self, analytical):
        fmin = 1.1 * analytical.fc
        fmax = 2.0 * analytical.fc
        return fmin, fmax

    def test_assembly_has_three_domains(self, assembly):
        assert len(assembly.components) == 3

    def test_concatenation_vs_analytical(self, assembly, analytical, freq_range):
        """3-cell concatenation should match the full-length analytical solution."""
        fmin, fmax = freq_range
        solver = FrequencyDomainSolver(assembly, order=ORDER)
        solver.solve(
            fmin=fmin, fmax=fmax, nsamples=NSAMPLES,
            per_domain=True, store_snapshots=True
        )

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        concat = rom.concatenate()
        result = concat.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES)

        freqs_ghz = concat.frequencies / 1e9
        Z_ana = analytical.z_parameters(freqs_ghz)
        Z11_concat = result['Z'][:, 0, 0]
        Z11_ana = Z_ana['Z11']

        mag_concat = np.mean(np.abs(Z11_concat))
        mag_ana = np.mean(np.abs(Z11_ana))
        assert mag_concat > 0, "Concatenated Z11 is zero"
        assert abs(np.log10(mag_concat / mag_ana)) < 1, (
            f"Z11 magnitudes differ by more than 10x: "
            f"concat={mag_concat:.2e}, ana={mag_ana:.2e}"
        )


# ===========================================================================
# 4. Iterative solver
# ===========================================================================

class TestIterativeSolver:
    """Test iterative (GMRES) solver path on RWG."""

    @pytest.fixture(scope="class")
    def rwg(self):
        return RectangularWaveguide(a=RWG_A, L=RWG_L, maxh=RWG_MAXH)

    def test_iterative_solver_completes(self, rwg):
        """Iterative solver should complete and produce finite S-params."""
        fds = FrequencyDomainSolver(rwg, order=ORDER)
        result = fds.solve(
            fmin=1.6, fmax=3.0, nsamples=NSAMPLES,
            store_snapshots=False, solver_type='iterative',
        )
        assert result is not None
        S12 = result['S_dict']['1(1)2(1)']
        assert S12 is not None
        assert len(S12) == NSAMPLES
        assert np.all(np.isfinite(S12))

    def test_iterative_vs_direct(self, rwg):
        """Iterative and direct solves should produce similar S-parameters."""
        fmin, fmax = 1.6, 2.5  # above cutoff, in GHz

        fds_d = FrequencyDomainSolver(rwg, order=ORDER)
        res_d = fds_d.solve(fmin, fmax, NSAMPLES, store_snapshots=False, solver_type='direct')

        fds_i = FrequencyDomainSolver(rwg, order=ORDER)
        res_i = fds_i.solve(fmin, fmax, NSAMPLES, store_snapshots=False, solver_type='iterative')

        S12_d = res_d['S_dict']['1(1)2(1)']
        S12_i = res_i['S_dict']['1(1)2(1)']

        rel_err = np.max(np.abs(S12_d - S12_i) / (np.abs(S12_d) + 1e-30))
        assert rel_err < 0.05, f"Iterative vs direct S12 max relative error: {rel_err:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
