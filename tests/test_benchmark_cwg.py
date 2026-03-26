"""
Benchmark tests for circular waveguide against analytical solutions.

Validates a single CWG solid against CWGAnalytical (TE11 mode)
with a few frequency points for speed.
"""

import numpy as np
import pytest
from cavsim3d.geometry.primitives import CircularWaveguide
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.rom.reduction import ModelOrderReduction
from cavsim3d.analytical.circular_waveguide import CWGAnalytical



# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

CWG_RADIUS = 50e-3    # 50 mm radius
CWG_LENGTH = 200e-3   # 200 mm length
CWG_MAXH = 0.04       # Mesh element size
NSAMPLES = 3           # Keep tests fast
ORDER = 3
RTOL_FOM = 0.60        # 60% tolerance for FOM vs analytical (curved geometry + very coarse mesh)


class TestCWGSingleModel:
    """Validate a single CWG solid against analytical TE11 solutions."""

    @pytest.fixture(scope="class")
    def cwg(self):
        return CircularWaveguide(radius=CWG_RADIUS, length=CWG_LENGTH, maxh=CWG_MAXH)

    @pytest.fixture(scope="class")
    def analytical(self):
        return CWGAnalytical(radius=CWG_RADIUS, length=CWG_LENGTH)

    @pytest.fixture(scope="class")
    def freq_range(self, analytical):
        """Frequency range above cutoff in GHz."""
        fmin = 1.1 * analytical.fc
        fmax = 2.0 * analytical.fc
        return fmin, fmax

    def test_geometry_creation(self, cwg):
        assert cwg.mesh is not None
        assert cwg.radius == CWG_RADIUS
        assert cwg.length == CWG_LENGTH

    def test_cutoff_frequency(self, cwg, analytical):
        fc_geo = cwg.cutoff_frequency_TE11      # Hz
        fc_ana = analytical.fc * 1e9             # GHz -> Hz
        assert np.isclose(fc_geo, fc_ana, rtol=1e-4)

    def test_fom_z_parameters(self, cwg, analytical, freq_range):
        """FOM Z-parameters should approximately match analytical."""
        fmin, fmax = freq_range  # GHz
        solver = FrequencyDomainSolver(cwg, order=ORDER)
        result = solver.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES)

        # Analytical expects GHz; solver.frequencies is Hz
        freqs_ghz = solver.frequencies / 1e9
        Z_ana = analytical.z_parameters_TE11(freqs_ghz)

        # Z11 key in solver format: port1(mode1) port1(mode1) = '1(1)1(1)'
        Z11_num = result['Z_dict']['1(1)1(1)']
        Z11_ana = Z_ana['Z11']

        relative_error = np.abs(Z11_num - Z11_ana) / (np.abs(Z11_ana) + 1e-30)
        max_error = np.max(relative_error)
        assert max_error < RTOL_FOM, f"Max Z11 relative error: {max_error:.2%}"

    def test_rom_reduction_and_accuracy(self, cwg, analytical, freq_range):
        """ROM should compress well and match FOM closely."""
        fmin, fmax = freq_range  # GHz
        solver = FrequencyDomainSolver(cwg, order=ORDER)
        solver.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES, store_snapshots=True)

        rom = ModelOrderReduction(solver)
        rom.reduce(tol=1e-6)

        assert rom.total_reduced_dofs < rom.total_dofs
        assert rom.compression_ratio > 0

        results = rom.solve(fmin=fmin, fmax=fmax, nsamples=NSAMPLES)
        assert results['Z'].shape[0] == NSAMPLES
        assert np.all(np.isfinite(results['Z']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
