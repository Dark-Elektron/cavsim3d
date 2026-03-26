"""
Utility and helper function tests.

Merged from: test_helpers.py, test_solver_performance.py (fixed for pr.* logging)

Validates:
  - build_F, permutation_matrix_from_list, build_global_excitation, z2s
  - ConcatenatedSystem solver threshold logic (direct vs iterative)
  - Iterative solver progress reporting
"""

import numpy as np
import pytest
import unittest.mock as mock

from cavsim3d.solvers.concatenation import ConcatenatedSystem
from cavsim3d.rom.structures import ReducedStructure

try:
    from cavsim3d.helpers import build_F, permutation_matrix_from_list, build_global_excitation, z2s
    _HAS_HELPERS = True
except (ModuleNotFoundError, ImportError):
    _HAS_HELPERS = False


# ===========================================================================
# Helper functions (skip if cavsim3d.helpers no longer exists)
# ===========================================================================

@pytest.mark.skipif(not _HAS_HELPERS, reason="cavsim3d.helpers module not available")
class TestHelpers:
    def test_build_F_shapes_and_values(self):
        F = build_F(3, dtype=int)
        assert F.shape == (6, 3)
        for j in range(3):
            assert F[2*j, j] == 1
            assert F[2*j+1, j] == -1

    def test_permutation_matrix_from_list(self):
        perm = [2, 0, 1]
        PT, P = permutation_matrix_from_list(perm)
        old = np.array([10, 20, 30])
        new = PT @ old
        assert list(new) == [30, 10, 20]
        assert np.all(P == PT.T)

    def test_build_global_excitation_padding_and_stack(self):
        a = np.array([1, 0])
        b = np.array([[0, 1], [1, 0]])
        I_global = build_global_excitation([a, b])
        assert I_global.shape == (3, 2)
        assert I_global[0, 0] == 1

    def test_z2s_zero_for_z_equals_z0_identity(self):
        Z0 = 50.0
        Z = Z0 * np.eye(2)
        S = z2s(Z, Z0)
        assert np.allclose(S, np.zeros_like(S), atol=1e-12)


# ===========================================================================
# Solver performance / threshold logic
# ===========================================================================

def _make_concat_system(size=600):
    """Create a minimal ConcatenatedSystem for threshold tests."""
    Ard = np.eye(size)
    Brd = np.ones((size, 2))

    struct = ReducedStructure(
        Ard=Ard, Brd=Brd,
        ports=['P1', 'P2'],
        port_modes={'P1': {0: None}, 'P2': {0: None}},
        domain='D1', r=size, n_full=size
    )

    cs = ConcatenatedSystem(structures=[struct], port_impedance_func=lambda p, m, f: 50.0)
    cs.A_coupled = Ard
    cs.B_coupled = Brd
    cs._n_external = 2
    cs._external_port_modes = [0, 1]
    cs._global_to_local = {0: (0, 'P1', 0), 1: (0, 'P2', 0)}
    return cs


class TestSolverThreshold:
    def test_small_system_uses_direct(self):
        """Systems below threshold should default to direct solver."""
        cs = _make_concat_system(size=600)

        with mock.patch('cavsim3d.utils.printing.debug') as mock_debug:
            cs.solve(1, 2, 5)
            # Check that the solver type message mentions 'direct'
            debug_calls = [str(c) for c in mock_debug.call_args_list]
            solver_msgs = [m for m in debug_calls if 'Solver' in m and 'direct' in m]
            assert len(solver_msgs) > 0, (
                f"Expected 'direct' solver message, got: {debug_calls}"
            )

    def test_iterative_progress_reporting(self):
        """Forcing iterative solver should report frequency progress."""
        cs = _make_concat_system(size=10)
        cs.frequencies = np.linspace(1, 2, 10) * 1e9

        with mock.patch('cavsim3d.utils.printing.debug') as mock_debug, \
             mock.patch('cavsim3d.utils.printing.running') as mock_running:
            cs.solve(1, 2, 10, solver_type='iterative')

            # Check for "Solving N frequencies" running message
            running_calls = [str(c) for c in mock_running.call_args_list]
            assert any('10' in m and 'frequencies' in m for m in running_calls), (
                f"Expected 'Solving 10 frequencies' message, got: {running_calls}"
            )

            # Check for frequency progress debug messages
            debug_calls = [str(c) for c in mock_debug.call_args_list]
            freq_msgs = [m for m in debug_calls if 'Frequency' in m]
            assert len(freq_msgs) > 0, "Expected frequency progress messages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
