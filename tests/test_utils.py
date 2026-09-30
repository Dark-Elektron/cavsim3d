"""
Utility tests.

Validates:
  - Z -> S conversion (ParameterConverter.z_to_s)
  - ConcatenatedSystem solver threshold logic (direct vs iterative)
  - Iterative solver progress reporting
"""

import numpy as np
import pytest
import unittest.mock as mock

from cavsim3d.solvers.concatenation import ConcatenatedSystem
from cavsim3d.solvers.base import ParameterConverter
from cavsim3d.rom.structures import ReducedStructure


def test_z_to_s_is_zero_when_z_equals_the_reference():
    Z0 = 50.0
    S = ParameterConverter.z_to_s(Z0 * np.eye(2), Z0)
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
