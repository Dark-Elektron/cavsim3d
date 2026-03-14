import numpy as np
import pytest
import unittest.mock as mock
from pathlib import Path
import sys
import os

# Ensure the package is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers.concatenation import ConcatenatedSystem
from rom.structures import ReducedStructure

def test_solver_threshold_logic():
    """Verify that systems below 10,000 DOFs use the direct solver by default."""
    # Create mock structures
    # A system with ~600 DOFs total
    size = 600
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
    
    # Mock print to capture solver choice
    with mock.patch('builtins.print') as mock_print:
        cs.solve(1, 2, 5)
        
        # Find the solver message
        all_msgs = [str(call.args[0]) for call in mock_print.call_args_list]
        solver_msgs = [m for m in all_msgs if "Solver:" in m]
        assert len(solver_msgs) > 0
        assert "direct" in solver_msgs[0]
        assert "600" in solver_msgs[0]

def test_iterative_progress_reporting():
    """Verify that forcing iterative solver shows progress messages."""
    size = 10
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
    cs.frequencies = np.linspace(1, 2, 10) * 1e9
    
    # Force iterative solver and capture output
    with mock.patch('builtins.print') as mock_print:
        cs.solve(1, 2, 10, solver_type='iterative')
        
        # Check for frequency reporting messages
        all_msgs = [str(call.args[0]) for call in mock_print.call_args_list]
        freq_msgs = [m for m in all_msgs if "Frequency" in m]
        
        assert len(freq_msgs) > 0
        assert "Solving 10 frequencies" in all_msgs[0] or any("Solving 10 frequencies" in m for m in all_msgs)
        # Check start and end frequencies at least
        assert any("Frequency 1/10" in m for m in freq_msgs)
        assert any("Frequency 10/10" in m for m in freq_msgs)

if __name__ == "__main__":
    pytest.main([__file__])
