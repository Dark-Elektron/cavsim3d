import os
import shutil
import numpy as np
from pathlib import Path
import unittest.mock as mock

# Adjust imports to local project structure
import sys
sys.path.append(os.getcwd())

from solvers.frequency_domain import FrequencyDomainSolver
from solvers.results import FOMResult, FOMCollection

class DummyMesh:
    def Materials(self, *args): return mock.MagicMock()
    def Boundaries(self, *args): return mock.MagicMock()
    def GetMaterials(self): return ["domain1"]
    def GetBoundaries(self): return ["P1", "P2"]

class DummyFES:
    ndof = 10
    def FreeDofs(self): return [True] * 10

def test_residuals_persistence():
    tmp_path = Path("tmp_test_persistence")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir()

    try:
        # Mock geometry
        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None

        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES()

        # Simulate results and residuals
        fds.frequencies = np.linspace(1e9, 2e9, 5)
        fds._Z_global_coupled = np.random.randn(5, 2, 2)
        fds._Z_matrix = fds._Z_global_coupled.copy()
        fds._ports = ['P1', 'P2']
        fds._n_ports = 2
        
        # Manually set residuals
        fds._residuals = {
            'global': {
                'frequencies': fds.frequencies.copy(),
                'iterations': np.array([10, 12, 11, 13, 12]),
                'residuals': np.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-7]),
                'solver_type': 'iterative'
            }
        }

        # Save project
        project_name = "test_resid"
        project_path = fds.save(project_name, base_dir=tmp_path)

        # Load project
        fds_loaded = FrequencyDomainSolver.load_from_path(project_path / 'fds', geometry=geom)
        
        print(f"Loaded residuals keys: {list(fds_loaded._residuals.keys())}")
        
        assert 'global' in fds_loaded._residuals, "Residuals 'global' key missing after load"
        np.testing.assert_array_almost_equal(
            fds_loaded._residuals['global']['iterations'], 
            fds._residuals['global']['iterations']
        )
        print("Success: Residuals persistence verified for single solver.")

        # Test Compound / FOMCollection
        fds.is_compound = True
        fds.domains = ['D1', 'D2']
        fds.domain_port_map = {'D1': ['P1'], 'D2': ['P2']}
        
        fom1 = FOMResult(domain='D1', frequencies=fds.frequencies, Z_matrix=np.random.randn(5,1,1), S_matrix=None, Z_dict=None, S_dict=None, n_ports=1, ports=['P1'], residual_data={'res': 1}, _solver_ref=fds)
        fom2 = FOMResult(domain='D2', frequencies=fds.frequencies, Z_matrix=np.random.randn(5,1,1), S_matrix=None, Z_dict=None, S_dict=None, n_ports=1, ports=['P2'], residual_data={'res': 2}, _solver_ref=fds)
        
        fds._foms_cache = FOMCollection([fom1, fom2], _fds_ref=fds)
        fds._Z_per_domain = {'D1': fom1.Z_dict, 'D2': fom2.Z_dict}
        fds._residuals = {
            'D1': {'res': 1},
            'D2': {'res': 2}
        }
        
        fds.save("test_resid_compound", base_dir=tmp_path)
        
        fds_loaded_c = FrequencyDomainSolver.load_from_path(tmp_path / "test_resid_compound" / 'fds', geometry=geom)
        print(f"Loaded compound residuals keys: {list(fds_loaded_c._residuals.keys())}")
        
        assert 'D1' in fds_loaded_c._residuals
        assert 'D2' in fds_loaded_c._residuals
        assert fds_loaded_c._residuals['D1']['res'] == 1
        assert fds_loaded_c._residuals['D2']['res'] == 2
        print("Success: Residuals persistence verified for compound solver.")

        # Test solve() without rerun
        with mock.patch('warnings.warn') as mock_warn:
            # Should not crash and should return residuals
            results = fds_loaded_c.solve(1, 2, 5)
            assert 'residuals' in results
            assert mock_warn.called
            print("Success: solve() returns residuals without crash on loaded result.")

    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

if __name__ == "__main__":
    test_residuals_persistence()
