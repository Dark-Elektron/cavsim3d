import pytest
import shutil
from pathlib import Path
import numpy as np
import os
from core.em_project import EMProject
from geometry.primitives import RectangularWaveguide
from solvers.frequency_domain import FrequencyDomainSolver
from unittest.mock import MagicMock

def test_em_project_structure(tmp_path):
    """Test that EMProject creates the correct folder structure."""
    project_name = "test_project"
    project = EMProject(project_name, base_dir=tmp_path)
    
    assert project.project_path.exists()
    project.save()
    assert (project.project_path / "geometry").exists()
    assert (project.project_path / "mesh").exists()
    assert (project.project_path / "fds").exists()
    assert (project.project_path / "eigenmode").exists()

def test_em_project_save_load(tmp_path):
    """Test saving and loading a minimal project."""
    project_name = "save_load_project"
    
    # Create and setup project
    geom = RectangularWaveguide(a=1.0, b=0.5, L=1.0)
    project = EMProject(project_name, base_dir=tmp_path, geometry=geom)
    
    fds = FrequencyDomainSolver(geom, order=1)
    project.fds = fds
    
    # Save
    project.save()
    assert (project.project_path / "project.json").exists()
    assert (project.geometry_path / "history.json").exists()
    
    # Load
    loaded_project = EMProject.load(project_name, base_dir=tmp_path)
    assert loaded_project.name == project_name
    assert loaded_project.geometry is not None
    assert loaded_project.fds is not None
    assert loaded_project.fds._project_path == project.project_path

def test_hierarchical_eigenmode_saving(tmp_path):
    """Test that eigenmodes are saved to the correct mirrored hierarchy."""
    project_name = "eigen_hierarchy_project"
    geom = RectangularWaveguide(a=1.0, b=0.5, L=1.0)
    project = EMProject(project_name, base_dir=tmp_path, geometry=geom)
    
    # Mock solver to avoid real computation
    fds = MagicMock(spec=FrequencyDomainSolver)
    fds.mesh = geom.mesh
    fds._project_path = project.project_path
    fds.all_ports = ['port1']
    fds.ports = ['port1']
    fds.external_ports = ['port1']
    fds.domain_port_map = {'global': ['port1']}
    fds.domains = ['global']
    fds.n_domains = 1
    fds.port_modes = {'port1': [None]}
    fds.port_solver = MagicMock()
    fds.port_solver.get_port_wave_impedance = MagicMock(return_value=lambda f: 377.0)
    fds.snapshots = {}
    fds.get_rom_data = MagicMock(return_value={'A': None, 'K': None, 'M': None, 'B': None, 'C': None, 'W': None})
    fds._n_modes_per_port = 1
    
    project.fds = fds
    
    # 1. Test FOM level (global)
    from solvers.results import FOMResult
    fom = FOMResult(
        domain='global', 
        frequencies=np.array([1.0e9]), 
        Z_matrix=np.zeros((1, 1, 1)), 
        S_matrix=np.zeros((1, 1, 1)), 
        Z_dict={}, 
        S_dict={}, 
        n_ports=1, 
        ports=['port1'], 
        _solver_ref=fds
    )
    
    # Mock save_eigenmodes to write a dummy file
    def mock_save_eigenmodes(path, **kwargs):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "eigenmodes.h5").touch()
    
    fds.save_eigenmodes.side_effect = mock_save_eigenmodes
    fds.get_eigenmodes.return_value = (np.array([1.0, 2.0]), np.zeros((10, 2)))
    
    fom.get_eigenmodes(n_modes=2)
    expected_path = project.project_path / "eigenmode" / "fom" / "eigenmodes.h5"
    assert expected_path.exists()

    # 2. Test ROM level
    from rom.reduction import ModelOrderReduction
    mor = ModelOrderReduction(fds)
    mor._is_reduced = True
    mor.n_domains = 1
    mor.domains = ['global']
    mor._A_r = {'global': np.eye(5)}
    mor._B_r = {'global': np.zeros((5, 1))}
    mor._W = {'global': np.zeros((10, 5))}
    mor._Q_L_inv = {'global': np.eye(5)}
    mor._r = {'global': 5}
    mor.solver = fds
    
    mor.get_eigenvectors = MagicMock(return_value=(np.array([1.0, 2.0]), np.zeros((5, 2))))
    mor.get_eigenmodes(n_modes=2)
    
    expected_rom_path = project.project_path / "eigenmode" / "fom" / "rom" / "eigenmodes.h5"
    assert expected_rom_path.exists()

def test_simplified_api_workflow(tmp_path):
    """Test the new simplified project-centric workflow."""
    project_name = "simplified_project"
    project = EMProject(project_name, base_dir=tmp_path, order=2)
    
    # 1. Create primitive via project
    project.create_primitive('rectangular_waveguide', a=1.0, b=0.5, L=1.0)
    assert project.geometry is not None
    assert project.geo == project.geometry
    
    # 2. Lazy FDS initialization
    fds = project.fds
    assert fds is not None
    assert fds.order == 2
    assert fds._project_path == project.project_path
    
    # 3. Automatic assembly in solve
    # Mock verify setup
    fds._ensure_matrices_assembled = MagicMock()
    fds._solve_global_coupled = MagicMock()
    fds._build_results_dict = MagicMock(return_value={})
    
    project.fds.solve(fmin=1.0, fmax=2.0, nsamples=10, nportmodes=2)
    
    # Verify nportmodes was passed
    fds._ensure_matrices_assembled.assert_called_once()
    _, kwargs = fds._ensure_matrices_assembled.call_args
    assert kwargs['nportmodes'] == 2

if __name__ == "__main__":
    pytest.main([__file__])
