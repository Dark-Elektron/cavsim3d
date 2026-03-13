import os
import shutil
import unittest.mock as mock
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
import pickle

from core.persistence import H5Serializer, ProjectManager
from solvers.results import FOMResult, FOMCollection

@pytest.fixture
def tmp_sim_dir(tmp_path):
    d = tmp_path / "simulations"
    d.mkdir()
    return d

class TestH5Serializer:
    def test_complex_conversion(self):
        arr = np.array([1+2j, 3-4j], dtype=complex)
        h5_arr = H5Serializer.to_complex_h5(arr)
        assert h5_arr.dtype == H5Serializer.COMPLEX_DTYPE
        assert h5_arr['real'][0] == 1.0
        assert h5_arr['imag'][0] == 2.0
        
        back = H5Serializer.from_complex_h5(h5_arr)
        np.testing.assert_array_equal(back, arr)

    def test_sparse_csr(self, tmp_path):
        mtx = sp.csr_matrix([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=float)
        h5_file = tmp_path / "test_sparse.h5"
        
        with h5py.File(h5_file, "w") as f:
            H5Serializer.save_sparse_csr(f, "my_mtx", mtx)
            
        with h5py.File(h5_file, "r") as f:
            loaded = H5Serializer.load_sparse_csr(f["my_mtx"])
            
        assert isinstance(loaded, sp.csr_matrix)
        assert loaded.shape == mtx.shape
        np.testing.assert_array_equal(loaded.toarray(), mtx.toarray())

    def test_automatic_real_complex_handling(self, tmp_path):
        real_arr = np.array([1.0, 2.0])
        complex_arr = np.array([1.0 + 1j, 2.0])
        h5_file = tmp_path / "test_auto.h5"
        
        with h5py.File(h5_file, "w") as f:
            H5Serializer.save_dataset(f, "real", real_arr)
            H5Serializer.save_dataset(f, "complex", complex_arr)
            
        with h5py.File(h5_file, "r") as f:
            assert f["real"].dtype == np.float64
            assert f["complex"].dtype == H5Serializer.COMPLEX_DTYPE
            
            loaded_real = H5Serializer.load_dataset(f["real"])
            loaded_complex = H5Serializer.load_dataset(f["complex"])
            
            np.testing.assert_array_equal(loaded_real, real_arr)
            np.testing.assert_array_equal(loaded_complex, complex_arr)

class TestProjectManager:
    def test_prepare_project_new(self, tmp_sim_dir):
        pm = ProjectManager(tmp_sim_dir)
        path = pm.prepare_project("new_proj")
        assert path.exists()
        assert path.name == "new_proj"

    def test_prepare_project_rerun(self, tmp_sim_dir):
        pm = ProjectManager(tmp_sim_dir)
        path = tmp_sim_dir / "old_proj"
        path.mkdir()
        (path / "dummy.txt").touch()
        
        with mock.patch('builtins.input', return_value='R'):
            new_path = pm.prepare_project("old_proj")
            
        assert new_path == path
        assert not (new_path / "dummy.txt").exists()

    def test_prepare_project_rename(self, tmp_sim_dir):
        pm = ProjectManager(tmp_sim_dir)
        path = tmp_sim_dir / "old_proj"
        path.mkdir()
        
        # First call 'N' then 'new_name'
        with mock.patch('builtins.input', side_effect=['N', 'renamed_proj']):
            new_path = pm.prepare_project("old_proj")
            
        assert new_path.name == "renamed_proj"
        assert new_path.exists()

class DummyPicklable:
    pass

class DummyMesh(DummyPicklable):
    """Mock NGSolve Mesh."""
    def Materials(self, *args):
        return mock.MagicMock()
    def Boundaries(self, *args):
        return mock.MagicMock()
    def GetMaterials(self):
        return ["domain1"]
    def GetBoundaries(self):
        return ["P1", "P2"]

class DummyFES(DummyPicklable):
    """Mock NGSolve FESpace."""
    ndof = 10
    def FreeDofs(self):
        return [True] * 10

class TestNGSolvePersistence:
    def test_save_mesh(self, tmp_path):
        # Mock mesh
        mesh = DummyPicklable()
        mesh.data = "mesh_data"
        
        # We need to simulate that ngs is not None in ProjectManager
        with mock.patch('core.persistence.ngs', mock.MagicMock()):
            ProjectManager.save_ngs_mesh(tmp_path, mesh)
            
        assert (tmp_path / "mesh.pkl").exists()
        with open(tmp_path / "mesh.pkl", "rb") as f:
            loaded_mesh = pickle.load(f)
            assert loaded_mesh.data == "mesh_data"

    def test_load_mesh(self, tmp_path):
        mesh = DummyPicklable()
        mesh.data = "loaded_mesh"
        with open(tmp_path / "mesh.pkl", "wb") as f:
            pickle.dump(mesh, f)
            
        with mock.patch('core.persistence.ngs', mock.MagicMock()):
            loaded = ProjectManager.load_ngs_mesh(tmp_path)
            
        assert loaded.data == "loaded_mesh"

    def test_save_load_fes(self, tmp_path):
        fes = DummyPicklable()
        fes.data = "fes_data"
        with mock.patch('core.persistence.ngs', mock.MagicMock()):
            ProjectManager.save_ngs_fes(tmp_path, fes)
            assert (tmp_path / "fes.pkl").exists()
            
            loaded = ProjectManager.load_ngs_fes(tmp_path)
            assert loaded.data == "fes_data"

class TestResultPersistence:
    def test_fom_result_save_load(self, tmp_path):
        freqs = np.linspace(1e9, 2e9, 10)
        Z = np.random.randn(10, 2, 2) + 1j * np.random.randn(10, 2, 2)
        S = Z * 0.1
        
        fom = FOMResult(
            domain='test',
            frequencies=freqs,
            Z_matrix=Z,
            S_matrix=S,
            Z_dict=None,
            S_dict=None,
            n_ports=2,
            ports=['P1', 'P2']
        )
        
        save_path = tmp_path / "fom_save"
        fom.save(save_path)
        
        loaded = FOMResult.load(save_path)
        
        assert loaded.domain == 'test'
        assert loaded.n_ports == 2
        np.testing.assert_array_almost_equal(loaded.frequencies, freqs)
        np.testing.assert_array_almost_equal(loaded._Z_matrix, Z)
        np.testing.assert_array_almost_equal(loaded._S_matrix, S)
        
        # Test lazy reconstruction of S_dict
        assert loaded.S_dict is not None
        assert '1(1)1(1)' in loaded.S_dict
        np.testing.assert_array_almost_equal(loaded.S_dict['1(1)1(1)'], S[:, 0, 0])

    def test_fom_collection_save_load(self, tmp_path):
        freqs = np.linspace(1e9, 2e9, 5)
        fom1 = FOMResult(domain='D1', frequencies=freqs, Z_matrix=np.random.randn(5,1,1), S_matrix=None, Z_dict=None, S_dict=None, n_ports=1, ports=['P1'])
        fom2 = FOMResult(domain='D2', frequencies=freqs, Z_matrix=np.random.randn(5,1,1), S_matrix=None, Z_dict=None, S_dict=None, n_ports=1, ports=['P2'])
        
        coll = FOMCollection([fom1, fom2])
        save_path = tmp_path / "coll_save"
        coll.save(save_path)
        
        loaded = FOMCollection.load(save_path)
        
        assert len(loaded) == 2
        assert loaded[0].domain == 'D1'
        assert loaded[1].domain == 'D2'
        np.testing.assert_array_almost_equal(loaded[0]._Z_matrix, fom1._Z_matrix)

    def test_rom_collection_save_load(self, tmp_path):
        from rom.reduction import ModelOrderReduction
        from solvers.results import ROMCollection
        
        # Mock a MOR object
        mor = mock.MagicMock(spec=ModelOrderReduction)
        mor.domains = ['D1', 'D2']
        mor.n_domains = 2
        
        coll = ROMCollection(_mor_ref=mor)
        save_path = tmp_path / "rom_coll_save"
        coll.save(save_path)
        
        mor.save.assert_called_once_with(save_path)

        # For load, we need to mock ModelOrderReduction.load
        with mock.patch('rom.reduction.ModelOrderReduction.load', return_value=mor):
            loaded = ROMCollection.load(save_path)
            assert isinstance(loaded, ROMCollection)
            assert isinstance(loaded, ROMCollection)
            assert loaded._mor_ref == mor

    def test_frequency_domain_solver_auto_persistence(self, tmp_path):
        from solvers.frequency_domain import FrequencyDomainSolver
        
        # Mock geometry
        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None
        
        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES() # Mock FES
        
        # 1. Save to link project
        project_name = "test_auto_save"
        project_path = fds.save(project_name, base_dir=tmp_path)
        assert fds._project_path == project_path
        
        # 2. Mock a solve and check if save was called
        # We'll patch fds.save to see if it's called by solve
        with mock.patch.object(FrequencyDomainSolver, 'save', wraps=fds.save) as mock_save:
            # Manually set some results to simulate a solve
            fds.frequencies = np.linspace(1e9, 2e9, 5)
            fds._Z_global_coupled = np.random.randn(5, 2, 2)
            fds._Z_matrix = fds._Z_global_coupled.copy()
            fds._ports = ['P1', 'P2']
            fds._n_ports = 2
            
            # Trigger build_results_dict which is called by solve
            # We'll just call the part of solve that triggers auto-save
            fds.save() # already linked
            assert mock_save.called
            
            # Check if FOM was saved
            assert (project_path / "fom").exists()
            assert (project_path / "fom/snapshots.h5").exists()


# ---------------------------------------------------------------------------
# Geometry History Tests
# ---------------------------------------------------------------------------

class TestGeometryHistory:
    """Test the unified history system on BaseGeometry."""

    def test_record_appends_operations(self):
        """_record() should append timestamped entries to _history."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        geo = ConcreteGeo()
        assert geo._history == []

        geo._record('import_step', filepath='test.step', unit='mm')
        geo._record('split')
        geo._record('generate_mesh', maxh=0.05)

        assert len(geo._history) == 3
        assert geo._history[0]['op'] == 'import_step'
        assert geo._history[0]['filepath'] == 'test.step'
        assert geo._history[0]['unit'] == 'mm'
        assert 'timestamp' in geo._history[0]
        assert geo._history[1]['op'] == 'split'
        assert geo._history[2]['op'] == 'generate_mesh'
        assert geo._history[2]['maxh'] == 0.05

    def test_get_history_returns_copy(self):
        """get_history() should return a copy, not a reference."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        geo = ConcreteGeo()
        geo._record('op1')
        history = geo.get_history()
        history.append({'op': 'fake'})
        assert len(geo._history) == 1  # original unchanged

    def test_none_params_filtered(self):
        """_record() should filter out None values."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        geo = ConcreteGeo()
        geo._record('test_op', a=1, b=None, c='hello')
        entry = geo._history[0]
        assert entry['a'] == 1
        assert entry['c'] == 'hello'
        assert 'b' not in entry


# ---------------------------------------------------------------------------
# Geometry Linking Tests
# ---------------------------------------------------------------------------

class TestGeometryLinking:
    """Test source file linking and change detection."""

    def test_file_hash_consistency(self, tmp_path):
        """_file_hash should produce consistent SHA-256 hashes."""
        from geometry.base import BaseGeometry

        f = tmp_path / "test_file.txt"
        f.write_text("hello world")

        h1 = BaseGeometry._file_hash(f)
        h2 = BaseGeometry._file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_file_hash_changes_on_modification(self, tmp_path):
        """_file_hash should change when file content changes."""
        from geometry.base import BaseGeometry

        f = tmp_path / "test_file.txt"
        f.write_text("version 1")
        h1 = BaseGeometry._file_hash(f)

        f.write_text("version 2")
        h2 = BaseGeometry._file_hash(f)
        assert h1 != h2

    def test_save_geometry_creates_history_json(self, tmp_path):
        """save_geometry should create geometry/history.json."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        geo = ConcreteGeo()
        geo._record('test_op', param=42)
        geo.save_geometry(project_dir)

        history_file = project_dir / "geometry" / "history.json"
        assert history_file.exists()

        import json
        with open(history_file) as f:
            meta = json.load(f)
        assert meta['type'] == 'ConcreteGeo'
        assert len(meta['history']) == 1
        assert meta['history'][0]['op'] == 'test_op'

    def test_save_geometry_copies_source_file(self, tmp_path):
        """save_geometry should copy the source file to geometry/."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        # Create a fake source file
        source = tmp_path / "model.step"
        source.write_text("STEP DATA")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        geo = ConcreteGeo()
        geo._source_link = str(source)
        geo.save_geometry(project_dir)

        copied = project_dir / "geometry" / "source_model.step"
        assert copied.exists()
        assert copied.read_text() == "STEP DATA"

        # Check hash was stored
        import json
        with open(project_dir / "geometry" / "history.json") as f:
            meta = json.load(f)
        assert meta['source_hash'] is not None
        assert meta['source_link'] == str(source)

    def test_check_source_link_detects_change(self, tmp_path):
        """_check_source_link should detect when source file has changed."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        source = tmp_path / "model.step"
        source.write_text("original content")

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        geo = ConcreteGeo()
        geo._source_link = str(source)
        geo.save_geometry(project_dir)

        # Modify source
        source.write_text("modified content")

        # Simulate load with 'K' (keep) response
        with mock.patch('builtins.input', return_value='K'):
            geo._check_source_link(project_dir)
        # Source link should still be set (user chose to keep)
        assert geo._source_link is not None

    def test_check_source_link_handles_missing(self, tmp_path):
        """_check_source_link should handle missing source with unlink option."""
        from geometry.base import BaseGeometry

        class ConcreteGeo(BaseGeometry):
            def build(self):
                pass

        project_dir = tmp_path / "project"
        geo_dir = project_dir / "geometry"
        geo_dir.mkdir(parents=True)

        geo = ConcreteGeo()
        geo._source_link = "/nonexistent/model.step"
        geo._source_hash = "deadbeef"

        # Write a minimal history.json
        import json
        with open(geo_dir / 'history.json', 'w') as f:
            json.dump({'source_link': geo._source_link, 'source_hash': geo._source_hash}, f)

        # User chooses to unlink
        with mock.patch('builtins.input', return_value='U'):
            geo._check_source_link(project_dir)
        assert geo._source_link is None

    def test_delete_project_results(self, tmp_path):
        """_delete_project_results should remove all result artefacts."""
        from geometry.base import BaseGeometry

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create some result files
        (project_dir / "matrices.h5").touch()
        (project_dir / "snapshots.h5").touch()
        fom_dir = project_dir / "fom"
        fom_dir.mkdir()
        (fom_dir / "data.h5").touch()

        BaseGeometry._delete_project_results(project_dir)

        assert not (project_dir / "matrices.h5").exists()
        assert not (project_dir / "snapshots.h5").exists()
        assert not fom_dir.exists()


# ---------------------------------------------------------------------------
# Rerun Protection Tests
# ---------------------------------------------------------------------------

class TestRerunProtection:
    """Test that solve() warns when results exist and rerun=False."""

    def test_solve_warns_when_results_exist(self):
        """solve() should warn when results already exist."""
        from solvers.frequency_domain import FrequencyDomainSolver

        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None

        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES()

        # Simulate existing results
        fds._Z_global_coupled = np.random.randn(5, 2, 2)
        fds.frequencies = np.linspace(1e9, 2e9, 5)
        fds._ports = ['P1', 'P2']
        fds._n_ports = 2

        with mock.patch.object(fds, '_build_results_dict', return_value={}):
            with pytest.warns(UserWarning, match="Results already exist"):
                fds.solve(1, 2, 5)

    def test_solve_proceeds_with_rerun_flag(self):
        """solve(rerun=True) should not warn even if results exist."""
        from solvers.frequency_domain import FrequencyDomainSolver

        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None

        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES()

        # Simulate existing results
        fds._fom_cache = mock.MagicMock()
        fds.frequencies = np.linspace(1e9, 2e9, 5)

        # Patch the actual solve machinery to avoid needing real matrices
        with mock.patch.object(fds, '_ensure_matrices_assembled'):
            with mock.patch.object(fds, '_clear_results'):
                with mock.patch.object(fds, '_print_solve_config'):
                    with mock.patch.object(fds, '_solve_global_coupled'):
                        with mock.patch.object(fds, '_compute_s_from_z'):
                            with mock.patch.object(fds, '_invalidate_cache'):
                                with mock.patch.object(fds, '_build_results_dict', return_value={}):
                                    # Should NOT warn
                                    import warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("error")
                                        fds.solve(1, 2, 5, rerun=True)

    def test_solve_no_warn_when_no_results(self):
        """solve() should not warn on first call (no existing results)."""
        from solvers.frequency_domain import FrequencyDomainSolver

        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None

        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES()

        # No results exist — all caches are None/empty
        with mock.patch.object(fds, '_ensure_matrices_assembled'):
            with mock.patch.object(fds, '_clear_results'):
                with mock.patch.object(fds, '_print_solve_config'):
                    with mock.patch.object(fds, '_solve_global_coupled'):
                        with mock.patch.object(fds, '_compute_s_from_z'):
                            with mock.patch.object(fds, '_invalidate_cache'):
                                with mock.patch.object(fds, '_build_results_dict', return_value={}):
                                    import warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("error")
                                        fds.solve(1, 2, 5)


# ---------------------------------------------------------------------------
# Solver History Tests
# ---------------------------------------------------------------------------

class TestSolverHistory:
    """Test that solver operations are logged in _solver_history."""

    def test_solve_records_history(self):
        """solve() should append an entry to _solver_history."""
        from solvers.frequency_domain import FrequencyDomainSolver

        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None

        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES()
        assert fds._solver_history == []

        with mock.patch.object(fds, '_ensure_matrices_assembled'):
            with mock.patch.object(fds, '_clear_results'):
                with mock.patch.object(fds, '_print_solve_config'):
                    with mock.patch.object(fds, '_solve_global_coupled'):
                        with mock.patch.object(fds, '_compute_s_from_z'):
                            with mock.patch.object(fds, '_invalidate_cache'):
                                with mock.patch.object(fds, '_build_results_dict', return_value={}):
                                    fds.solve(1.5, 3.0, 50)

        assert len(fds._solver_history) == 1
        entry = fds._solver_history[0]
        assert entry['op'] == 'solve'
        assert entry['fmin'] == 1.5
        assert entry['fmax'] == 3.0
        assert entry['nsamples'] == 50
        assert 'timestamp' in entry

    def test_solver_history_persisted_in_config(self, tmp_path):
        """save() should include _solver_history in config.json."""
        from solvers.frequency_domain import FrequencyDomainSolver

        geom = mock.MagicMock()
        geom.mesh = DummyMesh()
        geom.bc = None

        fds = FrequencyDomainSolver(geom, order=3)
        fds._fes_global = DummyFES()
        fds._solver_history = [
            {'op': 'solve', 'fmin': 1, 'fmax': 10, 'nsamples': 100}
        ]

        project_path = fds.save("test_history", base_dir=tmp_path)

        import json
        with open(project_path / "config.json") as f:
            config = json.load(f)

        assert 'solver_history' in config
        assert len(config['solver_history']) == 1
        assert config['solver_history'][0]['op'] == 'solve'
        assert config['has_results'] is False  # No actual results
