"""
Persistence module for cavsim3d.
Handles HDF5 serialization and project management.
"""

from __future__ import annotations
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type, TypeVar

import h5py
import numpy as np
import scipy.sparse as sp
import pickle
from datetime import datetime

# Import NGSolve only if available (needed for mesh/fes persistence)
try:
    import ngsolve as ngs
except ImportError:
    ngs = None

T = TypeVar('T')

class H5Serializer:
    """Helper for HDF5 serialization of numpy and scipy objects."""

    COMPLEX_DTYPE = np.dtype([('real', '<f8'), ('imag', '<f8')])

    @staticmethod
    def to_complex_h5(arr: np.ndarray) -> np.ndarray:
        """Convert complex numpy array to HDF5-compatible compound type."""
        if not np.iscomplexobj(arr):
            return arr
        
        h5_arr = np.empty(arr.shape, dtype=H5Serializer.COMPLEX_DTYPE)
        h5_arr['real'] = arr.real
        h5_arr['imag'] = arr.imag
        return h5_arr

    @staticmethod
    def from_complex_h5(arr: np.ndarray) -> np.ndarray:
        """Convert HDF5 compound type back to complex numpy array."""
        if arr.dtype != H5Serializer.COMPLEX_DTYPE:
            return arr
        return arr['real'] + 1j * arr['imag']

    @classmethod
    def save_dataset(cls, group: h5py.Group, name: str, data: Any):
        """Save any supported data type to HDF5 group."""
        if data is None:
            return

        if isinstance(data, (np.ndarray, list)):
            data_np = np.asarray(data)
            if np.iscomplexobj(data_np):
                group.create_dataset(name, data=cls.to_complex_h5(data_np))
            else:
                group.create_dataset(name, data=data_np)
        
        elif sp.issparse(data):
            cls.save_sparse_csr(group, name, data)
        
        elif isinstance(data, dict):
            subgroup = group.create_group(name)
            for k, v in data.items():
                cls.save_dataset(subgroup, k, v)
        
        else:
            # Fallback for scalars
            group.attrs[name] = data

    @classmethod
    def load_dataset(cls, item: Union[h5py.Group, h5py.Dataset]) -> Any:
        """Load data from HDF5 item."""
        if isinstance(item, h5py.Dataset):
            data = item[()]
            if item.dtype == cls.COMPLEX_DTYPE:
                return cls.from_complex_h5(data)
            return data
        
        elif isinstance(item, h5py.Group):
            # Check if it's a sparse matrix
            if 'data' in item and 'indices' in item and 'indptr' in item:
                return cls.load_sparse_csr(item)
            
            # Otherwise it's a nested dict
            res = {}
            for k, v in item.items():
                res[k] = cls.load_dataset(v)
            # Mix in attributes
            for k, v in item.attrs.items():
                res[k] = v
            return res
        
        return None

    @classmethod
    def save_sparse_csr(cls, group: h5py.Group, name: str, mtx: sp.csr_matrix):
        """Save scipy CSR matrix to HDF5."""
        if not sp.isspmatrix_csr(mtx):
            mtx = mtx.tocsr()
        
        subgroup = group.create_group(name)
        cls.save_dataset(subgroup, 'data', mtx.data)
        subgroup.create_dataset('indices', data=mtx.indices)
        subgroup.create_dataset('indptr', data=mtx.indptr)
        subgroup.create_dataset('shape', data=np.array(mtx.shape))
        subgroup.attrs['type'] = 'sparse_csr'

    @classmethod
    def load_sparse_csr(cls, group: h5py.Group) -> sp.csr_matrix:
        """Load scipy CSR matrix from HDF5 group."""
        data = cls.load_dataset(group['data'])
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        shape = group['shape'][()]
        return sp.csr_matrix((data, indices, indptr), shape=tuple(shape))


class ProjectManager:
    """
    Manages simulation directory lifecycle and metadata.
    """

    def __init__(self, base_dir: Union[str, Path] = "simulations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def prepare_project(self, project_name: str) -> Path:
        """
        Prepare project directory. Handles existing projects.
        """
        project_path = self.base_dir / project_name
        
        if project_path.exists():
            print(f"Project '{project_name}' already exists at {project_path}")
            print("Options:")
            print(" [L] Load existing data")
            print(" [R] Rerun (OVERWRITE existing data)")
            print(" [N] New name (Rename current project)")
            
            choice = input("Select option [L/R/N]: ").strip().upper()
            
            if choice == 'R':
                print(f"Overwriting project '{project_name}'...")
                shutil.rmtree(project_path)
                project_path.mkdir()
            elif choice == 'N':
                new_name = input("Enter new project name: ").strip()
                return self.prepare_project(new_name)
            elif choice == 'L':
                print(f"Loading from '{project_name}'...")
                return project_path
            else:
                print("Invalid choice, defaulting to Load.")
                return project_path

        project_path.mkdir(parents=True, exist_ok=True)
        return project_path

    @staticmethod
    def save_config(path: Path, config: Dict[str, Any]):
        """Save project configuration to JSON."""
        with open(path / "config.json", "w") as f:
            # Handle non-serializable objects (Convert to str)
            json.dump(config, f, indent=2, default=str)

    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]:
        """Load project configuration from JSON."""
        config_path = path / "config.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_ngs_mesh(path: Path, mesh: 'ngsolve.Mesh'):
        """Save NGSolve mesh to file using pickle."""
        if ngs is None or mesh is None:
            return
        mesh_path = path / "mesh.pkl"
        with open(mesh_path, "wb") as f:
            pickle.dump(mesh, f)

    @staticmethod
    def load_ngs_mesh(path: Path) -> Optional['ngsolve.Mesh']:
        """Load NGSolve mesh from file using pickle."""
        if ngs is None:
            return None
        
        # Try .pkl first
        mesh_path = path / "mesh.pkl"
        if mesh_path.exists():
            with open(mesh_path, "rb") as f:
                return pickle.load(f)
        
        # Fallback to .vol
        mesh_path = path / "mesh.vol"
        if mesh_path.exists():
            return ngs.Mesh(str(mesh_path))
            
        return None

    @staticmethod
    def save_ngs_fes(path: Path, fes: 'ngsolve.FESpace'):
        """Save NGSolve FESpace to file using pickle."""
        if ngs is None or fes is None:
            return
        fes_path = path / "fes.pkl"
        with open(fes_path, "wb") as f:
            pickle.dump(fes, f)

    @staticmethod
    def load_ngs_fes(path: Path) -> Optional['ngsolve.FESpace']:
        """Load NGSolve FESpace from file using pickle."""
        if ngs is None:
            return None
        fes_path = path / "fes.pkl"
        if not fes_path.exists():
            return None
        with open(fes_path, "rb") as f:
            return pickle.load(f)
