from __future__ import annotations
import os
from core.persistence import ProjectManager
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import json
from datetime import datetime
import shutil

if TYPE_CHECKING:
    from geometry.base import BaseGeometry
    from solvers.frequency_domain import FrequencyDomainSolver
    from ngsolve import Mesh

class EMProject:
    """
    Central class for managing electromagnetic simulation projects.
    
    Responsibility:
    - Manage the project directory structure.
    - Orchestrate saving and loading of geometry, mesh, and solvers.
    - Provide a unified entry point for simulation.
    """
    
    def __init__(
        self,
        name: str,
        base_dir: Optional[Union[str, Path]] = None,
        geometry: Optional[BaseGeometry] = None,
        bc: str = 'wall',
    ):
        self.name = name
        # Use current directory if base_dir is not provided
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.project_path = self.base_dir / self.name
        
        self.geometry = geometry
        self.bc = bc
        self.mesh: Optional[Mesh] = None
        self._fds: Optional[FrequencyDomainSolver] = None
        self._order = 3
        self._n_port_modes = 1
        
        # Automatic Loading or Creation
        if self.project_path.exists():
            print(f"Project '{self.name}' exists. Loading automatically...")
            self._initial_load()
        else:
            print(f"Creating new project '{self.name}' at {self.project_path}")
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.geometry_path.mkdir(parents=True, exist_ok=True)
            self.mesh_path.mkdir(parents=True, exist_ok=True)
            self.fds_path.mkdir(parents=True, exist_ok=True)
            # Automatic save on creation if geometry is provided
            if self.geometry:
                self.save()

    def _initial_load(self):
        """Internal helper for automatic loading during instantiation."""
        metadata_file = self.project_path / "project.json"
        if not metadata_file.exists():
            return
            
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            
        self.bc = metadata.get("bc", self.bc)
        self._order = metadata.get("order", self._order)
        self._n_port_modes = metadata.get("n_port_modes", self._n_port_modes)
        
        # 1. Load Geometry
        if metadata.get("has_geometry"):
            from geometry.base import BaseGeometry
            try:
                self.geometry = BaseGeometry.load_geometry(self.project_path)
            except Exception as e:
                print(f"Warning: Could not load geometry: {e}")
            
        # 2. Load Mesh
        if metadata.get("has_mesh"):
            from core.persistence import ProjectManager
            pm = ProjectManager(self.base_dir)
            self.mesh = pm.load_ngs_mesh(self.mesh_path)
            
        # 3. Load Solver (FDS)
        if metadata.get("has_fds"):
            from solvers.frequency_domain import FrequencyDomainSolver
            self.fds = FrequencyDomainSolver.load_from_path(self.fds_path, geometry=self.geometry)
            if self.fds:
                self.fds._project_path = self.project_path
                if self.mesh:
                    self.fds.mesh = self.mesh
                    # Restore FES
                    pm = ProjectManager(self.base_dir)
                    self.fds._fes_global = pm.load_ngs_fes(self.mesh_path)

    @property
    def geo(self) -> Optional[BaseGeometry]:
        """Shortcut for geometry."""
        return self.geometry

    @property
    def fds(self) -> FrequencyDomainSolver:
        """Lazy initialization of the FrequencyDomainSolver."""
        if self._fds is None:
            if self.geometry is None:
                raise RuntimeError("Cannot initialize solver without geometry.")
            from solvers.frequency_domain import FrequencyDomainSolver
            self._fds = FrequencyDomainSolver(
                geometry=self.geometry,
                order=self.order,
                bc=self.bc
            )
            self._fds._project_path = self.project_path
            self._fds._project_name = self.name
            self._fds._project_ref = self
        return self._fds

    @fds.setter
    def fds(self, value: Optional[FrequencyDomainSolver]):
        self._fds = value
        if value:
            value._project_path = self.project_path
            value._project_name = self.name
            value._project_ref = self
            value.order = self._order
            value.n_port_modes = self._n_port_modes

    def import_geometry(self, filepath: Union[str, Path], **kwargs) -> BaseGeometry:
        """Import geometry from a STEP file and associate it with the project."""
        from geometry.importers import STEPImporter
        self.geometry = STEPImporter(str(filepath), **kwargs)
        return self.geometry

    def create_primitive(self, primitive_type: str, **kwargs) -> BaseGeometry:
        """Create a primitive geometry and associate it with the project."""
        import geometry.primitives as primitives
        
        # Map string names to classes
        mapping = {
            'rectangular_waveguide': primitives.RectangularWaveguide,
            'circular_waveguide': primitives.CircularWaveguide,
            'rwg': primitives.RectangularWaveguide,
            'cwg': primitives.CircularWaveguide,
        }
        
        cls = mapping.get(primitive_type.lower())
        if not cls:
            raise ValueError(f"Unknown primitive type: {primitive_type}")
            
        self.geometry = cls(**kwargs)
        return self.geometry

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value: int):
        self._order = value
        if self._fds:
            self._fds.order = value

    @property
    def n_port_modes(self) -> int:
        return self._n_port_modes

    @n_port_modes.setter
    def n_port_modes(self, value: int):
        self._n_port_modes = value
        if self._fds:
            self._fds.n_port_modes = value

    @property
    def mesh_path(self) -> Path:
        return self.project_path / "mesh"

    @property
    def fds_path(self) -> Path:
        return self.project_path / "fds"

    @property
    def geometry_path(self) -> Path:
        return self.project_path / "geometry"

    @property
    def fom_path(self) -> Path:
        return self.project_path / "fom"

    @property
    def foms_path(self) -> Path:
        return self.project_path / "foms"

    @property
    def eigenmode_path(self) -> Path:
        return self.project_path / "eigenmode"

    def save(self):
        """Save the entire project with the new folder structure."""
        print(f"Saving project to {self.project_path}")
        
        # 1. Save Geometry
        if self.geometry:
            self.geometry.save_geometry(self.project_path)
            
        # 2. Save Mesh
        # We prefer to save the mesh that is currently in use by the solver
        current_mesh = self.mesh
        if self._fds and self._fds.mesh:
            current_mesh = self._fds.mesh
            
        if current_mesh:
            self.mesh_path.mkdir(parents=True, exist_ok=True)
            from core.persistence import ProjectManager
            pm = ProjectManager(self.base_dir)
            pm.save_ngs_mesh(self.mesh_path, current_mesh)
            
            # Also save global FES if available from solver
            if self._fds and hasattr(self._fds, '_fes_global') and self._fds._fes_global:
                pm.save_ngs_fes(self.mesh_path, self._fds._fes_global)

        # 3. Save FDS Results
        if self._fds:
            self._fds._project_path = self.project_path
            self.fds_path.mkdir(parents=True, exist_ok=True)
            self._fds.save(path=self.fds_path)

        # 4. Global Metadata
        metadata = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "has_geometry": self.geometry is not None,
            "has_mesh": current_mesh is not None,
            "has_fds": self._fds is not None,
            "order": self._order,
            "n_port_modes": self._n_port_modes,
            "bc": self.bc,
        }
        ProjectManager.save_json(self.project_path, metadata, filename="project.json")

    @classmethod
    def load(cls, name: str, base_dir: Optional[Union[str, Path]] = None) -> EMProject:
        """Load a project from disk."""
        base_dir = Path(base_dir) if base_dir else Path.cwd()
        # The __init__ already handles searching and automatic loading
        return cls(name=name, base_dir=base_dir)

    def __repr__(self) -> str:
        return f"EMProject({self.name}, path={self.project_path})"
