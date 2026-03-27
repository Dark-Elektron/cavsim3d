from __future__ import annotations
import os
from core.persistence import ProjectManager
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
import json
from datetime import datetime
import shutil
from utils.io_utils import get_user_confirmation

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
    
    def create_assembly(self, main_axis: str = 'Z') -> 'Assembly':
        """
        Create a new multi-component assembly for this project.
        
        This sets the project's geometry to an empty Assembly and returns it.
        You can then add components to the assembly using assembly.add().
        
        Parameters
        ----------
        main_axis : str
            Primary axis for concatenation ('X', 'Y', or 'Z')
            
        Returns
        -------
        Assembly
            The new assembly instance
        """
        from geometry.assembly import Assembly
        self.geometry = Assembly(main_axis=main_axis)
        self.save()
        return self.geometry

    def __init__(
        self,
        name: str,
        base_dir: Optional[Union[str, Path]] = None,
        geometry: Optional[BaseGeometry] = None,
        bc: str = None,
    ):
        self.name = name
        # Use current directory if base_dir is not provided
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.project_path = self.base_dir / self.name
        
        self._geometry = geometry
        self.bc = bc
        self._mesh: Optional[Mesh] = None
        self._fds: Optional[FrequencyDomainSolver] = None
        self._order = 3
        self._n_port_modes = 1
        self._loading = False  # Guard flag to prevent save() during _initial_load()
        
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

        self._loading = True  # Prevent save() from being triggered during load

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        self.bc = metadata.get("bc", self.bc)
        self._order = metadata.get("order", self._order)
        self._n_port_modes = metadata.get("n_port_modes", self._n_port_modes)

        # 1. Load Geometry FIRST
        has_geo = metadata.get("has_geometry", False)
        # Self-healing: check if geometry folder has contents even if flag is false
        if not has_geo and self.geometry_path.exists() and any(self.geometry_path.iterdir()):
            has_geo = True

        if has_geo:
            from geometry.base import BaseGeometry
            try:
                self.geometry = BaseGeometry.load_geometry(self.project_path)
            except Exception as e:
                print(f"Warning: Could not load geometry: {e}")

        # 2. Load Mesh SECOND (needed for FDS port modes)
        has_mesh = metadata.get("has_mesh", False)
        # Self-healing: check if mesh folder has contents even if flag is false
        if not has_mesh and self.mesh_path.exists() and (self.mesh_path / "mesh.pkl").exists():
            has_mesh = True

        if has_mesh:
            from core.persistence import ProjectManager
            pm = ProjectManager(self.base_dir)
            self.mesh = pm.load_ngs_mesh(self.mesh_path)

        # 3. Load Solver (FDS) LAST - needs mesh for port mode reconstruction
        if metadata.get("has_fds"):
            from solvers.frequency_domain import FrequencyDomainSolver

            # Pass mesh to load method so port modes can be reconstructed
            self._fds = FrequencyDomainSolver.load_from_path(
                self.fds_path,
                geometry=self.geometry,
                mesh=self.mesh,  # Pass mesh here
                order=self._order,
                bc=self.bc
            )

            if self._fds:
                self._fds._project_path = self.project_path
                # Only sync mesh if the FDS doesn't have one already (load_from_path handles it)
                if self.mesh and self._fds.mesh is None:
                    self._fds.mesh = self.mesh
                    # Restore FES
                    pm = ProjectManager(self.base_dir)
                    fes = pm.load_ngs_fes(self.fds_path)
                    if fes:
                        self._fds._fes_global = fes

        self._loading = False  # Re-enable save()

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

    def import_geometry(self, filepath: Union[str, Path], force: bool = False, **kwargs) -> 'OCCImporter':
        """Import geometry from a file into the project."""
        if (self.has_mesh() or self.has_results()) and not force:
            from utils.io_utils import get_user_confirmation
            if not get_user_confirmation(
                "\nWARNING: Importing new geometry will invalidate the current mesh and simulation results.\n"
                "Do you want to continue and delete existing results?"
            ):
                print("Aborting geometry import.")
                return self.geometry

            self.invalidate_mesh()

        self.geometry = self.create_importer(filepath, **kwargs)
        self.save()  # Auto-save after successful import
        return self.geometry

    def create_importer(self, filepath: Union[str, Path], **kwargs) -> 'OCCImporter':
        """Create an OCCImporter for a CAD file (without necessarily setting it as the project geometry)."""
        from geometry.importers import OCCImporter
        return OCCImporter(str(filepath), **kwargs)

    def create_primitive(self, primitive_type: str, force: bool = False, **kwargs) -> BaseGeometry:
        """Create a primitive geometry and associate it with the project."""
        if (self.has_mesh() or self.has_results()) and not force:
            if not get_user_confirmation(
                "\nWARNING: Geometry change will invalidate the current mesh and simulation results.\n"
                "Do you want to continue and delete existing results?"
            ):
                print("Aborting primitive creation.")
                return self.geometry

            self.invalidate_mesh()

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
        self.save()  # Auto-save
        return self.geometry

    def generate_mesh(self, force: bool = False, **kwargs) -> Mesh:
        """
        Generate mesh from current geometry.
        
        Automatically invalidates existing simulation results if the mesh changes.
        """
        if self.geometry is None:
            raise RuntimeError("Cannot generate mesh without geometry.")

        if self.has_results() and not force:
            if not get_user_confirmation(
                "\nWARNING: Re-generating the mesh will invalidate existing simulation results.\n"
                "Do you want to continue and delete existing results?"
            ):
                print("Aborting mesh generation.")
                return self.mesh

            self.invalidate_results()

        self.mesh = self.geometry.generate_mesh(**kwargs)
        self.save()
        return self.mesh

    def has_mesh(self) -> bool:
        """Check if mesh exists (either in memory or on disk)."""
        if self.mesh is not None:
            return True
        return (self.mesh_path / "mesh.pkl").exists()

    def has_results(self) -> bool:
        """Check if any simulation results exist (either in memory or on disk)."""
        # Check in memory
        if self._fds and (getattr(self._fds, '_fom_cache', None) or getattr(self._fds, '_resonant_mode_cache', None)):
            return True
        
        # Check on disk (config.json marks a valid simulation save)
        return (self.fds_path / "config.json").exists()

    def invalidate_mesh(self) -> None:
        """Invalidate the mesh and all downstream results (fom, rom, etc.)."""
        print(f"Invalidating mesh for project '{self.name}'...")
        
        # 1. Physical Cleanup (Content only, preserve directory)
        if self.mesh_path.exists():
            for item in self.mesh_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # 2. In-Memory Reset
        self.mesh = None
        if self.geometry:
            self.geometry.mesh = None # Sync with geometry object

        # 3. Propagate Downstream
        self.invalidate_results()

    def invalidate_results(self) -> None:
        """Invalidate all simulation results (fom, rom, concat, etc.)."""
        print(f"Invalidating simulation results for project '{self.name}'...")
        
        # 1. Physical Cleanup (Content only, preserve directories)
        for path in [self.fds_path, self.fom_path, self.foms_path, self.eigenmode_path]:
            if path.exists():
                for item in path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        
        # 2. In-Memory Reset
        if self._fds:
            # We don't delete self._fds object, but we clear its entire state.
            # Use full_reset to ensure matrices, FES, and flags are cleared.
            self._fds.full_reset()
            self._fds.mesh = self.mesh # Ensure solver still has access to new mesh if it exists

    @property
    def geometry(self) -> Optional[BaseGeometry]:
        """Current project geometry."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: Optional[BaseGeometry]):
        self._geometry = value
        # Sync solver with new geometry object
        if self._fds:
            self._fds.geometry = value

    @property
    def mesh(self) -> Optional[Mesh]:
        """Project mesh (NGSolve Mesh object)."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Optional[Mesh]):
        self._mesh = value
        if self._fds:
            self._fds.mesh = value
        
        # Auto-save mesh to disk if it exists (but NOT during _initial_load)
        if value is not None and not getattr(self, '_loading', False):
            self.save()

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
        from core.persistence import ProjectManager
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
