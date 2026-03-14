# base.py (updated with tagging)
"""Base geometry class with tagging support."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Literal, Any
import numpy as np
import hashlib
import json
import shutil
import warnings
from pathlib import Path
from datetime import datetime

from ngsolve import Mesh, BND, Integrate, specialcf
from netgen.occ import OCCGeometry, X, Y, Z
from ngsolve.webgui import Draw
from netgen.occ import Glue

from .component_registry import (
    TaggableMixin, ComponentTag, ComputeMethod, 
    get_global_cache, CachedSolution
)


class BaseGeometry(ABC, TaggableMixin):
    """
    Abstract base class for all geometries with tagging support.
    
    Attributes
    ----------
    tag : ComponentTag
        Unique identifier for this component configuration
    compute_method : ComputeMethod
        Method for computing solution (NUMERIC, ANALYTICAL, SEMI_ANALYTICAL)
    custom_tag : str
        User-defined tag for identification
    """

    _AXIS_MAP = {'X': X, 'Y': Y, 'Z': Z}
    _AXIS_ORDER = ['X', 'Y', 'Z']

    def __init__(self):
        self.geo = None
        self.mesh = None
        self.bc = None
        self._bc_explicitly_set = False
        self._ports = None
        self._boundaries = None
        
        # Tagging (from TaggableMixin)
        self._tag = None
        self._custom_tag = None
        self._compute_method = ComputeMethod.NUMERIC

        # History system (unified across all geometry types)
        self._history: List[dict] = []
        self._source_link: Optional[str] = None  # original file path
        self._source_hash: Optional[str] = None  # SHA-256 of source at save time

    @abstractmethod
    def build(self) -> None:
        """Build the geometry."""
        pass

    def generate_mesh(self, maxh=None, curve_order: int = 3) -> Mesh:
        """Generate mesh from geometry."""
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")
        
        if maxh:
            self.mesh = Mesh(OCCGeometry(self.geo).GenerateMesh(maxh=maxh))
        else:
            self.mesh = Mesh(OCCGeometry(self.geo).GenerateMesh())

        self.mesh.Curve(curve_order)
        self.invalidate_tag()  # Mesh changed

        self._record('generate_mesh', maxh=maxh, curve_order=curve_order)
        return self.mesh

    def show(
        self,
        what: Literal["geometry", "mesh", "geo"] = "geometry",
        **kwargs
    ) -> None:
        """Display the geometry or mesh."""
        what = what.lower()

        if what in ("geometry", "geo"):
            if self.geo is None:
                raise ValueError("Geometry not built. Call build() first.")
            Draw(self.geo, **kwargs)
        elif what == "mesh":
            if self.mesh is None:
                raise ValueError("Mesh not generated. Call generate_mesh() first.")
            # Use NGSolve's Draw for NGSolve Mesh objects
            NGSolveDraw(self.mesh, **kwargs)
        else:
            raise ValueError(f"Invalid option '{what}'.")

    @property
    def ports(self) -> List[str]:
        """Get list of port boundary names."""
        if self._ports is None:
            self._ports = [b for b in self.mesh.GetBoundaries() if "port" in b]
        return self._ports

    @property
    def boundaries(self) -> List[str]:
        """Get all boundary names."""
        if self._boundaries is None:
            self._boundaries = list(self.mesh.GetBoundaries())
        return self._boundaries

    @property
    def n_ports(self) -> int:
        """Number of ports."""
        return len(self.ports)

    # === Caching helpers ===
    
    def get_cached_solution(self) -> Optional[CachedSolution]:
        """Get cached solution if available."""
        return get_global_cache().get(self.tag)
    
    def has_cached_solution(self) -> bool:
        """Check if a cached solution exists."""
        return self.tag in get_global_cache()
    
    def cache_solution(self, solution_data: Any, **metadata) -> None:
        """Store a solution in the cache."""
        get_global_cache().store(
            self.tag,
            solution_data,
            self.compute_method,
            **metadata
        )

    # === Keep all other methods from the original base.py ===
    # (define_ports, validate_boundaries, get_boundary_normal, etc.)
    
    def define_ports(self, **kwargs) -> 'BaseGeometry':
        """Define port faces by axis position."""
        # [Same implementation as before]
        # ...
        self.invalidate_tag()  # Ports changed
        return self

    def get_boundary_normal(self, boundary_label: str) -> Optional[np.ndarray]:
        """Calculate outward normal vector for a planar boundary face."""
        nhat = specialcf.normal(3)
        integral_n = Integrate(
            nhat, self.mesh, BND,
            definedon=self.mesh.Boundaries(boundary_label)
        )
        face_area = Integrate(
            1, self.mesh, BND,
            definedon=self.mesh.Boundaries(boundary_label)
        )
        if abs(face_area) < 1e-12:
            return None
        normal = -np.array(integral_n) / face_area
        return np.round(normal, decimals=6)

    def get_point_on_boundary(self, boundary_label: str) -> Optional[Tuple[float, ...]]:
        """Get coordinates of a vertex on the specified boundary."""
        for bel in self.mesh.Elements(BND):
            if bel.mat == boundary_label:
                vertex_index = bel.vertices[0]
                return tuple(self.mesh[vertex_index].point)
        return None

    def save_step(self, filename: str) -> None:
        """Save geometry to STEP file."""
        self._export('step', filename)

    def save_brep(self, filename: str) -> None:
        """Save geometry to BREP file."""
        self._export('brep', filename)

    def _export(self, format: Literal['step', 'brep', 'stl'], filename: str) -> None:
        """Export geometry to file."""
        if self.geo is None:
            raise ValueError("Geometry not built.")
        
        path = Path(filename)
        config = {
            'step': ('.step', lambda f: self.geo.WriteStep(str(f))),
            'brep': ('.brep', lambda f: self.geo.WriteBrep(str(f))),
            'stl': ('.stl', lambda f: self.geo.WriteSTL(str(f)))
        }
        ext, writer = config[format]
        if path.suffix.lower() != ext:
            path = path.with_suffix(ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        print(f"Saved to: {path}")

    def print_info(self) -> None:
        """Print detailed geometry information including tag."""
        class_name = self.__class__.__name__
        
        print("\n" + "=" * 70)
        print(f"{class_name} Geometry Information")
        print("=" * 70)
        
        print(f"Geometry type:          {class_name}")
        print(f"Compute method:         {self.compute_method}")
        print(f"Supports analytical:    {self.supports_analytical}")
        print(f"Boundary condition:     {self.bc}")
        
        print(f"\nComponent Tag:")
        print(f"  Full:                 {self.tag}")
        print(f"  Geometry hash:        {self.tag.geometry_hash[:16]}...")
        if self.custom_tag:
            print(f"  Custom tag:           {self.custom_tag}")
        
        has_cached = self.has_cached_solution()
        print(f"\nCache status:           {'CACHED' if has_cached else 'NOT CACHED'}")
        
        has_mesh = self.mesh is not None
        print(f"\nMesh generated:         {has_mesh}")
        if has_mesh:
            print(f"  Vertices:             {self.mesh.nv:,}")
            print(f"  Elements:             {self.mesh.ne:,}")
            print(f"  Ports:                {self.ports}")
        
        print("=" * 70)

    # === History system ===

    def _record(self, op: str, **params) -> None:
        """Append an operation to the history log."""
        entry = {
            'op': op,
            'timestamp': datetime.now().isoformat(),
        }
        # Filter out None values for cleaner history
        entry.update({k: v for k, v in params.items() if v is not None})
        self._history.append(entry)

    def get_history(self) -> List[dict]:
        """Return the full operation history."""
        return list(self._history)

    # === Geometry persistence ===

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    def save_geometry(self, project_path: Path) -> None:
        """
        Save geometry files, history, and source link to the project.

        Creates ``project_path/geometry/`` containing:
        - ``source_model.<ext>`` — copy of the original source file
        - ``history.json`` — operation log + source link metadata
        """
        geo_dir = Path(project_path) / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)

        source_hash = None
        source_filename = None

        # Copy original source file if we have one
        if self._source_link is not None:
            src = Path(self._source_link)
            if src.exists():
                source_filename = f'source_model{src.suffix}'
                dest = geo_dir / source_filename
                shutil.copy2(str(src), str(dest))
                source_hash = self._file_hash(dest)
                self._source_hash = source_hash

        # Build history metadata
        meta = {
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'source_link': str(self._source_link) if self._source_link else None,
            'source_filename': source_filename,
            'source_hash': source_hash,
            'history': self._history,
        }

        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load_geometry(cls, project_path: Path) -> 'BaseGeometry':
        """
        Load geometry from a saved project by replaying the operation history.

        Reads ``project_path/geometry/history.json``, dispatches to the
        correct subclass, and reconstructs the geometry.
        """
        geo_dir = Path(project_path) / 'geometry'
        history_file = geo_dir / 'history.json'

        if not history_file.exists():
            raise FileNotFoundError(
                f"No geometry history found at {history_file}"
            )

        with open(history_file, 'r') as f:
            meta = json.load(f)

        geo_type = meta['type']
        history = meta['history']
        source_link = meta.get('source_link')
        source_filename = meta.get('source_filename')
        source_hash = meta.get('source_hash')

        # Resolve the geometry file path for replay
        # Use the project-local copy stored in geometry/
        source_file = geo_dir / source_filename if source_filename else None

        # Try to import the module stored in history to register the subclass
        if 'module' in meta:
            try:
                import importlib
                importlib.import_module(meta['module'])
            except (ImportError, KeyError):
                pass

        # Dispatch to the correct subclass
        subclass = cls._get_subclass(geo_type)
        if subclass is None:
            raise ValueError(
                f"Unknown geometry type '{geo_type}'. "
                f"Cannot reconstruct geometry."
            )

        geo = subclass._rebuild_from_history(history, project_path, source_file)
        geo._source_link = source_link
        geo._source_hash = source_hash
        geo._history = history

        # Check if source has changed since save
        geo._check_source_link(project_path)

        return geo

    @classmethod
    def _get_subclass(cls, type_name: str) -> Optional[type]:
        """Find a BaseGeometry subclass by name (searches all subclasses)."""
        # Lazy import to ensure all standard subclasses are registered if not already
        try:
            from . import primitives
            from . import importers
        except (ImportError, ValueError):
            pass

        def _search(klass):
            # Check the class itself
            if klass.__name__ == type_name:
                return klass
            # Search subclasses
            for sub in klass.__subclasses__():
                if sub.__name__ == type_name:
                    return sub
                found = _search(sub)
                if found:
                    return found
            return None
            
        return _search(cls)

    @classmethod
    def _rebuild_from_history(
        cls,
        history: List[dict],
        project_path: Path,
        source_file: Optional[Path] = None,
    ) -> 'BaseGeometry':
        """
        Reconstruct a geometry by replaying its history.

        Subclasses override this to handle their specific operations.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support history-based reconstruction."
        )

    def _check_source_link(self, project_path: Path) -> None:
        """
        Compare the project's geometry copy against the linked source.

        * Source changed → prompt user to update (deletes results) or keep.
        * Source missing → prompt user to unlink (project becomes standalone).
        """
        if self._source_link is None:
            return  # No link — standalone project

        source_path = Path(self._source_link)
        geo_dir = Path(project_path) / 'geometry'

        if not source_path.exists():
            print(f"\nLinked source file not found: {source_path}")
            print("Options:")
            print("  [U] Unlink — project becomes standalone")
            print("  [K] Keep link — ignore for now")
            choice = input("Select [U/K]: ").strip().upper()
            if choice == 'U':
                self._source_link = None
                # Update history.json
                self._update_link_in_history(geo_dir)
                print("Source unlinked. Project is now standalone.")
            else:
                warnings.warn(
                    f"Source file '{source_path}' not found. "
                    f"Link preserved but geometry cannot be updated.",
                    UserWarning,
                    stacklevel=2,
                )
            return

        # Source exists — check for changes
        current_hash = self._file_hash(source_path)
        if self._source_hash is not None and current_hash != self._source_hash:
            print(f"\nLinked source file has been modified: {source_path}")
            print("WARNING: Updating the geometry will DELETE all existing")
            print("         simulation results (matrices, snapshots, FOM, ROM).")
            print("Options:")
            print("  [U] Update — replace geometry and delete results")
            print("  [K] Keep — use the project's saved copy")
            choice = input("Select [U/K]: ").strip().upper()
            if choice == 'U':
                # Copy updated source
                source_filename = None
                for f in geo_dir.iterdir():
                    if f.name.startswith('source_model'):
                        source_filename = f.name
                        break
                if source_filename:
                    shutil.copy2(str(source_path), str(geo_dir / source_filename))
                self._source_hash = current_hash
                self._update_link_in_history(geo_dir)

                # Delete existing results
                self._delete_project_results(project_path)
                print("Geometry updated. Previous results deleted.")
                print("Re-run the simulation to generate new results.")

    def _update_link_in_history(self, geo_dir: Path) -> None:
        """Update the source_link and source_hash in history.json."""
        history_file = geo_dir / 'history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                meta = json.load(f)
            meta['source_link'] = str(self._source_link) if self._source_link else None
            meta['source_hash'] = self._source_hash
            with open(history_file, 'w') as f:
                json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def _delete_project_results(project_path: Path) -> None:
        """Delete all simulation results from a project directory."""
        result_paths = [
            'matrices.h5', 'snapshots.h5', 'fds',
            'fom', 'foms', 'roms', 'port_modes', 'eigenmode'
        ]
        for name in result_paths:
            p = Path(project_path) / name
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)