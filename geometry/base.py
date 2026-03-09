# base.py (updated with tagging)
"""Base geometry class with tagging support."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Literal, Any
import numpy as np
from pathlib import Path

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