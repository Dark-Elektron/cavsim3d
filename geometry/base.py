"""Base geometry class and common geometry operations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
from pathlib import Path
from typing import Literal
from ngsolve import Mesh, BND, Integrate, specialcf
from netgen.occ import OCCGeometry
from netgen.webgui import Draw
from netgen.occ import Glue


class BaseGeometry(ABC):
    """Abstract base class for all geometries."""

    def __init__(self):
        self.geo = None
        self.mesh = None
        self.bc = 'default'
        self._ports = None
        self._boundaries = None

    @abstractmethod
    def build(self) -> None:
        """Build the geometry."""
        pass

    def generate_mesh(self, maxh = None, curve_order: int = 3) -> Mesh:
        """Generate mesh from geometry."""
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")
        if maxh:
            self.mesh = Mesh(OCCGeometry(self.geo).GenerateMesh(maxh=maxh))
        else:
            self.mesh = Mesh(OCCGeometry(self.geo).GenerateMesh())

        self.mesh.Curve(curve_order)
        return self.mesh

    def show(
        self,
        what: Literal["geometry", "mesh", "geo"] = "geometry",
        **kwargs
    ) -> None:
        """
        Display the geometry or mesh.

        Parameters
        ----------
        what : str, optional
            What to display: "geometry" (or "geo") for the CAD geometry,
            "mesh" for the mesh. Default is "geometry".
        **kwargs
            Additional keyword arguments passed to the Draw function.
            Common options include:
            - clipping : dict, e.g., {"x": 0, "y": 0, "z": -1} for clipping plane
            - euler_angles : list, e.g., [30, 20, 0] for rotation
            - deformation : bool, for showing deformation

        Raises
        ------
        ValueError
            If geometry/mesh is not built or invalid option is provided.

        Examples
        --------
        >>> wg = RectangularWaveguide(a=0.1, L=0.5)
        >>> wg.show()  # Show geometry
        >>> wg.show("mesh")  # Show mesh
        >>> wg.show("mesh", clipping={"z": -1})  # Show mesh with clipping
        """
        what = what.lower()

        if what in ("geometry", "geo"):
            if self.geo is None:
                raise ValueError("Geometry not built. Call build() first.")
            # For OCC geometry, use Draw from netgen.occ
            from netgen import occ
            Draw(self.geo, **kwargs)

        elif what == "mesh":
            if self.mesh is None:
                raise ValueError("Mesh not generated. Call generate_mesh() first.")
            Draw(self.mesh, **kwargs)

        else:
            raise ValueError(
                f"Invalid option '{what}'. Use 'geometry', 'geo', or 'mesh'."
            )

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

    def get_boundary_normal(self, boundary_label: str) -> Optional[np.ndarray]:
        """
        Calculate the outward normal vector for a planar boundary face.

        Parameters
        ----------
        boundary_label : str
            Name of the boundary

        Returns
        -------
        np.ndarray or None
            Normal vector (nx, ny, nz), or None if area is zero
        """
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

    # In base.py - add these methods to the BaseGeometry class
    def save_step(self, filename: str) -> None:
        """
        Save the geometry to a STEP file.

        Parameters
        ----------
        filename : str
            Output filename (should end with .step or .stp)

        Raises
        ------
        ValueError
            If geometry has not been built yet.
        """
        self._export('step', filename)


    def save_brep(self, filename: str) -> None:
        """
        Save the geometry to a BREP file.

        Parameters
        ----------
        filename : str
            Output filename (should end with .brep)
        """
        self._export('brep', filename)


    def save_stl(self, filename: str) -> None:
        """
        Save the geometry to an STL file.

        Parameters
        ----------
        filename : str
            Output filename (should end with .stl)
        """
        self._export('stl', filename)


    def _export(
            self,
            format: Literal['step', 'brep', 'stl'],
            filename: str
    ) -> None:
        """
        Internal method to export geometry to various formats.

        Parameters
        ----------
        format : str
            Export format ('step', 'brep', or 'stl')
        filename : str
            Output filename
        """
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")

        path = Path(filename)

        format_config = {
            'step': {
                'extensions': ('.step', '.stp'),
                'default_ext': '.step',
                'writer': lambda f: self.geo.WriteStep(str(f))
            },
            'brep': {
                'extensions': ('.brep',),
                'default_ext': '.brep',
                'writer': lambda f: self.geo.WriteBrep(str(f))
            },
            'stl': {
                'extensions': ('.stl',),
                'default_ext': '.stl',
                'writer': lambda f: self.geo.WriteSTL(str(f))
            }
        }

        config = format_config[format]

        # Add extension if not present
        if path.suffix.lower() not in config['extensions']:
            path = path.with_suffix(config['default_ext'])

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        config['writer'](path)
        print(f"Geometry saved to: {path}")

    def print_info(self) -> None:
        """
        Print detailed summary of the geometry, mesh status, ports, and
        class-specific information.
        """
        class_name = self.__class__.__name__

        print("\n" + "=" * 70)
        print(f"{class_name} Geometry Information")
        print("=" * 70)

        # Common info (available on all geometries)
        print(f"Geometry type:          {class_name}")
        print(f"Boundary condition:     {self.bc}")

        has_geo = self.geo is not None
        has_mesh = self.mesh is not None and isinstance(self.mesh, Mesh)

        print(f"OCC geometry built:     {has_geo}")
        print(f"Mesh generated:         {has_mesh}")

        if has_mesh:
            try:
                nv = self.mesh.nv
                ne = self.mesh.ne
                print(f"  → Vertices:           {nv:,}")
                print(f"  → Elements:           {ne:,}")
            except:
                print("  → Mesh statistics:    (could not read nv/ne)")

        # Ports
        if has_mesh:
            ports = self.ports
            print(f"Detected ports:         {len(ports)}")
            if ports:
                print("  → " + ", ".join(ports))
            else:
                print("  → (no boundaries with 'port' in name)")
        else:
            print("Detected ports:         (mesh not generated yet)")

        # Class-specific details
        print("\nSpecific parameters:")

        if class_name == "RectangularWaveguide":
            print(f"  Width (a):            {getattr(self, 'a', '—'):.6f} m")
            print(f"  Height (b):           {getattr(self, 'b', '—'):.6f} m")
            print(f"  Length (L):           {getattr(self, 'L', '—'):.6f} m")
            if hasattr(self, 'cutoff_frequency_TE10'):
                fc = self.cutoff_frequency_TE10 / 1e9
                print(f"  TE₁₀ cutoff:          {fc:.4f} GHz")

        elif class_name == "CircularWaveguide":
            print(f"  Radius:               {getattr(self, 'radius', '—'):.6f} m")
            print(f"  Length:               {getattr(self, 'length', '—'):.6f} m")
            if hasattr(self, 'cutoff_frequency_TE11'):
                fc_te11 = self.cutoff_frequency_TE11 / 1e9
                fc_tm01 = self.cutoff_frequency_TM01 / 1e9 if hasattr(self, 'cutoff_frequency_TM01') else None
                print(f"  TE₁₁ cutoff:          {fc_te11:.4f} GHz")
                if fc_tm01 is not None:
                    print(f"  TM₀₁ cutoff:          {fc_tm01:.4f} GHz")

        elif class_name == "Box":
            dims = getattr(self, 'dimensions', (None, None, None))
            print(f"  Dimensions (x,y,z):   {dims[0]:.6f} × {dims[1]:.6f} × {dims[2]:.6f} m")
            print(f"  Port faces:           {getattr(self, 'port_faces', '—')}")

        else:
            # Fallback for unknown/future geometries
            print("  (no specific parameters defined for this geometry type)")

        print("=" * 70)


class GeometryCollection:
    """Collection of geometries that can be glued together."""

    def __init__(self, geometries: List[BaseGeometry]):
        self.geometries = geometries
        self.combined_geo = None
        self.combined_mesh = None

    def glue(self, offsets: Optional[List[np.ndarray]] = None):
        """Glue geometries together with optional offsets."""

        geo_list = []
        for i, geom in enumerate(self.geometries):
            if offsets and i < len(offsets):
                geo_list.append(geom.geo.Move(tuple(offsets[i])))
            else:
                geo_list.append(geom.geo)

        self.combined_geo = Glue(geo_list)
        return self.combined_geo

    def show(
        self,
        what: Literal["geometry", "mesh", "geo"] = "geometry",
        **kwargs
    ) -> None:
        """
        Display the combined geometry or mesh.

        Parameters
        ----------
        what : str, optional
            What to display: "geometry" (or "geo") for the CAD geometry,
            "mesh" for the mesh. Default is "geometry".
        **kwargs
            Additional keyword arguments passed to the Draw function.
        """
        what = what.lower()

        if what in ("geometry", "geo"):
            if self.combined_geo is None:
                raise ValueError("Combined geometry not built. Call glue() first.")
            from netgen import occ
            Draw(self.combined_geo, **kwargs)

        elif what == "mesh":
            if self.combined_mesh is None:
                raise ValueError("Combined mesh not generated.")
            Draw(self.combined_mesh, **kwargs)

        else:
            raise ValueError(
                f"Invalid option '{what}'. Use 'geometry', 'geo', or 'mesh'."
            )

