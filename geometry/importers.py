"""Geometry importers for STEP and GMSH files with visualization."""

import os
import tempfile
from typing import List, Optional, Tuple, Callable, Dict, Union, Literal
import numpy as np

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Vec
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

# Must import OCC stuff before netgen
from netgen.occ import OCCGeometry, Glue, X, Y, Z, Axis
from netgen.webgui import Draw as NetgenDraw
from ngsolve import Mesh

from .base import BaseGeometry


# ==================== COLOR UTILITIES ====================

def generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        # Convert HSV to RGB (saturation=0.7, value=0.9)
        h = hue * 6
        c = 0.9 * 0.7
        x = c * (1 - abs(h % 2 - 1))
        m = 0.9 - c

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append((r + m, g + m, b + m))
    return colors


def get_shape_bounding_box(shape: TopoDS_Shape) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Get bounding box of an OCC shape."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def count_solids(shape: TopoDS_Shape) -> int:
    """Count number of solids in a shape."""
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def get_solids(shape: TopoDS_Shape) -> List[TopoDS_Shape]:
    """Extract individual solids from a compound shape."""
    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    return solids


# ==================== COORDINATE AXES ====================

def create_coordinate_axes(
        origin: Tuple[float, float, float] = (0, 0, 0),
        length: float = 1.0,
        radius: float = 0.02
) -> Dict[str, TopoDS_Shape]:
    """
    Create coordinate axes as OCC shapes.

    Parameters
    ----------
    origin : tuple
        Origin point (x, y, z)
    length : float
        Length of each axis
    radius : float
        Radius of axis cylinders (as fraction of length)

    Returns
    -------
    dict
        Dictionary with 'x', 'y', 'z' axis shapes
    """
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
    from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    r = length * radius
    cone_height = length * 0.15
    cone_radius = r * 2.5
    cyl_length = length - cone_height

    axes = {}

    # X-axis (red)
    ax_x = gp_Ax2(gp_Pnt(*origin), gp_Dir(1, 0, 0))
    cyl_x = BRepPrimAPI_MakeCylinder(ax_x, r, cyl_length).Shape()
    ax_cone_x = gp_Ax2(gp_Pnt(origin[0] + cyl_length, origin[1], origin[2]), gp_Dir(1, 0, 0))
    cone_x = BRepPrimAPI_MakeCone(ax_cone_x, cone_radius, 0, cone_height).Shape()
    axes['x'] = BRepAlgoAPI_Fuse(cyl_x, cone_x).Shape()

    # Y-axis (green)
    ax_y = gp_Ax2(gp_Pnt(*origin), gp_Dir(0, 1, 0))
    cyl_y = BRepPrimAPI_MakeCylinder(ax_y, r, cyl_length).Shape()
    ax_cone_y = gp_Ax2(gp_Pnt(origin[0], origin[1] + cyl_length, origin[2]), gp_Dir(0, 1, 0))
    cone_y = BRepPrimAPI_MakeCone(ax_cone_y, cone_radius, 0, cone_height).Shape()
    axes['y'] = BRepAlgoAPI_Fuse(cyl_y, cone_y).Shape()

    # Z-axis (blue)
    ax_z = gp_Ax2(gp_Pnt(*origin), gp_Dir(0, 0, 1))
    cyl_z = BRepPrimAPI_MakeCylinder(ax_z, r, cyl_length).Shape()
    ax_cone_z = gp_Ax2(gp_Pnt(origin[0], origin[1], origin[2] + cyl_length), gp_Dir(0, 0, 1))
    cone_z = BRepPrimAPI_MakeCone(ax_cone_z, cone_radius, 0, cone_height).Shape()
    axes['z'] = BRepAlgoAPI_Fuse(cyl_z, cone_z).Shape()

    return axes


# ==================== STEP IMPORTER WITH VISUALIZATION ====================

class STEPImporter(BaseGeometry):
    """
    Import geometry from STEP file with optional splitting planes.

    Parameters
    ----------
    filepath : str
        Path to STEP file
    unit : str
        Unit of the STEP file ('mm', 'cm', 'm')
    auto_build : bool
        If True, automatically build geometry and mesh on init.
        Set to False if you want to add splitting planes first.
    maxh : float
        Maximum mesh element size (only used if auto_build=True)

    Examples
    --------
    # Simple import without splitting
    >>> geo = STEPImporter("cavity.step", unit='mm', maxh=0.05)
    >>> geo.show('mesh')

    # Import with splitting (deferred build)
    >>> geo = STEPImporter("cavity.step", unit='mm', auto_build=False)
    >>> geo.add_splitting_plane((-0.1, -0.1, 0.05), (0.1, 0.1, 0.05))
    >>> geo.show_occ(show_planes=True)  # Preview before splitting
    >>> geo.split()
    >>> geo.show_occ()  # View split result
    >>> geo.finalize(maxh=0.05)
    >>> geo.show('mesh')
    """

    UNIT_MAP = {'mm': 1000, 'cm': 100, 'm': 1}

    # Default colors
    GEOMETRY_COLOR = (0.7, 0.7, 0.8)  # Light gray-blue
    PLANE_COLOR = (1.0, 0.5, 0.0)     # Orange
    PLANE_TRANSPARENCY = 0.7

    # Axis colors
    AXIS_COLORS = {
        'x': (1.0, 0.0, 0.0),  # Red
        'y': (0.0, 1.0, 0.0),  # Green
        'z': (0.0, 0.0, 1.0),  # Blue
    }

    def __init__(
            self,
            filepath: str,
            unit: str = 'mm',
            auto_build: bool = True,
            maxh: float = None
    ):
        super().__init__()
        self.filepath = filepath
        self.unit = unit
        self.maxh = maxh

        # Internal OCC state
        self._occ_shape = None
        self._planes = []
        self._plane_corners = []  # Store corner coordinates for reference
        self._is_split = False

        # Load the raw OCC shape
        self._load_occ_shape()

        # Auto-build if requested
        if auto_build:
            self.build()
            self.generate_mesh(maxh=maxh)

    def _load_occ_shape(self) -> None:
        """Load STEP file into raw OCC shape."""
        if self.unit not in self.UNIT_MAP:
            raise ValueError(f"Unknown unit: {self.unit}. Use 'mm', 'cm', or 'm'.")

        reader = STEPControl_Reader()
        reader.SetSystemLengthUnit(self.UNIT_MAP[self.unit])

        status = reader.ReadFile(self.filepath)
        if status != IFSelect_RetDone:
            raise RuntimeError(f"Failed to read STEP file: {self.filepath}")

        reader.TransferRoots()
        self._occ_shape = reader.OneShape()

    def add_splitting_plane(
            self,
            corner1: Tuple[float, float, float],
            corner2: Tuple[float, float, float],
            normal_axis: str = 'auto'
    ) -> 'STEPImporter':
        """
        Add a plane for splitting the geometry.

        The plane is defined by two opposite corners of a rectangle. The plane's
        normal direction is determined by which coordinate is constant between
        the two corners (or can be explicitly specified).

        Parameters
        ----------
        corner1 : tuple
            First corner (x1, y1, z1)
        corner2 : tuple
            Opposite corner (x2, y2, z2)
            One coordinate should be the same as corner1 for axis-aligned planes.
        normal_axis : str
            Normal direction: 'x', 'y', 'z', or 'auto' (default).
            If 'auto', detects from which coordinate is constant.

        Returns
        -------
        self : STEPImporter
            Returns self for method chaining

        Examples
        --------
        # Z-normal plane (XY plane at z=0.5)
        importer.add_splitting_plane((0, 0, 0.5), (1, 1, 0.5))

        # X-normal plane (YZ plane at x=0.3)
        importer.add_splitting_plane((0.3, 0, 0), (0.3, 1, 1))

        # Y-normal plane (XZ plane at y=0.2)
        importer.add_splitting_plane((0, 0.2, 0), (1, 0.2, 1))
        """
        if self.geo is not None:
            raise RuntimeError(
                "Cannot add splitting planes after build(). "
                "Use auto_build=False in constructor."
            )

        x1, y1, z1 = corner1
        x2, y2, z2 = corner2

        # Determine normal axis
        if normal_axis == 'auto':
            # Find which coordinate is constant (or nearly constant)
            tol = 1e-10
            x_const = abs(x2 - x1) < tol
            y_const = abs(y2 - y1) < tol
            z_const = abs(z2 - z1) < tol

            const_count = sum([x_const, y_const, z_const])

            if const_count == 0:
                raise ValueError(
                    "Cannot auto-detect plane normal: no coordinate is constant. "
                    "Specify normal_axis='x', 'y', or 'z' explicitly."
                )
            elif const_count > 1:
                # Multiple constant coordinates - plane degenerates to line or point
                # Pick the first constant one, but warn
                if x_const:
                    normal_axis = 'x'
                elif y_const:
                    normal_axis = 'y'
                else:
                    normal_axis = 'z'
            else:
                if x_const:
                    normal_axis = 'x'
                elif y_const:
                    normal_axis = 'y'
                else:
                    normal_axis = 'z'

        normal_axis = normal_axis.lower()

        # Build the rectangular face based on normal direction
        wire = BRepBuilderAPI_MakePolygon()

        if normal_axis == 'z':
            # XY plane at constant z
            z = z1  # Use z from corner1
            wire.Add(gp_Pnt(x1, y1, z))
            wire.Add(gp_Pnt(x1, y2, z))
            wire.Add(gp_Pnt(x2, y2, z))
            wire.Add(gp_Pnt(x2, y1, z))

        elif normal_axis == 'x':
            # YZ plane at constant x
            x = x1  # Use x from corner1
            wire.Add(gp_Pnt(x, y1, z1))
            wire.Add(gp_Pnt(x, y1, z2))
            wire.Add(gp_Pnt(x, y2, z2))
            wire.Add(gp_Pnt(x, y2, z1))

        elif normal_axis == 'y':
            # XZ plane at constant y
            y = y1  # Use y from corner1
            wire.Add(gp_Pnt(x1, y, z1))
            wire.Add(gp_Pnt(x1, y, z2))
            wire.Add(gp_Pnt(x2, y, z2))
            wire.Add(gp_Pnt(x2, y, z1))

        else:
            raise ValueError(
                f"Invalid normal_axis: '{normal_axis}'. Must be 'x', 'y', 'z', or 'auto'."
            )

        wire.Close()

        face = BRepBuilderAPI_MakeFace(wire.Shape())
        self._planes.append(face)
        self._plane_corners.append((corner1, corner2, normal_axis))

        return self

    def add_splitting_plane_at_x(
            self,
            x: float,
            margin: float = 0.1
    ) -> 'STEPImporter':
        """
        Add a splitting plane at a given x-coordinate (YZ plane).

        Parameters
        ----------
        x : float
            X-coordinate for the plane
        margin : float
            Extra margin beyond geometry bounds (as fraction)

        Returns
        -------
        self : STEPImporter
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        pmin, pmax = get_shape_bounding_box(self._occ_shape)

        dy = (pmax[1] - pmin[1]) * margin
        dz = (pmax[2] - pmin[2]) * margin

        corner1 = (x, pmin[1] - dy, pmin[2] - dz)
        corner2 = (x, pmax[1] + dy, pmax[2] + dz)

        return self.add_splitting_plane(corner1, corner2, normal_axis='x')

    def add_splitting_plane_at_y(
            self,
            y: float,
            margin: float = 0.1
    ) -> 'STEPImporter':
        """
        Add a splitting plane at a given y-coordinate (XZ plane).

        Parameters
        ----------
        y : float
            Y-coordinate for the plane
        margin : float
            Extra margin beyond geometry bounds (as fraction)

        Returns
        -------
        self : STEPImporter
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        pmin, pmax = get_shape_bounding_box(self._occ_shape)

        dx = (pmax[0] - pmin[0]) * margin
        dz = (pmax[2] - pmin[2]) * margin

        corner1 = (pmin[0] - dx, y, pmin[2] - dz)
        corner2 = (pmax[0] + dx, y, pmax[2] + dz)

        return self.add_splitting_plane(corner1, corner2, normal_axis='y')

    def add_splitting_plane_at_z(
            self,
            z: float,
            margin: float = 0.1
    ) -> 'STEPImporter':
        """
        Add a splitting plane at a given z-coordinate (XY plane).

        Parameters
        ----------
        z : float
            Z-coordinate for the plane
        margin : float
            Extra margin beyond geometry bounds (as fraction)

        Returns
        -------
        self : STEPImporter
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        pmin, pmax = get_shape_bounding_box(self._occ_shape)

        dx = (pmax[0] - pmin[0]) * margin
        dy = (pmax[1] - pmin[1]) * margin

        corner1 = (pmin[0] - dx, pmin[1] - dy, z)
        corner2 = (pmax[0] + dx, pmax[1] + dy, z)

        return self.add_splitting_plane(corner1, corner2, normal_axis='z')

    def split(self) -> 'STEPImporter':
        """
        Split geometry using added planes, then auto-rebuild.

        After splitting, :meth:`build` is called automatically so that
        solids, ports, and wall boundaries are named deterministically.
        Call :meth:`generate_mesh` next.

        Returns
        -------
        self : STEPImporter
            Returns self for method chaining
        """
        if not self._planes:
            raise ValueError(
                "No splitting planes added. Use add_splitting_plane() first."
            )

        # Reset geometry if re-splitting
        self.geo = None
        self._bc_explicitly_set = False

        splitter = BOPAlgo_Splitter()
        splitter.SetNonDestructive(False)
        splitter.AddArgument(self._occ_shape)

        for plane in self._planes:
            splitter.AddTool(plane.Shape())

        splitter.Perform()
        self._occ_shape = splitter.Shape()
        self._is_split = True

        # Auto-rebuild: build() will name solids, ports, and walls
        self.build()

        return self

    def _export_occ_shape_to_temp(self) -> str:
        """Export raw OCC shape to a temporary STEP file."""
        fd, temp_path = tempfile.mkstemp(suffix='.step')
        os.close(fd)

        Interface_Static.SetCVal("write.step.schema", "AP214")
        writer = STEPControl_Writer()
        writer.Transfer(self._occ_shape, STEPControl_AsIs)
        writer.Write(temp_path)

        return temp_path

    def build(self) -> None:
        """
        Build NGSolve geometry from OCC shape.

        This converts the internal OCC shape to an NGSolve-compatible
        OCCGeometry object. If the geometry was split, solids are
        glued together for proper mesh connectivity.

        After building:
        - All faces are named ``'wall'``
        - If split: solids are named ``cell_1``, ``cell_2``, etc. and
          port faces are named ``port1``, ``port2``, etc.
        - ``self.bc`` is set to ``'wall'``
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        # Export to temp file and reimport as NGSolve OCCGeometry
        temp_path = self._export_occ_shape_to_temp()

        try:
            occ_geo = OCCGeometry(temp_path)

            if self._is_split and hasattr(occ_geo, 'solids') and len(occ_geo.solids) > 1:
                # Glue solids together for mesh connectivity
                self.geo = Glue([solid for solid in occ_geo.solids])
            else:
                # Single solid or not split - extract the shape
                self.geo = occ_geo.shape
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # --- Deterministic boundary naming ---
        # Step 1: Name ALL faces 'wall'
        self._name_all_faces_wall()

        # Step 2: If split, auto-name solids and ports
        if self._is_split:
            self._auto_name_split_geometry()

        # Step 3: Set boundary condition
        self.bc = 'wall'
        self._bc_explicitly_set = True

    def finalize(self, maxh: Optional[float] = None) -> 'STEPImporter':
        """
        Build geometry and generate mesh.

        Convenience method for deferred building workflow.

        Parameters
        ----------
        maxh : float, optional
            Maximum mesh element size. Uses value from constructor if not specified.

        Returns
        -------
        self : STEPImporter
            Returns self for method chaining
        """
        if maxh is not None:
            self.maxh = maxh

        self.build()
        self.generate_mesh(maxh=self.maxh)
        return self

    # ==================== OCC VISUALIZATION ====================

    def show_occ(
            self,
            show_planes: bool = True,
            show_axes: bool = True,
            show_edges: bool = True,
            color_by_solid: bool = True,
            geometry_color: Optional[Tuple[float, float, float]] = None,
            plane_color: Optional[Tuple[float, float, float]] = None,
            plane_transparency: float = 0.7,
            axes_scale: Optional[float] = None,
            **kwargs
    ):
        """
        Display the OCC geometry using JupyterRenderer.

        Parameters
        ----------
        show_planes : bool
            Whether to show splitting planes
        show_axes : bool
            Whether to show coordinate axes
        show_edges : bool
            Whether to render edges
        color_by_solid : bool
            If True, color each solid differently. If False, use single color.
        geometry_color : tuple, optional
            RGB color for geometry (if not coloring by solid)
        plane_color : tuple, optional
            RGB color for splitting planes
        plane_transparency : float
            Transparency of splitting planes (0=opaque, 1=fully transparent)
        axes_scale : float, optional
            Scale factor for coordinate axes. Auto-computed from geometry if None.
        **kwargs
            Additional arguments passed to JupyterRenderer

        Returns
        -------
        renderer
            JupyterRenderer instance
        """

        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        # Create renderer
        rnd = JupyterRenderer()

        # Get bounding box for axes scaling
        pmin, pmax = get_shape_bounding_box(self._occ_shape)
        bbox_size = max(pmax[0] - pmin[0], pmax[1] - pmin[1], pmax[2] - pmin[2])

        if axes_scale is None:
            axes_scale = bbox_size * 0.3

        # Display geometry
        if color_by_solid and self._is_split:
            # Color each solid differently
            solids = get_solids(self._occ_shape)
            n_solids = len(solids)

            if n_solids > 1:
                colors = generate_distinct_colors(n_solids)
                for i, solid in enumerate(solids):
                    rnd.DisplayShape(
                        solid,
                        render_edges=show_edges,
                        # color=colors[i],
                        **kwargs
                    )
                print(f"Displayed {n_solids} solids with distinct colors")
            else:
                # Single solid
                color = geometry_color or self.GEOMETRY_COLOR
                rnd.DisplayShape(
                    self._occ_shape,
                    render_edges=show_edges,
                    # color=color,
                    **kwargs
                )
        else:
            # Single color for entire geometry
            color = geometry_color or self.GEOMETRY_COLOR
            rnd.DisplayShape(
                self._occ_shape,
                render_edges=show_edges,
                # color=color,
                **kwargs
            )

        # Display splitting planes
        if show_planes and self._planes:
            p_color = plane_color or self.PLANE_COLOR
            for i, plane in enumerate(self._planes):
                rnd.DisplayShape(
                    plane.Shape(),
                    render_edges=True,
                    # color=p_color,
                    # transparency=False
                )
            print(f"Displayed {len(self._planes)} splitting plane(s)")

        # Display coordinate axes
        if show_axes:
            # Position axes at minimum corner of bounding box
            origin = (pmin[0] - bbox_size * 0.1,
                      pmin[1] - bbox_size * 0.1,
                      pmin[2] - bbox_size * 0.1)

            axes = create_coordinate_axes(origin, length=axes_scale)
            for axis_name, axis_shape in axes.items():
                rnd.DisplayShape(
                    axis_shape,
                    render_edges=False,
                    # color=self.AXIS_COLORS[axis_name]
                )

        # Show the renderer
        rnd.Display()

        # return rnd

    def show_planes_only(self, **kwargs):
        """
        Display only the splitting planes.

        Parameters
        ----------
        **kwargs
            Arguments passed to JupyterRenderer

        Returns
        -------
        renderer
            JupyterRenderer instance
        """
        try:
            from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer
        except ImportError:
            raise ImportError("JupyterRenderer not available.")

        if not self._planes:
            raise ValueError("No splitting planes added.")

        rnd = JupyterRenderer()

        colors = generate_distinct_colors(len(self._planes))
        for i, plane in enumerate(self._planes):
            rnd.DisplayShape(
                plane.Shape(),
                render_edges=True,
                # color=colors[i],
                # transparency=0.3
            )
            # Print plane info
            c1, c2 = self._plane_corners[i]
            print(f"Plane {i+1}: z={c1[2]:.4f}")

        rnd.Display()
        return rnd

    def show_split_preview(self, **kwargs):
        """
        Preview the geometry with planes before splitting.

        Alias for show_occ(show_planes=True).
        """
        return self.show_occ(show_planes=True, **kwargs)

    # ==================== NETGEN/NGSOLVE VISUALIZATION ====================

    def show(
            self,
            what: Literal["geometry", "mesh", "geo", "occ"] = "geometry",
            **kwargs
    ) -> None:
        """
        Display the geometry or mesh.

        Parameters
        ----------
        what : str
            What to display:
            - "geometry" or "geo": NGSolve/Netgen geometry view
            - "mesh": NGSolve mesh view
            - "occ": Raw OCC geometry with JupyterRenderer
        **kwargs
            Additional arguments passed to Draw function

        Raises
        ------
        ValueError
            If geometry/mesh is not built or invalid option is provided.
        """
        what = what.lower()

        if what == "occ":
            self.show_occ(**kwargs)
        elif what in ("geometry", "geo"):
            if self.geo is None:
                raise ValueError("Geometry not built. Call build() first.")
            NetgenDraw(self.geo, **kwargs)
        elif what == "mesh":
            if self.mesh is None:
                raise ValueError("Mesh not generated. Call generate_mesh() first.")
            NetgenDraw(self.mesh, **kwargs)
        else:
            raise ValueError(
                f"Invalid option '{what}'. Use 'geometry', 'geo', 'mesh', or 'occ'."
            )

    def show_colored_solids(self, maxh: Optional[float] = None, **kwargs):
        """
        Build geometry with colored solids and display.

        Useful for visualizing split geometry in NGSolve.

        Parameters
        ----------
        maxh : float, optional
            Mesh size for visualization
        **kwargs
            Arguments passed to Draw
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        temp_path = self._export_occ_shape_to_temp()

        try:
            occ_geo = OCCGeometry(temp_path)

            if hasattr(occ_geo, 'solids') and len(occ_geo.solids) > 1:
                geo = Glue([solid for solid in occ_geo.solids])
                n_solids = len(geo.solids)

                colors = generate_distinct_colors(n_solids)
                for i, solid in enumerate(geo.solids):
                    solid.mat(f"cell_{i+1}")
                    # Convert to 0-1 range for netgen
                    solid.faces.col = colors[i]

                print(f"Colored {n_solids} solids")
            else:
                geo = occ_geo.shape

            NetgenDraw(geo, **kwargs)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # ==================== GEOMETRY INFO ====================

    def get_bounding_box(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Get bounding box of the geometry."""
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")
        return get_shape_bounding_box(self._occ_shape)

    def get_info(self) -> Dict:
        """Get geometry information."""
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        pmin, pmax = self.get_bounding_box()

        return {
            'filepath': self.filepath,
            'unit': self.unit,
            'n_solids': count_solids(self._occ_shape),
            'is_split': self._is_split,
            'n_planes': len(self._planes),
            'plane_positions': [c[0][2] for c in self._plane_corners],
            'bounding_box': {
                'min': pmin,
                'max': pmax,
                'size': tuple(pmax[i] - pmin[i] for i in range(3))
            },
            'is_built': self.geo is not None,
            'has_mesh': self.mesh is not None
        }

    def print_info(self) -> None:
        """Print geometry information."""
        info = self.get_info()

        print("\n" + "=" * 60)
        print("STEP Geometry Information")
        print("=" * 60)
        print(f"File: {info['filepath']}")
        print(f"Unit: {info['unit']}")
        print(f"Number of solids: {info['n_solids']}")
        print(f"Is split: {info['is_split']}")
        print(f"Number of splitting planes: {info['n_planes']}")
        if info['plane_positions']:
            print(f"Plane positions (z): {info['plane_positions']}")
        print(f"\nBounding Box:")
        print(f"  Min: ({info['bounding_box']['min'][0]:.4f}, "
              f"{info['bounding_box']['min'][1]:.4f}, "
              f"{info['bounding_box']['min'][2]:.4f})")
        print(f"  Max: ({info['bounding_box']['max'][0]:.4f}, "
              f"{info['bounding_box']['max'][1]:.4f}, "
              f"{info['bounding_box']['max'][2]:.4f})")
        print(f"  Size: ({info['bounding_box']['size'][0]:.4f}, "
              f"{info['bounding_box']['size'][1]:.4f}, "
              f"{info['bounding_box']['size'][2]:.4f})")
        print(f"\nNGSolve geometry built: {info['is_built']}")
        print(f"Mesh generated: {info['has_mesh']}")
        print("=" * 60)

    # ==================== EXISTING METHODS ====================

    def name_solids(
            self,
            naming_func: Optional[Callable[[int, 'solid'], str]] = None,
            sort_axis: str = 'Z',
            port_axis: Optional[str] = None,
            port_prefix: str = 'port',
            print_info: bool = False,
    ) -> 'STEPImporter':
        """
        Name solids in the geometry.

        .. deprecated::
            For split geometries, :meth:`split` now auto-names everything.
            For single-solid, use :meth:`define_ports` instead.
            This method is kept for backward compatibility.

        Solids are sorted by their centroid position along the specified axis
        before naming, ensuring consistent port assignment regardless of
        internal OCC ordering.

        Parameters
        ----------
        naming_func : callable, optional
            Function that takes (index, solid) and returns a material name.
            Default: 'cell_1', 'cell_2', etc.
        sort_axis : str
            Axis to sort solids by. Default 'Z'.
        port_axis : str, optional
            Axis for port faces. Defaults to sort_axis.
        port_prefix : str
            Prefix for port names. Default 'port'.

        Returns
        -------
        self : STEPImporter
        """
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")

        # Step 1: Name all faces 'wall' first
        self._name_all_faces_wall()

        # Step 2: Name solids and ports
        self._auto_name_split_geometry(
            sort_axis=sort_axis,
            port_axis=port_axis,
            port_prefix=port_prefix,
            naming_func=naming_func,
            print_info=print_info,
        )

        # Step 3: Set boundary condition
        self.bc = 'wall'
        self._bc_explicitly_set = True

        return self

    # ==================== INTERNAL NAMING HELPERS ====================

    def _name_all_faces_wall(self) -> None:
        """
        Name every face on the geometry ``'wall'``.

        Called from :meth:`build` before port assignment, so that all
        non-port boundaries have a deterministic PEC-compatible name.
        """
        if self.geo is None:
            return

        try:
            solids = list(self.geo.solids)
            for solid in solids:
                for face in solid.faces:
                    face.name = 'wall'
        except AttributeError:
            # Single OCC shape without .solids
            try:
                for face in self.geo.faces:
                    face.name = 'wall'
            except AttributeError:
                pass

    def _auto_name_split_geometry(
            self,
            sort_axis: str = 'Z',
            port_axis: str = None,
            port_prefix: str = 'port',
            naming_func=None,
            print_info: bool = True,
    ) -> None:
        """
        Name solids and ports for a split geometry.

        Solids are sorted by centroid along *sort_axis* and named
        ``cell_1``, ``cell_2``, …  Port faces are named sequentially
        ``port1``, ``port2``, …  All other faces remain ``'wall'``.

        Parameters
        ----------
        sort_axis : str
            Axis to sort solids by ('X', 'Y', 'Z').
        port_axis : str, optional
            Axis for port faces.  Defaults to *sort_axis*.
        port_prefix : str
            Prefix for port names.
        naming_func : callable, optional
            ``(index, solid) -> str`` for material names.
        print_info : bool
            Print naming summary.
        """
        if self.geo is None:
            return

        if naming_func is None:
            naming_func = lambda i, s: f"cell_{i + 1}"

        if port_axis is None:
            port_axis = sort_axis

        axis_map = {'X': X, 'Y': Y, 'Z': Z}
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}

        if sort_axis.upper() not in axis_map:
            raise ValueError(f"Invalid sort_axis: {sort_axis}")
        if port_axis.upper() not in axis_map:
            raise ValueError(f"Invalid port_axis: {port_axis}")

        ax = axis_map[port_axis.upper()]
        sort_idx = axis_index[sort_axis.upper()]

        try:
            solids = list(self.geo.solids)
            n_solids = len(solids)

            if n_solids <= 1:
                # Single solid — name material, assign 2 ports
                solid = solids[0] if n_solids == 1 else self.geo
                solid.mat(naming_func(0, solid))
                solid.faces.Min(ax).name = f'{port_prefix}1'
                solid.faces.Max(ax).name = f'{port_prefix}2'
                if print_info:
                    print(f"Single solid: {port_prefix}1 and {port_prefix}2 assigned")
                return

            # Sort solids by centroid position along sort_axis
            def get_solid_centroid(solid):
                bb = solid.bounding_box
                pmin, pmax = bb
                return (pmin[sort_idx] + pmax[sort_idx]) / 2

            solids_with_pos = [(get_solid_centroid(s), i, s) for i, s in enumerate(solids)]
            solids_with_pos.sort(key=lambda x: x[0])

            # Name solids and ports in sorted order
            for new_idx, (pos, orig_idx, solid) in enumerate(solids_with_pos):
                mat_name = naming_func(new_idx, solid)
                solid.mat(mat_name)

                min_port_num = new_idx + 1
                max_port_num = new_idx + 2

                solid.faces.Min(ax).name = f'{port_prefix}{min_port_num}'
                solid.faces.Max(ax).name = f'{port_prefix}{max_port_num}'

            if print_info:
                print(f"Named {n_solids} solids, {n_solids + 1} ports")
                print(f"  External: {port_prefix}1, {port_prefix}{n_solids + 1}")
                if n_solids > 1:
                    internal = ', '.join(f"{port_prefix}{i}" for i in range(2, n_solids + 1))
                    print(f"  Internal: {internal}")

        except AttributeError:
            # Fallback for geometry without .solids attribute
            self.geo.mat(naming_func(0, self.geo))
            self.geo.faces.Min(ax).name = f'{port_prefix}1'
            self.geo.faces.Max(ax).name = f'{port_prefix}2'
            if print_info:
                print(f"Single solid: {port_prefix}1 and {port_prefix}2 assigned")

    def name_faces_by_position(
            self,
            axis: str = 'Z',
            port_prefix: str = 'port'
    ) -> 'STEPImporter':
        """
        Name faces based on their position along an axis.

        Parameters
        ----------
        axis : str
            Axis to use for ordering ('X', 'Y', or 'Z')
        port_prefix : str
            Prefix for port names

        Returns
        -------
        self : STEPImporter
        """
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")

        axis_map = {'X': X, 'Y': Y, 'Z': Z}
        if axis.upper() not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Use 'X', 'Y', or 'Z'.")

        ax = axis_map[axis.upper()]

        try:
            self.geo.faces.Min(ax).name = f"{port_prefix}1"
            self.geo.faces.Max(ax).name = f"{port_prefix}2"
        except Exception as e:
            print(f"Warning: Could not name faces: {e}")

        return self

    def color_solids(self) -> 'STEPImporter':
        """
        Apply distinct colors to each solid in the geometry.

        Returns
        -------
        self : STEPImporter
        """
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")

        try:
            solids = self.geo.solids
            n_solids = len(solids)
            colors = generate_distinct_colors(n_solids)

            for i, solid in enumerate(solids):
                solid.faces.col = colors[i]

            print(f"Applied colors to {n_solids} solids")
        except AttributeError:
            print("Single solid geometry - no color variation needed")

        return self

    @property
    def n_solids(self) -> int:
        """Number of solids in the geometry."""
        if self._occ_shape is None:
            return 0
        return count_solids(self._occ_shape)

    @property
    def n_planes(self) -> int:
        """Number of splitting planes."""
        return len(self._planes)


# ==================== GMSH IMPORTER (unchanged, but add show_occ compatibility) ====================

class GMSHImporter(BaseGeometry):
    """
    Import geometry from GMSH .geo file.

    Parameters
    ----------
    filepath : str
        Path to .geo file
    revolve : bool
        Whether to revolve 2D geometry around X-axis
    auto_build : bool
        If True, automatically build geometry and mesh on init
    maxh : float
        Maximum mesh element size
    """

    def __init__(
            self,
            filepath: str,
            revolve: bool = False,
            auto_build: bool = True,
            maxh: float = 0.05
    ):
        super().__init__()
        self.filepath = filepath
        self.revolve = revolve
        self.maxh = maxh
        self._boundary_map = {}
        self._2d_shape = None

        self._load()

        if auto_build:
            self.build()
            self.generate_mesh(maxh=maxh)

    def _load(self) -> None:
        """Load GMSH geometry."""
        import gmsh

        output_dir = os.path.dirname(self.filepath) or '.'
        step_path = os.path.join(output_dir, "temp_mesh.step")

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 0)

        try:
            gmsh.open(self.filepath)
            gmsh.model.mesh.generate(2)

            # Extract boundary conditions before writing
            self._extract_boundary_conditions()

            gmsh.write(step_path)
        finally:
            gmsh.finalize()

        self._2d_shape = OCCGeometry(step_path, dim=2).shape


        # Clean up temp file
        if os.path.exists(step_path):
            os.remove(step_path)

    def _extract_boundary_conditions(self) -> None:
        """Extract boundary condition labels from GMSH model."""
        import gmsh

        for dim, phys_tag in gmsh.model.getPhysicalGroups():
            if dim != 1:  # Only 1D groups (lines)
                continue
            name = gmsh.model.getPhysicalName(dim, phys_tag)
            for line_id in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
                if isinstance(line_id, tuple):
                    _, line_id = line_id
                self._boundary_map[line_id] = name

    def build(self) -> None:
        """Build the geometry, optionally revolving around X-axis."""
        if self._2d_shape is None:
            raise RuntimeError("No shape loaded.")

        if self.revolve:
            ax = Axis((0, 0, 0), X)
            self.geo = self._2d_shape.faces[0].Revolve(ax, 360)
        else:
            self.geo = self._2d_shape

        # Name ports
        self.geo.faces.Min(X).name = "port1"
        self.geo.faces.Max(X).name = "port2"

    def finalize(self, maxh: Optional[float] = None) -> 'GMSHImporter':
        """
        Build geometry and generate mesh.

        Parameters
        ----------
        maxh : float, optional
            Maximum mesh element size.

        Returns
        -------
        self : GMSHImporter
        """
        if maxh is not None:
            self.maxh = maxh

        self.build()
        self.generate_mesh(maxh=self.maxh)
        return self

    @property
    def boundary_map(self) -> dict:
        """Get mapping of line IDs to boundary names from GMSH."""
        return self._boundary_map.copy()


class TESLACavity(GMSHImporter):
    """
    TESLA 9-cell SRF cavity geometry.

    Parameters
    ----------
    filepath : str
        Path to .geo file defining the cavity cross-section
    maxh : float
        Maximum mesh element size
    """

    def __init__(self, filepath: str, maxh: float = 1.0):
        # Initialize without auto_build so we can customize
        super().__init__(filepath, revolve=True, auto_build=False, maxh=maxh)

        # Build and customize
        self.build()
        self._setup_boundaries()
        self.generate_mesh(maxh=maxh)

    def _setup_boundaries(self) -> None:
        """Set up port and boundary names."""
        if self.geo is None:
            raise RuntimeError("Geometry not built.")

        # Name ports
        self.geo.faces.Min(X).name = "port1"
        self.geo.faces.Max(X).name = "port2"

        # Set default BC
        self.bc = 'default'