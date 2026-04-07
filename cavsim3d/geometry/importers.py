"""Geometry importers with OCC-based loading, splitting, and visualization."""

import os
import re
import tempfile
import warnings
from typing import List, Optional, Tuple, Callable, Dict, Union, Literal
import numpy as np

# PythonOCC imports — must come before netgen imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BOPAlgo import BOPAlgo_Splitter
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon, BRepBuilderAPI_Transform
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Vec
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, topods
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepTools import breptools
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Display.WebGl.jupyter_renderer import JupyterRenderer

# Netgen/NGSolve imports — must come after OCC imports
from netgen.occ import OCCGeometry, Glue, X, Y, Z, Axis
from netgen.webgui import Draw as NetgenDraw
from ngsolve import Mesh

from .base import BaseGeometry, _display_webgui_fallback


# ==================== STEP LABEL EXTRACTION ====================

def extract_brep_names_from_step(filename: str) -> List[str]:
    """Extract MANIFOLD_SOLID_BREP names from a STEP file by parsing the text."""
    names = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.match(
                r"#\d+\s*=\s*MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'\s*,",
                line.strip()
            )
            if match:
                names.append(match.group(1))
    return names


def _simplify_label(full_label: str) -> str:
    """Extract the short name from a STEP label.

    Strips the CST-style folder path (``folder/subfolder/...``) and the
    solid-type prefix (``type|``), returning just the final identifier.

    Examples
    --------
    >>> _simplify_label('HC_FPC_orientation_optimisation/hook_top|lh4')
    'lh4'
    >>> _simplify_label('component1|solid1')
    'solid1'
    >>> _simplify_label('beampipe')
    'beampipe'
    """
    # Take part after last '/'
    short = full_label.split('/')[-1]
    # Take part after last '|'
    if '|' in short:
        short = short.split('|')[-1]
    return short


def get_solid_centroid(occ_solid: TopoDS_Shape) -> Tuple[float, float, float]:
    """Compute the volume-weighted centroid of an OCC solid."""
    props = GProp_GProps()
    brepgprop.VolumeProperties(occ_solid, props)
    cog = props.CentreOfMass()
    return (cog.X(), cog.Y(), cog.Z())


# ==================== STEP PORT PARSING ====================

def parse_step_entities(filename: str) -> Tuple[List[dict], List[dict], List[dict]]:
    """Parse STEP file text for solid, port, and surface entities.

    Ports are identified by ``MANIFOLD_SURFACE_SHAPE_REPRESENTATION``
    entries whose name contains ``'Ports|'`` or ``'port'`` (case-insensitive).

    Returns
    -------
    solids, ports, others : list[dict]
        Each dict has ``entity_id`` (int) and ``name`` (str).
    """
    solids, ports, others = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r"#(\d+)\s*=\s*MANIFOLD_SOLID_BREP\s*\(\s*'([^']*)'\s*,", line)
            if m:
                solids.append({"entity_id": int(m.group(1)), "name": m.group(2)})
                continue
            m = re.match(
                r"#(\d+)\s*=\s*ADVANCED_BREP_SHAPE_REPRESENTATION\s*\(\s*'([^']*)'\s*,", line)
            if m:
                solids.append({"entity_id": int(m.group(1)), "name": m.group(2)})
                continue
            m = re.match(
                r"#(\d+)\s*=\s*MANIFOLD_SURFACE_SHAPE_REPRESENTATION\s*\(\s*'([^']*)'\s*,", line)
            if m:
                entity = {"entity_id": int(m.group(1)), "name": m.group(2)}
                if 'Ports|' in m.group(2) or 'port' in m.group(2).lower():
                    ports.append(entity)
                else:
                    others.append(entity)
                continue
    return solids, ports, others


# ==================== FACE PROPERTY HELPERS ====================

def get_face_properties(face: TopoDS_Shape) -> dict:
    """Compute geometric properties of a single OCC face.

    Returns dict with keys: center (gp_Pnt), normal (gp_Dir | None),
    area (float), is_planar (bool), face (TopoDS_Shape).
    """
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    adaptor = BRepAdaptor_Surface(topods.Face(face))
    is_planar = adaptor.GetType() == GeomAbs_Plane
    normal = adaptor.Plane().Axis().Direction() if is_planar else None
    return {
        "center": props.CentreOfMass(),
        "normal": normal,
        "area": props.Mass(),
        "is_planar": is_planar,
        "face": face,
    }


def get_port_shape_properties(port_shape: TopoDS_Shape) -> Optional[dict]:
    """Get the largest planar face from a port shell shape."""
    explorer = TopExp_Explorer(port_shape, TopAbs_FACE)
    faces = []
    while explorer.More():
        faces.append(get_face_properties(topods.Face(explorer.Current())))
        explorer.Next()
    planar = [f for f in faces if f["is_planar"]]
    if planar:
        return max(planar, key=lambda x: x["area"])
    return max(faces, key=lambda x: x["area"]) if faces else None


def faces_coincide(
        fp1: dict, fp2: dict,
        angle_tol: float = 0.05,
        distance_tol: Optional[float] = None,
        center_tol: Optional[float] = None,
) -> bool:
    """Check if two planar faces are coplanar and spatially overlapping.

    Parameters
    ----------
    fp1, fp2 : dict
        Face property dicts (from :func:`get_face_properties`).
    angle_tol : float
        Maximum angle (radians) between normals.
    distance_tol : float, optional
        Max perpendicular distance.  Default: 1% of smaller equivalent radius.
    center_tol : float, optional
        Max lateral distance between centres.  Default: 2x larger equiv radius.
    """
    if not fp1["is_planar"] or not fp2["is_planar"]:
        return False
    n1, n2 = fp1["normal"], fp2["normal"]
    c1, c2 = fp1["center"], fp2["center"]
    # Parallel normals (allow anti-parallel)
    if abs(n1.X()*n2.X() + n1.Y()*n2.Y() + n1.Z()*n2.Z()) < np.cos(angle_tol):
        return False
    dx, dy, dz = c2.X() - c1.X(), c2.Y() - c1.Y(), c2.Z() - c1.Z()
    plane_dist = abs(dx*n1.X() + dy*n1.Y() + dz*n1.Z())
    r1 = np.sqrt(fp1["area"] / np.pi)
    r2 = np.sqrt(fp2["area"] / np.pi)
    if plane_dist > (distance_tol or min(r1, r2) * 0.01):
        return False
    lat_dist = np.sqrt(max(dx**2 + dy**2 + dz**2 - plane_dist**2, 0))
    if lat_dist > (center_tol or max(r1, r2) * 2.0):
        return False
    return True


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


# ==================== OCC IMPORTER WITH VISUALIZATION ====================

class OCCImporter(BaseGeometry):
    """
    Import geometry from CAD files (STEP, BREP, IGES) with optional splitting.

    Uses PythonOCC for loading and splitting, then transfers directly
    to netgen.occ via ``From_PyOCC()`` for meshing — no temp files needed.

    Parameters
    ----------
    filepath : str
        Path to CAD file (STEP, BREP, or IGES format)
    unit : str
        Unit of the geometry file ('mm', 'cm', 'm')
    auto_build : bool
        If True, automatically build geometry and mesh on init.
        Set to False if you want to add splitting planes first.
    maxh : float
        Maximum mesh element size (only used if auto_build=True)

    Examples
    --------
    # Simple import without splitting
    >>> geo = OCCImporter("cavity.step", unit='mm', maxh=0.05)
    >>> geo.show('mesh')

    # Import with splitting (deferred build)
    >>> geo = OCCImporter("cavity.step", unit='mm', auto_build=False)
    >>> geo.add_splitting_plane((-0.1, -0.1, 0.05), (0.1, 0.1, 0.05))
    >>> geo.show_occ(show_planes=True)  # Preview before splitting
    >>> geo.split()
    >>> geo.show_occ()  # View split result
    >>> geo.finalize(maxh=0.05)
    >>> geo.show('mesh')

    # Import IGES file
    >>> geo = OCCImporter("cavity.iges", unit='mm', maxh=0.05)
    """

    UNIT_MAP = {'mm': 1000, 'cm': 100, 'm': 1}

    # Supported file extensions mapped to format identifiers
    FORMAT_MAP = {
        '.step': 'step', '.stp': 'step',
        '.brep': 'brep',
        '.iges': 'iges', '.igs': 'iges',
    }

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

        # STEP label tracking
        self._solid_labels: List[str] = []           # Labels from STEP BREP names
        self._original_solids_info: List[dict] = []  # Pre-split solid info for mapping
        self._materials: Dict[str, dict] = {}         # Material properties per label

        # Port detection
        self._port_entities: List[dict] = []          # Port entities from STEP text
        self._port_shapes: List[Tuple[TopoDS_Shape, str]] = []  # (shape, name) for port shells
        self._matched_ports: List[dict] = []          # Matched port results
        self._geometry_faces: List[dict] = []         # All face properties for inspection

        # Detect file format
        import os
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.FORMAT_MAP:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported formats: {', '.join(sorted(self.FORMAT_MAP.keys()))}"
            )
        self._format = self.FORMAT_MAP[ext]

        # Set source link for geometry linking
        self._source_link = str(filepath)

        # Load the raw OCC shape
        self._load_occ_shape()

        # Record import in history
        self._record('import_occ', filepath=str(filepath), unit=unit,
                     format=self._format, auto_build=auto_build, maxh=maxh)

        # Auto-build if requested
        if auto_build:
            self.build()
            self.generate_mesh(maxh=maxh)

    def _load_occ_shape(self) -> None:
        """Load CAD file into raw PythonOCC TopoDS_Shape.

        Supports STEP, BREP, and IGES formats.
        All geometries are converted to METERS for NGSolve.
        """
        if self.unit not in self.UNIT_MAP:
            raise ValueError(f"Unknown unit: {self.unit}. Use 'mm', 'cm', or 'm'.")

        # Scale factor to convert FROM input unit TO meters
        # mm → m: divide by 1000 (scale = 0.001)
        # cm → m: divide by 100 (scale = 0.01)
        # m → m: no change (scale = 1.0)
        scale_to_meters = 1.0 / self.UNIT_MAP[self.unit]

        if self._format == 'step':
            reader = STEPControl_Reader()
            status = reader.ReadFile(self.filepath)
            if status != IFSelect_RetDone:
                raise RuntimeError(f"Failed to read STEP file: {self.filepath}")
            reader.TransferRoots()
            shape = reader.OneShape()

            # Check for STEP labels — if present, all solids are intentional
            step_labels = extract_brep_names_from_step(self.filepath)
            has_labels = len(step_labels) > 0

            solids = get_solids(shape)
            if solids:
                if has_labels and len(step_labels) == len(solids):
                    # All solids have labels — skip ghost filtering, keep them all
                    significant_solids = solids
                else:
                    # No labels or mismatch — filter ghosts by volume
                    solid_volumes = []
                    for s in solids:
                        props = GProp_GProps()
                        brepgprop.VolumeProperties(s, props)
                        solid_volumes.append(props.Mass())

                    max_vol = max(solid_volumes) if solid_volumes else 0
                    significant_solids = [s for s, v in zip(solids, solid_volumes) if v > max_vol * 0.1]
                    if not significant_solids:
                        significant_solids = [solids[0]]

                if len(significant_solids) > 1:
                    builder = BRep_Builder()
                    comp = TopoDS_Compound()
                    builder.MakeCompound(comp)
                    for s in significant_solids:
                        builder.Add(comp, s)
                    self._occ_shape = comp
                elif significant_solids:
                    self._occ_shape = significant_solids[0]
                else:
                    self._occ_shape = solids[0]
            else:
                self._occ_shape = shape

            # Apply scaling if not already in meters
            if self.unit != 'm':
                trsf = gp_Trsf()
                trsf.SetScale(gp_Pnt(0, 0, 0), scale_to_meters)
                app = BRepBuilderAPI_Transform(trsf)
                app.Perform(self._occ_shape, True)
                self._occ_shape = app.Shape()

        elif self._format == 'iges':
            reader = IGESControl_Reader()
            status = reader.ReadFile(self.filepath)
            if status != IFSelect_RetDone:
                raise RuntimeError(f"Failed to read IGES file: {self.filepath}")
            reader.TransferRoots()
            shape = reader.OneShape()
            
            # Filter to keep only significant solids
            solids = get_solids(shape)
            if solids:
                solid_volumes = []
                for s in solids:
                    props = GProp_GProps()
                    brepgprop.VolumeProperties(s, props)
                    solid_volumes.append(props.Mass())
                
                max_vol = max(solid_volumes) if solid_volumes else 0
                significant_solids = [s for s, v in zip(solids, solid_volumes) if v > max_vol * 0.1]
                
                if len(significant_solids) > 1:
                    builder = BRep_Builder()
                    comp = TopoDS_Compound()
                    builder.MakeCompound(comp)
                    for s in significant_solids:
                        builder.Add(comp, s)
                    self._occ_shape = comp
                elif significant_solids:
                    self._occ_shape = significant_solids[0]
                else:
                    self._occ_shape = solids[0]
            else:
                self._occ_shape = shape

            # Apply scaling if not already in meters
            if self.unit != 'm':
                trsf = gp_Trsf()
                trsf.SetScale(gp_Pnt(0, 0, 0), scale_to_meters)
                app = BRepBuilderAPI_Transform(trsf)
                app.Perform(self._occ_shape, True)
                self._occ_shape = app.Shape()

        elif self._format == 'brep':
            shape = TopoDS_Shape()
            builder = BRep_Builder()
            success = breptools.Read(shape, self.filepath, builder)
            if not success:
                raise RuntimeError(f"Failed to read BREP file: {self.filepath}")
            
            # Filter to keep only significant solids
            solids = get_solids(shape)
            if solids:
                solid_volumes = []
                for s in solids:
                    props = GProp_GProps()
                    brepgprop.VolumeProperties(s, props)
                    solid_volumes.append(props.Mass())
                
                max_vol = max(solid_volumes) if solid_volumes else 0
                significant_solids = [s for s, v in zip(solids, solid_volumes) if v > max_vol * 0.1]
                
                if len(significant_solids) > 1:
                    comp = TopoDS_Compound()
                    builder.MakeCompound(comp)
                    for s in significant_solids:
                        builder.Add(comp, s)
                    self._occ_shape = comp
                elif significant_solids:
                    self._occ_shape = significant_solids[0]
                else:
                    self._occ_shape = solids[0]
            else:
                self._occ_shape = shape

            # Apply scaling if not already in meters
            if self.unit != 'm':
                trsf = gp_Trsf()
                trsf.SetScale(gp_Pnt(0, 0, 0), scale_to_meters)
                app = BRepBuilderAPI_Transform(trsf)
                app.Perform(self._occ_shape, True)
                self._occ_shape = app.Shape()

        if self._occ_shape is None or self._occ_shape.IsNull():
            raise RuntimeError(f"Failed to extract valid shape from {self.filepath}. Check if file is valid.")

        # Store the scale factor for reference
        self._scale_to_meters = scale_to_meters

        # Extract STEP labels and store per-solid info for split mapping
        if self._format == 'step':
            self._solid_labels = extract_brep_names_from_step(self.filepath)
            solids = get_solids(self._occ_shape)
            n_labels = len(self._solid_labels)
            n_solids = len(solids)

            if n_labels > 0 and n_labels != n_solids:
                warnings.warn(
                    f"STEP label count ({n_labels}) does not match solid count "
                    f"({n_solids}). Labels may be misaligned — ghost geometry "
                    f"filtering may have removed solids. Falling back to auto-naming."
                )
                self._solid_labels = []

            # Print labels (short names)
            if self._solid_labels:
                print(f"\nSTEP solids found:")
                for name in self._solid_labels:
                    print(f"  - {_simplify_label(name)}")
                print()

            # Store original solid info for split mapping
            self._original_solids_info = []
            for i, solid in enumerate(solids):
                centroid = get_solid_centroid(solid)
                bbox_min, bbox_max = get_shape_bounding_box(solid)
                label = self._solid_labels[i] if i < len(self._solid_labels) else f"solid_{i+1}"
                self._original_solids_info.append({
                    'label': label,
                    'centroid': centroid,
                    'bbox_min': bbox_min,
                    'bbox_max': bbox_max,
                })

            # Extract port entities and shell shapes
            _, port_entities, _ = parse_step_entities(self.filepath)
            self._port_entities = port_entities
            if port_entities:
                try:
                    from OCC.Extend.DataExchange import read_step_file_with_names_colors
                    # Suppress noisy prints from OCC library
                    import io, contextlib
                    with contextlib.redirect_stdout(io.StringIO()):
                        output_shapes = read_step_file_with_names_colors(self.filepath)
                    port_names = [p["name"] for p in port_entities]
                    shell_shapes = []
                    for shape, (name, color) in output_shapes.items():
                        if shape.ShapeType() == 3:  # TopAbs_SHELL
                            shell_shapes.append(shape)

                    self._port_shapes = []
                    for i, shell in enumerate(shell_shapes):
                        pname = port_names[i] if i < len(port_names) else f"port_{i+1}"
                        self._port_shapes.append((shell, pname))

                    # Apply same scaling to port shapes
                    if self.unit != 'm':
                        scaled = []
                        for pshape, pname in self._port_shapes:
                            trsf = gp_Trsf()
                            trsf.SetScale(gp_Pnt(0, 0, 0), scale_to_meters)
                            app = BRepBuilderAPI_Transform(trsf)
                            app.Perform(pshape, True)
                            scaled.append((app.Shape(), pname))
                        self._port_shapes = scaled

                    print(f"Ports found: {len(self._port_shapes)}")
                    print()
                except Exception as e:
                    warnings.warn(f"Could not extract port shapes: {e}")
                    self._port_shapes = []

    def add_splitting_plane(
            self,
            corner1: Tuple[float, float, float],
            corner2: Tuple[float, float, float],
            normal_axis: str = 'auto'
    ) -> 'OCCImporter':
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
        self : OCCImporter
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

        # Record in history
        self._record('add_splitting_plane',
                     corner1=list(corner1), corner2=list(corner2),
                     normal_axis=normal_axis)

        return self

    def add_splitting_plane_at_x(
            self,
            x: float,
            margin: float = 0.1
    ) -> 'OCCImporter':
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
        self : OCCImporter
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
    ) -> 'OCCImporter':
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
        self : OCCImporter
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
    ) -> 'OCCImporter':
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
        self : OCCImporter
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        pmin, pmax = get_shape_bounding_box(self._occ_shape)

        dx = (pmax[0] - pmin[0]) * margin
        dy = (pmax[1] - pmin[1]) * margin

        corner1 = (pmin[0] - dx, pmin[1] - dy, z)
        corner2 = (pmax[0] + dx, pmax[1] + dy, z)

        return self.add_splitting_plane(corner1, corner2, normal_axis='z')

    def split(self) -> 'OCCImporter':
        """
        Split geometry using added planes, then auto-rebuild.

        After splitting, :meth:`build` is called automatically so that
        solids, ports, and wall boundaries are named deterministically.
        Call :meth:`generate_mesh` next.

        Returns
        -------
        self : OCCImporter
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

        # Record in history
        self._record('split')

        # Auto-rebuild: build() will name solids, ports, and walls
        self.build()

        return self

    @staticmethod
    def _pyocc_to_netgen(occ_shape: TopoDS_Shape) -> OCCGeometry:
        """Transfer PythonOCC shape to netgen.occ via BREP.

        Uses OpenCASCADE's native binary BREP format for fast,
        lossless in-memory transfer between PythonOCC (SWIG) and
        netgen.occ (pybind11) bindings.

        Parameters
        ----------
        occ_shape : TopoDS_Shape
            PythonOCC shape to convert

        Returns
        -------
        OCCGeometry
            Netgen OCCGeometry wrapping the transferred shape
        """
        fd, tmp = tempfile.mkstemp(suffix='.brep')
        os.close(fd)
        try:
            breptools.Write(occ_shape, tmp)
            return OCCGeometry(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def build(self) -> None:
        """
        Build NGSolve geometry from OCC shape.

        Transfers the PythonOCC shape to netgen.occ via BREP format
        (fast, lossless binary transfer).

        If the geometry was split, solids are glued together for proper
        mesh connectivity.

        After building:
        - All faces are named ``'default'``
        - Solids are named using STEP labels if available, otherwise
          ``cell_1``, ``cell_2``, etc.
        - Port faces are named ``port1``, ``port2``, etc.
        - ``self.bc`` is set to ``'default'``
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        # Transfer from PythonOCC to netgen.occ via BREP
        occ_geo = self._pyocc_to_netgen(self._occ_shape)

        if self._is_split and hasattr(occ_geo, 'solids') and len(occ_geo.solids) > 1:
            # Glue solids together for mesh connectivity
            self.geo = Glue([solid for solid in occ_geo.solids])
        elif hasattr(occ_geo, 'solids') and len(occ_geo.solids) > 1:
            # Multi-solid import (not split) — still need Glue for connectivity
            self.geo = Glue([solid for solid in occ_geo.solids])
        else:
            # Single solid or not split - extract the shape
            self.geo = occ_geo.shape

        # --- Deterministic boundary naming ---
        # Step 1: Name ALL faces 'default' (PEC)
        self._name_all_faces_wall()

        # Step 2: Name solids (using STEP labels or cell_N)
        self._auto_name_solids()

        # Step 3: Detect and assign ports
        if self._port_shapes:
            self._auto_detect_ports()
        else:
            self._auto_assign_ports_by_position()

        # Step 4: Set boundary condition
        self.bc = 'default'
        self._bc_explicitly_set = True

        # Step 5: Subtract PEC solids if materials were set before build
        if self._materials:
            pec_labels = [k for k, v in self._materials.items() if v == 'PEC']
            if pec_labels:
                self._subtract_pec_solids(pec_labels)

    def finalize(self, maxh: Optional[float] = None) -> 'OCCImporter':
        """
        Build geometry and generate mesh.

        Convenience method for deferred building workflow.

        Parameters
        ----------
        maxh : float, optional
            Maximum mesh element size. Uses value from constructor if not specified.

        Returns
        -------
        self : OCCImporter
            Returns self for method chaining
        """
        if maxh is not None:
            self.maxh = maxh

        self.build()
        self.generate_mesh(maxh=self.maxh)

        self._record('finalize', maxh=self.maxh)
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
            return
        elif what in ("geometry", "geo"):
            if self.geo is None:
                raise ValueError("Geometry not built. Call build() first.")
            scene = NetgenDraw(self.geo, **kwargs)
        elif what == "mesh":
            if self.mesh is None:
                raise ValueError("Mesh not generated. Call generate_mesh() first.")
            scene = NetgenDraw(self.mesh, **kwargs)
        else:
            raise ValueError(
                f"Invalid option '{what}'. Use 'geometry', 'geo', 'mesh', or 'occ'."
            )

        _display_webgui_fallback(scene)

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

        # Transfer from PythonOCC to netgen.occ via BREP
        occ_geo = self._pyocc_to_netgen(self._occ_shape)

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
    ) -> 'OCCImporter':
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
        self : OCCImporter
        """
        if self.geo is None:
            self.build()

        # Step 1: Name all faces 'wall' first
        self._name_all_faces_wall()

        # Step 2: Name solids
        self._auto_name_solids(
            sort_axis=sort_axis,
            naming_func=naming_func,
            print_info=print_info,
        )

        # Step 3: Assign ports
        if port_axis is None:
            port_axis = sort_axis
        self._auto_assign_ports_by_position(
            port_axis=port_axis,
            port_prefix=port_prefix,
        )

        # Step 4: Set boundary condition
        self.bc = 'default'
        self._bc_explicitly_set = True

        self._record('name_solids', sort_axis=sort_axis, port_axis=port_axis,
                     port_prefix=port_prefix, print_info=print_info)

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
                    face.name = 'default'
        except AttributeError:
            # Single OCC shape without .solids
            try:
                for face in self.geo.faces:
                    face.name = 'default'
            except AttributeError:
                pass

    def _map_sub_solid_to_parent(self, sub_solid_centroid: Tuple[float, float, float]) -> str:
        """Map a sub-solid's centroid to its parent label using bounding box containment.

        Parameters
        ----------
        sub_solid_centroid : tuple
            (x, y, z) centroid of the sub-solid after splitting.

        Returns
        -------
        str
            The label of the parent solid that contains this sub-solid.
        """
        cx, cy, cz = sub_solid_centroid
        best_label = None
        best_dist = float('inf')

        for info in self._original_solids_info:
            bmin = info['bbox_min']
            bmax = info['bbox_max']
            # Check containment with small tolerance
            tol = 1e-8
            if (bmin[0] - tol <= cx <= bmax[0] + tol and
                bmin[1] - tol <= cy <= bmax[1] + tol and
                bmin[2] - tol <= cz <= bmax[2] + tol):
                # Inside this parent's bounding box — pick closest centroid
                pc = info['centroid']
                dist = ((cx - pc[0])**2 + (cy - pc[1])**2 + (cz - pc[2])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_label = info['label']

        if best_label is None:
            # Fallback: find nearest parent centroid
            for info in self._original_solids_info:
                pc = info['centroid']
                dist = ((cx - pc[0])**2 + (cy - pc[1])**2 + (cz - pc[2])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_label = info['label']

        return best_label

    def _auto_name_solids(
            self,
            sort_axis: str = 'Z',
            naming_func=None,
            print_info: bool = True,
    ) -> None:
        """Name solids using STEP labels or a naming function.

        Solids are sorted by centroid along *sort_axis* for deterministic
        ordering.  STEP labels are used when available and *naming_func*
        is ``None``.
        """
        if self.geo is None:
            return

        has_parent_info = bool(self._original_solids_info) and naming_func is None

        if naming_func is None and not has_parent_info:
            naming_func = lambda i, s: f"cell_{i + 1}"

        axis_index = {'X': 0, 'Y': 1, 'Z': 2}
        sort_idx = axis_index.get(sort_axis.upper(), 2)

        try:
            solids = list(self.geo.solids)
            n_solids = len(solids)

            if n_solids <= 1:
                solid = solids[0] if n_solids == 1 else self.geo
                if has_parent_info:
                    bb = solid.bounding_box
                    pmin, pmax = bb
                    centroid = ((pmin[0]+pmax[0])/2, (pmin[1]+pmax[1])/2, (pmin[2]+pmax[2])/2)
                    mat_name = _simplify_label(self._map_sub_solid_to_parent(centroid))
                elif naming_func is not None:
                    mat_name = naming_func(0, solid)
                else:
                    mat_name = "cell_1"
                solid.mat(mat_name)
                if print_info:
                    print(f"Solid: '{mat_name}'")
                return

            # Sort solids by centroid
            def _centroid(solid):
                bb = solid.bounding_box
                return (bb[0][sort_idx] + bb[1][sort_idx]) / 2

            solids_sorted = sorted(enumerate(solids), key=lambda x: _centroid(x[1]))

            # Determine material names
            if has_parent_info:
                raw_names = []
                label_counts = {}
                for orig_idx, solid in solids_sorted:
                    bb = solid.bounding_box
                    centroid = tuple((bb[0][j]+bb[1][j])/2 for j in range(3))
                    parent = self._map_sub_solid_to_parent(centroid)
                    short = _simplify_label(parent)
                    label_counts[short] = label_counts.get(short, 0) + 1
                    raw_names.append(short)

                seen = {}
                needs_suffix = {k for k, v in label_counts.items() if v > 1}
                mat_names = []
                for name in raw_names:
                    if name in needs_suffix:
                        seen[name] = seen.get(name, 0) + 1
                        mat_names.append(f"{name}_{seen[name]}")
                    else:
                        mat_names.append(name)
            else:
                mat_names = [naming_func(i, s) if naming_func else f"cell_{i+1}"
                             for i, (_, s) in enumerate(solids_sorted)]

            for new_idx, (orig_idx, solid) in enumerate(solids_sorted):
                solid.mat(mat_names[new_idx])

            if print_info:
                print(f"Named {n_solids} solids:")
                for i, name in enumerate(mat_names):
                    print(f"  solid {i}: '{name}'")

        except AttributeError:
            if has_parent_info:
                mat_name = _simplify_label(self._original_solids_info[0]['label'])
            elif naming_func is not None:
                mat_name = naming_func(0, self.geo)
            else:
                mat_name = "cell_1"
            self.geo.mat(mat_name)
            if print_info:
                print(f"Solid: '{mat_name}'")

    # ==================== PORT DETECTION ====================

    def _collect_occ_geometry_faces(self) -> List[dict]:
        """Enumerate all faces on the raw OCC compound and compute properties.

        Populates ``self._geometry_faces`` — a list of dicts, each with
        ``index``, ``center``, ``normal``, ``area``, ``is_planar``, ``face``.
        """
        if self._occ_shape is None:
            return []

        faces = []
        explorer = TopExp_Explorer(self._occ_shape, TopAbs_FACE)
        idx = 0
        while explorer.More():
            fp = get_face_properties(topods.Face(explorer.Current()))
            fp["index"] = idx
            faces.append(fp)
            explorer.Next()
            idx += 1

        self._geometry_faces = faces
        return faces

    def _auto_detect_ports(self) -> None:
        """Match STEP-defined port shells to netgen geometry faces.

        For each port shell shape, the largest planar face is extracted and
        matched against all planar faces on the geometry using
        :func:`faces_coincide`.  Matched faces are renamed to the port name
        found in the STEP file (e.g. ``Ports|port1`` → ``port1``).
        """
        if not self._port_shapes or self.geo is None:
            return

        # Collect OCC-level face properties for matching
        if not self._geometry_faces:
            self._collect_occ_geometry_faces()

        self._matched_ports = []
        used_face_indices = set()

        for port_shape, raw_port_name in self._port_shapes:
            port_fp = get_port_shape_properties(port_shape)
            if port_fp is None or not port_fp["is_planar"]:
                warnings.warn(f"Port '{raw_port_name}': not planar or has no faces — skipped")
                continue

            # Find best matching geometry face
            pc = port_fp["center"]
            best_match = None
            best_score = float('inf')

            for gf in self._geometry_faces:
                if gf["index"] in used_face_indices:
                    continue
                if not gf["is_planar"]:
                    continue
                if faces_coincide(port_fp, gf):
                    dx = gf["center"].X() - pc.X()
                    dy = gf["center"].Y() - pc.Y()
                    dz = gf["center"].Z() - pc.Z()
                    score = np.sqrt(dx**2 + dy**2 + dz**2)
                    if score < best_score:
                        best_score = score
                        best_match = gf

            if best_match is None:
                warnings.warn(f"Port '{raw_port_name}': no matching geometry face found")
                continue

            used_face_indices.add(best_match["index"])

            # Derive clean port name: "Ports|port1" -> "port1"
            clean_name = raw_port_name
            if '|' in clean_name:
                clean_name = clean_name.split('|', 1)[1]

            n = best_match["normal"]
            mc = best_match["center"]
            self._matched_ports.append({
                "name": clean_name,
                "raw_name": raw_port_name,
                "face_index": best_match["index"],
                "center": (mc.X(), mc.Y(), mc.Z()),
                "normal": (n.X(), n.Y(), n.Z()),
                "area": best_match["area"],
            })

        # Now apply port names to netgen geometry
        self._apply_matched_ports_to_netgen()

        # Collect unmatched ports (internal ports needing splitting)
        matched_raw = {p["raw_name"] for p in self._matched_ports}
        unmatched = [(s, n) for s, n in self._port_shapes if n not in matched_raw]

        # Print summary
        n_matched = len(self._matched_ports)
        n_total = len(self._port_shapes)
        print(f"Ports matched: {n_matched}/{n_total}")
        for p in self._matched_ports:
            cx, cy, cz = p["center"]
            nx, ny, nz = p["normal"]
            print(f"  {p['name']:20s} center=({cx:.5f}, {cy:.5f}, {cz:.5f})  "
                  f"normal=({nx:.4f}, {ny:.4f}, {nz:.4f})")

        if unmatched:
            print(f"\nUnmatched ports ({len(unmatched)}) — these are internal ports "
                  f"that require splitting:")
            for port_shape, raw_name in unmatched:
                fp = get_port_shape_properties(port_shape)
                if fp and fp["is_planar"]:
                    c = fp["center"]
                    n = fp["normal"]
                    # Determine axis from normal
                    nx, ny, nz = abs(n.X()), abs(n.Y()), abs(n.Z())
                    if nz > nx and nz > ny:
                        axis_hint = f"add_splitting_plane_at_z({c.Z():.6f})"
                    elif nx > ny:
                        axis_hint = f"add_splitting_plane_at_x({c.X():.6f})"
                    else:
                        axis_hint = f"add_splitting_plane_at_y({c.Y():.6f})"
                    clean = raw_name.split('|', 1)[1] if '|' in raw_name else raw_name
                    print(f"  {clean:20s} -> geo.{axis_hint}")
                else:
                    print(f"  {raw_name}: not planar")

    def _apply_matched_ports_to_netgen(self) -> None:
        """Apply matched port names to netgen geometry faces.

        For each matched port, find the netgen face whose centroid and
        normal match the OCC-level match, and rename it.
        """
        if not self._matched_ports or self.geo is None:
            return

        # Collect all netgen faces with their properties
        netgen_faces = []
        try:
            solids = list(self.geo.solids)
            for solid in solids:
                for face in solid.faces:
                    try:
                        bb = face.bounding_box
                        fc = tuple((bb[0][j] + bb[1][j]) / 2 for j in range(3))
                        netgen_faces.append({"face": face, "center": fc})
                    except Exception:
                        continue
        except AttributeError:
            for face in self.geo.faces:
                try:
                    bb = face.bounding_box
                    fc = tuple((bb[0][j] + bb[1][j]) / 2 for j in range(3))
                    netgen_faces.append({"face": face, "center": fc})
                except Exception:
                    continue

        for port in self._matched_ports:
            pc = port["center"]
            best_face = None
            best_dist = float('inf')

            for nf in netgen_faces:
                fc = nf["center"]
                dist = sum((pc[j] - fc[j])**2 for j in range(3))
                if dist < best_dist:
                    best_dist = dist
                    best_face = nf["face"]

            if best_face is not None:
                best_face.name = port["name"]
                best_face.col = (1, 0, 0)

    def _auto_assign_ports_by_position(
            self,
            port_axis: str = 'Z',
            port_prefix: str = 'port',
    ) -> None:
        """Fallback port assignment using Min/Max faces along an axis.

        Used when no STEP port definitions are found.  For multi-solid
        geometry, ports are assigned sequentially: ``port1`` at the
        global minimum, ``portN+1`` at the global maximum, with internal
        ports at solid boundaries.
        """
        if self.geo is None:
            return

        axis_map = {'X': X, 'Y': Y, 'Z': Z}
        axis_index = {'X': 0, 'Y': 1, 'Z': 2}

        ax = axis_map.get(port_axis.upper(), Z)
        sort_idx = axis_index.get(port_axis.upper(), 2)

        try:
            solids = list(self.geo.solids)
            n_solids = len(solids)

            if n_solids <= 1:
                solid = solids[0] if n_solids == 1 else self.geo
                solid.faces.Min(ax).name = f'{port_prefix}1'
                solid.faces.Min(ax).col = (1, 0, 0)
                solid.faces.Max(ax).name = f'{port_prefix}2'
                solid.faces.Max(ax).col = (1, 0, 0)
                print(f"Ports: {port_prefix}1, {port_prefix}2 (by position)")
                return

            # Sort solids by centroid
            def _centroid(solid):
                bb = solid.bounding_box
                return (bb[0][sort_idx] + bb[1][sort_idx]) / 2

            solids_sorted = sorted(solids, key=_centroid)

            for i, solid in enumerate(solids_sorted):
                solid.faces.Min(ax).name = f'{port_prefix}{i + 1}'
                solid.faces.Min(ax).col = (1, 0, 0)
                solid.faces.Max(ax).name = f'{port_prefix}{i + 2}'
                solid.faces.Max(ax).col = (1, 0, 0)

            print(f"Ports: {n_solids + 1} ports assigned by position "
                  f"({port_prefix}1 ... {port_prefix}{n_solids + 1})")

        except AttributeError:
            self.geo.faces.Min(ax).name = f'{port_prefix}1'
            self.geo.faces.Min(ax).col = (1, 0, 0)
            self.geo.faces.Max(ax).name = f'{port_prefix}2'
            self.geo.faces.Max(ax).col = (1, 0, 0)
            print(f"Ports: {port_prefix}1, {port_prefix}2 (by position)")

    # ==================== PORT INSPECTION & MANUAL ASSIGNMENT ====================

    def list_planar_faces(self) -> List[dict]:
        """List all planar faces on the OCC geometry with their properties.

        Useful for inspecting the geometry before manual port assignment.

        Returns
        -------
        list[dict]
            Each dict contains ``index``, ``center`` (tuple), ``normal`` (tuple),
            ``area`` (float).
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        if not self._geometry_faces:
            self._collect_occ_geometry_faces()

        planar = [f for f in self._geometry_faces if f["is_planar"]]

        print(f"\nPlanar faces: {len(planar)} / {len(self._geometry_faces)} total\n")
        print(f"{'ID':<6} {'Center':<42} {'Normal':<35} {'Area':<12}")
        print("-" * 100)

        result = []
        for pf in planar:
            c, n = pf["center"], pf["normal"]
            cx, cy, cz = c.X(), c.Y(), c.Z()
            nx, ny, nz = n.X(), n.Y(), n.Z()
            print(f"{pf['index']:<6} ({cx:>10.5f}, {cy:>10.5f}, {cz:>10.5f})   "
                  f"({nx:>7.4f}, {ny:>7.4f}, {nz:>7.4f})   {pf['area']:<12.6f}")
            result.append({
                "index": pf["index"],
                "center": (cx, cy, cz),
                "normal": (nx, ny, nz),
                "area": pf["area"],
            })
        return result

    def assign_ports(
            self,
            port_face_map: Dict[str, int],
    ) -> 'OCCImporter':
        """Manually assign ports by mapping port names to OCC face indices.

        Use :meth:`list_planar_faces` or :meth:`show_planar_faces` to
        identify the face indices first.

        Parameters
        ----------
        port_face_map : dict
            ``{port_name: face_index}`` — e.g. ``{"port1": 42, "port2": 107}``.

        Returns
        -------
        self : OCCImporter

        Examples
        --------
        >>> geo.list_planar_faces()        # inspect
        >>> geo.assign_ports({"port1": 42, "port2": 107})
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")
        if self.geo is None:
            raise RuntimeError("Geometry not built. Call build() first.")

        if not self._geometry_faces:
            self._collect_occ_geometry_faces()

        face_by_index = {f["index"]: f for f in self._geometry_faces}

        self._matched_ports = []
        for port_name, face_idx in port_face_map.items():
            if face_idx not in face_by_index:
                warnings.warn(f"Face index {face_idx} not found — skipping port '{port_name}'")
                continue

            gf = face_by_index[face_idx]
            c = gf["center"]
            n = gf["normal"]
            self._matched_ports.append({
                "name": port_name,
                "raw_name": port_name,
                "face_index": face_idx,
                "center": (c.X(), c.Y(), c.Z()),
                "normal": (n.X(), n.Y(), n.Z()) if n else (0, 0, 0),
                "area": gf["area"],
            })
            print(f"  {port_name} -> Face #{face_idx} "
                  f"at ({c.X():.4f}, {c.Y():.4f}, {c.Z():.4f})")

        # Apply to netgen
        self._apply_matched_ports_to_netgen()

        self._record('assign_ports', port_face_map=port_face_map)
        return self

    # ==================== VISUAL INSPECTION ====================

    def show_planar_faces(
            self,
            highlight_ports: bool = True,
            backend: str = 'auto',
            **kwargs,
    ):
        """Display the geometry with planar faces highlighted.

        Parameters
        ----------
        highlight_ports : bool
            If True, highlight matched port faces in red.
        backend : str
            ``'jupyter'`` for JupyterRenderer, ``'occ'`` for the standalone
            OCC viewer (SimpleGui), ``'auto'`` tries Jupyter first.
        """
        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        if not self._geometry_faces:
            self._collect_occ_geometry_faces()

        port_indices = {p["face_index"] for p in self._matched_ports}

        if backend == 'auto':
            try:
                get_ipython  # noqa: F821 — available in Jupyter
                backend = 'jupyter'
            except NameError:
                backend = 'occ'

        if backend == 'jupyter':
            self._show_planar_faces_jupyter(port_indices, highlight_ports, **kwargs)
        else:
            self._show_planar_faces_occ(port_indices, highlight_ports, **kwargs)

    def _show_planar_faces_jupyter(self, port_indices, highlight_ports, **kwargs):
        """Jupyter-based planar face viewer."""
        rnd = JupyterRenderer()

        # Display geometry with transparency
        solids = get_solids(self._occ_shape)
        for solid in solids:
            rnd.DisplayShape(solid, render_edges=True, **kwargs)

        # Highlight planar faces
        for gf in self._geometry_faces:
            if not gf["is_planar"]:
                continue
            if highlight_ports and gf["index"] in port_indices:
                rnd.DisplayShape(gf["face"], render_edges=True)
            # Non-port planar faces shown as-is (part of geometry)

        rnd.Display()

    def _show_planar_faces_occ(self, port_indices, highlight_ports, **kwargs):
        """Standalone OCC viewer for planar face inspection."""
        try:
            from OCC.Display.SimpleGui import init_display
            from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
        except ImportError:
            warnings.warn(
                "OCC.Display.SimpleGui not available. "
                "Use backend='jupyter' or install pythonocc-core with GUI support."
            )
            return

        display, start_display, _, _ = init_display()

        # Display geometry
        solids = get_solids(self._occ_shape)
        colors = generate_distinct_colors(len(solids))
        for i, solid in enumerate(solids):
            r, g, b = colors[i]
            display.DisplayShape(
                solid,
                color=Quantity_Color(r, g, b, Quantity_TOC_RGB),
                transparency=0.5,
                update=False,
            )

        # Highlight port faces in red
        if highlight_ports:
            for gf in self._geometry_faces:
                if gf["index"] in port_indices:
                    display.DisplayShape(
                        gf["face"],
                        color=Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB),
                        update=False,
                    )

        display.FitAll()
        start_display()

    def name_faces_by_position(
            self,
            axis: str = 'Z',
            port_prefix: str = 'port'
    ) -> 'OCCImporter':
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
        self : OCCImporter
        """
        if self.geo is None:
            raise ValueError("Geometry not built. Call build() first.")

        axis_map = {'X': X, 'Y': Y, 'Z': Z}
        if axis.upper() not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Use 'X', 'Y', or 'Z'.")

        ax = axis_map[axis.upper()]

        try:
            self.geo.faces.Min(ax).name = f"{port_prefix}1"
            self.geo.faces.Min(ax).col = (1, 0, 0)
            self.geo.faces.Max(ax).name = f"{port_prefix}2"
            self.geo.faces.Max(ax).col = (1, 0, 0)
        except Exception as e:
            print(f"Warning: Could not name faces: {e}")

        return self

    def color_solids(self) -> 'OCCImporter':
        """
        Apply distinct colors to each solid in the geometry.

        Returns
        -------
        self : OCCImporter
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

    # ==================== MATERIAL ASSIGNMENT ====================

    def set_materials(self, material_config: Dict[str, dict]) -> 'OCCImporter':
        """
        Assign material properties to solids by their STEP label or material name.

        The keys in *material_config* must match the solid labels (from the STEP
        file) or the material names assigned during build/split.  Unmatched keys
        produce a warning.

        A value of ``'PEC'`` marks the solid as a perfect electric conductor.
        PEC solids are subtracted from the computational domain and their
        exposed surfaces become Dirichlet boundaries.  This triggers an
        automatic geometry/mesh rebuild.

        Parameters
        ----------
        material_config : dict
            Mapping of solid label → material properties dict or ``'PEC'``.
            Each properties dict can contain::

                {
                    "eps_r": float,       # relative permittivity (default 1.0)
                    "epsilon_r": float,   # accepted as alias for eps_r
                    "mu_r": float,        # relative permeability (default 1.0)
                    "sigma": float,       # electrical conductivity S/m (default 0)
                    "tan_delta": float,   # loss tangent (default 0)
                }

        Returns
        -------
        self : OCCImporter

        Examples
        --------
        >>> geo.set_materials({
        ...     "hook_top": "PEC",
        ...     "ceramic": {"eps_r": 9.4, "tan_delta": 1e-4},
        ...     "beampipe": {"eps_r": 1.0},
        ... })
        """
        # Normalise: accept 'epsilon_r' as alias for 'eps_r'
        normalised = {}
        for key, val in material_config.items():
            if isinstance(val, str) and val.upper() == 'PEC':
                normalised[key] = 'PEC'
            elif isinstance(val, dict):
                d = dict(val)
                if 'epsilon_r' in d and 'eps_r' not in d:
                    d['eps_r'] = d.pop('epsilon_r')
                normalised[key] = d
            else:
                normalised[key] = val

        # Validate material keys against known solid names
        import fnmatch
        known_names = set()
        if self.mesh is not None:
            known_names = set(self.mesh.GetMaterials())
        elif self.geo is not None:
            try:
                for solid in self.geo.solids:
                    known_names.add(solid.name)
            except AttributeError:
                pass
        # Add all STEP label variants (full path + segments)
        for label in self._solid_labels:
            known_names.add(label)
            short = label.split('/')[-1]
            known_names.add(short)
            known_names.update(short.split('|'))

        for key in normalised:
            if '*' in key:
                matched = any(fnmatch.fnmatchcase(kn, key) for kn in known_names)
            else:
                matched = key in known_names

            if not matched and known_names:
                warnings.warn(
                    f"Material key '{key}' does not match any known solid label. "
                    f"Known labels: {sorted(known_names)}"
                )

        self._materials = normalised

        # Check if any PEC solids need to be subtracted
        pec_labels = [k for k, v in normalised.items() if v == 'PEC']
        if pec_labels and self.geo is not None:
            self._subtract_pec_solids(pec_labels)

        self._record('set_materials', material_config=material_config)

        n_pec = len(pec_labels)
        n_other = len(normalised) - n_pec
        parts = []
        if n_other:
            parts.append(f"{n_other} dielectric")
        if n_pec:
            parts.append(f"{n_pec} PEC")
        print(f"Material properties assigned: {', '.join(parts)}")
        return self

    def _get_all_names_for_solid(self, solid_name: str) -> List[str]:
        """Return all name variants for a solid (simplified + full STEP label parts).

        This lets material keys reference a solid by its simplified mesh name,
        the full STEP path, or any segment of it (e.g. ``hook_top|lh4``).
        """
        names = {solid_name}
        # Strip auto-generated suffix (_1, _2) to find the base
        base = solid_name.rsplit('_', 1)[0] if '_' in solid_name else solid_name
        # Look up the original STEP label for this solid
        for info in self._original_solids_info:
            full_label = info['label']
            if _simplify_label(full_label) == base:
                names.add(full_label)
                # Add intermediate segments: "hook_top|lh4", "hook_top", "lh4"
                short = full_label.split('/')[-1]
                names.add(short)
                names.update(short.split('|'))
                break
        return list(names)

    def _resolve_material_key(self, mat_key: str, solid_name: str) -> bool:
        """Check if a material config key matches a solid name.

        Matching rules:
        - **Exact**: key equals the solid name or any of its STEP label variants.
        - **Wildcard** (``*``): glob-style matching via ``fnmatch``.

        No fuzzy substring matching — ``'solid'`` does NOT match ``'solid1'``.
        """
        import fnmatch

        candidates = self._get_all_names_for_solid(solid_name)

        if '*' in mat_key:
            return any(fnmatch.fnmatchcase(c, mat_key) for c in candidates)

        return mat_key in candidates

    def _subtract_pec_solids(self, pec_labels: List[str]) -> None:
        """Remove PEC solids from the geometry and name exposed faces as BC.

        PEC solids are identified by matching *pec_labels* against the
        netgen solid names.  The remaining (non-PEC) solids are re-glued
        and the faces that were shared with PEC solids automatically
        become external boundary faces named ``self.bc`` (the Dirichlet BC).
        """
        if self.geo is None:
            return

        try:
            all_solids = list(self.geo.solids)
        except AttributeError:
            warnings.warn("Cannot subtract PEC solids from single-solid geometry")
            return

        keep = []
        removed_names = []
        for solid in all_solids:
            is_pec = False
            for pec_key in pec_labels:
                if self._resolve_material_key(pec_key, solid.name):
                    is_pec = True
                    break
            if is_pec:
                removed_names.append(solid.name)
            else:
                keep.append(solid)

        if not removed_names:
            warnings.warn("No PEC solids matched — geometry unchanged")
            return

        if not keep:
            raise ValueError("All solids are PEC — no computational domain remains")

        print(f"Subtracting {len(removed_names)} PEC solid(s): {removed_names}")

        # Rebuild geometry from non-PEC solids
        if len(keep) == 1:
            self.geo = keep[0]
        else:
            self.geo = Glue(keep)

        # Name all faces as PEC boundary (exposed interfaces get this automatically)
        bc_name = self.bc or 'default'
        try:
            for solid in self.geo.solids:
                for face in solid.faces:
                    if face.name == 'default' or face.name == '':
                        face.name = bc_name
        except AttributeError:
            for face in self.geo.faces:
                if face.name == 'default' or face.name == '':
                    face.name = bc_name

        # Invalidate mesh — must re-mesh after PEC subtraction
        self.mesh = None
        self._ports = None
        self._boundaries = None

    def get_material(self, domain_name: str) -> dict:
        """
        Get material properties for a domain, resolving label matching.

        Returns default vacuum properties if no material is assigned.
        PEC domains (already subtracted) should not appear in the mesh,
        but if queried they return ``{"PEC": True}``.

        Parameters
        ----------
        domain_name : str
            The material/domain name as it appears in the mesh.

        Returns
        -------
        dict
            Material properties with defaults filled in::

                {"eps_r": 1.0, "mu_r": 1.0, "sigma": 0.0, "tan_delta": 0.0}
        """
        import fnmatch

        defaults = {"eps_r": 1.0, "mu_r": 1.0, "sigma": 0.0, "tan_delta": 0.0}

        def _apply(raw):
            if raw == 'PEC':
                return {"PEC": True}
            props = dict(defaults)
            if isinstance(raw, dict):
                props.update(raw)
            return props

        candidates = self._get_all_names_for_solid(domain_name)

        # 1. Exact match (non-wildcard keys only)
        for key, mat in self._materials.items():
            if '*' not in key and key in candidates:
                return _apply(mat)

        # 2. Wildcard match
        for key, mat in self._materials.items():
            if '*' in key and any(fnmatch.fnmatchcase(c, key) for c in candidates):
                return _apply(mat)

        return defaults

    @property
    def solid_labels(self) -> List[str]:
        """STEP solid labels extracted from the file."""
        return list(self._solid_labels)

    @property
    def materials(self) -> Dict[str, dict]:
        """Currently assigned material properties."""
        return dict(self._materials)

    # ==================== PARTS INVENTORY ====================

    def save_parts_inventory(
            self,
            save_dir: Optional[str] = None,
            ncols: int = 4,
            figsize_per_cell: Tuple[float, float] = (4, 4),
            deflection: float = 0.1,
            elev: float = 30,
            azim: float = 45,
    ) -> str:
        """
        Generate and save a parts inventory image showing each solid with its label.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save the image. Defaults to ``geometry/`` next to the
            STEP file.
        ncols : int
            Number of columns in the grid.
        figsize_per_cell : tuple
            (width, height) per subplot cell in inches.
        deflection : float
            Mesh deflection for triangulation quality.
        elev, azim : float
            3D view angles.

        Returns
        -------
        str
            Path to the saved image file.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TopLoc import TopLoc_Location
        except ImportError as e:
            warnings.warn(f"Cannot generate parts inventory: {e}")
            return ""

        if self._occ_shape is None:
            raise RuntimeError("No OCC shape loaded.")

        solids = get_solids(self._occ_shape)
        n = len(solids)
        if n == 0:
            warnings.warn("No solids found for parts inventory.")
            return ""

        # Determine labels
        labels = []
        for i in range(n):
            if i < len(self._solid_labels):
                labels.append(self._solid_labels[i])
            else:
                labels.append(f"solid_{i+1}")

        nrows = int(np.ceil(n / ncols))
        fig = plt.figure(figsize=(figsize_per_cell[0] * ncols,
                                   figsize_per_cell[1] * nrows))

        colors = generate_distinct_colors(n)

        for idx, solid in enumerate(solids):
            # Triangulate
            mesh_inc = BRepMesh_IncrementalMesh(solid, deflection, False, 0.5, True)
            mesh_inc.Perform()

            vertices = []
            triangles = []
            vertex_offset = 0

            explorer = TopExp_Explorer(solid, TopAbs_FACE)
            while explorer.More():
                face = explorer.Current()
                loc = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, loc)

                if triangulation is not None:
                    trsf = loc.Transformation()
                    for i in range(1, triangulation.NbNodes() + 1):
                        node = triangulation.Node(i)
                        node.Transform(trsf)
                        vertices.append([node.X(), node.Y(), node.Z()])
                    for i in range(1, triangulation.NbTriangles() + 1):
                        tri = triangulation.Triangle(i)
                        n1, n2, n3 = tri.Get()
                        triangles.append([
                            n1 - 1 + vertex_offset,
                            n2 - 1 + vertex_offset,
                            n3 - 1 + vertex_offset
                        ])
                    vertex_offset += triangulation.NbNodes()

                explorer.Next()

            ax_plot = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')

            if vertices and triangles:
                verts = np.array(vertices)
                tris = np.array(triangles)

                center = (verts.max(axis=0) + verts.min(axis=0)) / 2
                verts_c = verts - center
                extent = (verts.max(axis=0) - verts.min(axis=0)).max()
                if extent > 0:
                    verts_c = verts_c / extent

                tri_verts = verts_c[tris]
                poly = Poly3DCollection(tri_verts, alpha=0.85)
                poly.set_facecolor(colors[idx])
                poly.set_edgecolor((0.2, 0.2, 0.2, 0.3))
                poly.set_linewidth(0.1)
                ax_plot.add_collection3d(poly)
                ax_plot.set_xlim(-0.6, 0.6)
                ax_plot.set_ylim(-0.6, 0.6)
                ax_plot.set_zlim(-0.6, 0.6)
            else:
                ax_plot.text2D(0.5, 0.5, "No mesh", ha='center', va='center',
                              transform=ax_plot.transAxes)

            ax_plot.set_box_aspect([1, 1, 1])
            ax_plot.set_axis_off()
            ax_plot.view_init(elev=elev, azim=azim)

            # Title
            parts = labels[idx].split('/')
            short_name = parts[-1]
            path = '/'.join(parts[:-1])
            if len(path) > 30:
                path = '...' + path[-27:]
            ax_plot.set_title(f"[{idx}] {short_name}\n{path}",
                              fontsize=8, fontweight='bold', pad=2)

        # Hide empty subplots
        for idx in range(n, nrows * ncols):
            fig.add_subplot(nrows, ncols, idx + 1).set_visible(False)

        basename = os.path.splitext(os.path.basename(self.filepath))[0]
        plt.suptitle(f"Parts Inventory: {basename} — {n} solids",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Determine save path
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(self.filepath), 'geometry')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{basename}_parts_inventory.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"Parts inventory saved to: {save_path}")
        return save_path

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

    # === Persistence ===

    def save_geometry(self, project_path) -> None:
        """
        Save OCCImporter geometry to project.

        Copies the source CAD file and writes the history so that
        the geometry can be reconstructed on load.
        """
        from pathlib import Path as _Path
        project_path = _Path(project_path)
        geo_dir = project_path / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)

        # Copy source STEP file
        source_hash = None
        source_filename = None
        src = _Path(self.filepath)
        if src.exists():
            source_filename = f'source_model{src.suffix}'
            dest = geo_dir / source_filename
            import shutil
            if src.absolute() != dest.absolute():
                shutil.copy2(str(src), str(dest))
            source_hash = self._file_hash(dest)
            self._source_hash = source_hash

        # Build history JSON — rewrite filepath to project-relative
        history_for_save = []
        for entry in self._history:
            e = dict(entry)
            if e.get('op') == 'import_occ' and source_filename:
                e['filepath'] = f'geometry/{source_filename}'
            history_for_save.append(e)

        meta = {
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'source_link': str(self._source_link) if self._source_link else None,
            'source_filename': source_filename,
            'source_hash': source_hash,
            'history': history_for_save,
        }

        import json
        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def _rebuild_from_history(cls, history, project_path, source_file=None):
        """
        Reconstruct an OCCImporter by replaying its operation history.
        """
        from pathlib import Path as _Path
        project_path = _Path(project_path)

        geo = None
        built = False
        for entry in history:
            op = entry['op']

            if op in ('import_occ', 'import_step'):
                # Resolve filepath: use project-local copy first, then history path
                filepath = str(source_file) if source_file else entry['filepath']
                # If filepath is project-relative, resolve it
                fp = _Path(filepath)
                if not fp.is_absolute():
                    fp = project_path / fp
                geo = cls(
                    filepath=str(fp),
                    unit=entry.get('unit', 'mm'),
                    auto_build=False,
                    maxh=entry.get('maxh'),
                )

            elif op == 'build' and geo is not None:
                geo.build()
                built = True

            elif op == 'add_splitting_plane' and geo is not None:
                geo.add_splitting_plane(
                    corner1=tuple(entry['corner1']),
                    corner2=tuple(entry['corner2']),
                    normal_axis=entry.get('normal_axis', 'auto'),
                )

            elif op == 'split' and geo is not None:
                geo.split()

            elif op == 'finalize' and geo is not None:
                geo.finalize(maxh=entry.get('maxh'))

            elif op == 'name_solids' and geo is not None:
                geo.name_solids(
                    sort_axis=entry.get('sort_axis', 'Z'),
                    port_axis=entry.get('port_axis'),
                    port_prefix=entry.get('port_prefix', 'port'),
                    print_info=entry.get('print_info', False),
                )

            elif op == 'generate_mesh' and geo is not None:
                # Ensure build before mesh
                if not built:
                    geo.build()
                    built = True
                geo.generate_mesh(
                    maxh=entry.get('maxh'),
                    curve_order=entry.get('curve_order', 3),
                )

        if geo is None:
            raise ValueError("History does not contain an 'import_occ' or 'import_step' operation.")

        # Always ensure the geometry is built for assembly use
        if not built:
            geo.build()

        return geo



# Backward-compatible alias
STEPImporter = OCCImporter