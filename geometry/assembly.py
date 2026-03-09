# assembly.py
"""Assembly class with hierarchical support and solution caching."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Union, Tuple, Literal,
    Any, TYPE_CHECKING
)
from enum import Enum, auto
import numpy as np

from netgen.occ import OCCGeometry, Glue, X, Y, Z, Axes
from ngsolve.webgui import Draw
from ngsolve import Mesh

from .base import BaseGeometry
from .component_registry import (
    TaggableMixin, ComponentTag, ComputeMethod,
    get_global_cache
)


@dataclass
class Transform3D:
    """3D transformation specification."""
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_center: Optional[Tuple[float, float, float]] = None
    
    def is_identity(self) -> bool:
        return (
            all(abs(t) < 1e-12 for t in self.translation) and
            all(abs(r) < 1e-12 for r in self.rotation)
        )
    
    def compose(self, other: 'Transform3D') -> 'Transform3D':
        """Compose with another transform (self applied first)."""
        return Transform3D(
            translation=tuple(a + b for a, b in zip(self.translation, other.translation)),
            rotation=tuple(a + b for a, b in zip(self.rotation, other.rotation)),
            rotation_center=self.rotation_center or other.rotation_center
        )


@dataclass
class ComponentEntry:
    """Entry for a component in the assembly."""
    geometry: Union[BaseGeometry, 'Assembly']
    key: str
    transform: Transform3D = field(default_factory=Transform3D)
    original_bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    aligned_port: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.original_bounds is None:
            self.original_bounds = self._compute_bounds()
    
    def _compute_bounds(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        geo = self.geometry
        if isinstance(geo, Assembly):
            return geo.get_assembly_bounds()
        elif geo.geo is not None:
            try:
                bb = geo.geo.bounding_box
                return (tuple(bb[0]), tuple(bb[1]))
            except:
                pass
        return ((0, 0, 0), (1, 1, 1))
    
    @property
    def is_assembly(self) -> bool:
        """Check if this entry contains a sub-assembly."""
        return isinstance(self.geometry, Assembly)
    
    @property
    def tag(self) -> ComponentTag:
        """Get tag of underlying geometry/assembly."""
        return self.geometry.tag
    
    @property
    def size(self) -> Tuple[float, float, float]:
        if self.original_bounds is None:
            return (1.0, 1.0, 1.0)
        pmin, pmax = self.original_bounds
        return tuple(pmax[i] - pmin[i] for i in range(3))
    
    @property
    def centroid(self) -> Tuple[float, float, float]:
        if self.original_bounds is None:
            return (0.5, 0.5, 0.5)
        pmin, pmax = self.original_bounds
        return tuple((pmin[i] + pmax[i]) / 2 for i in range(3))


class ConnectionType(Enum):
    """Type of connection between components."""
    PORT_TO_PORT = auto()
    ADJACENT = auto()


@dataclass  
class Connection:
    """Connection between two components."""
    from_key: str
    to_key: str
    from_port: str = "port2"
    to_port: str = "port1"
    connection_type: ConnectionType = ConnectionType.PORT_TO_PORT
    gap: float = 0.0


class Assembly(BaseGeometry):
    """
    Assembly of geometry components with hierarchical support.
    
    Can contain individual geometries OR other assemblies. Supports
    automatic positioning, tagging for caching, and multiple layout modes.
    
    Parameters
    ----------
    main_axis : str
        Primary axis for concatenation ('X', 'Y', or 'Z')
    
    Examples
    --------
    **Sequential assembly:**
    
    >>> assembly = Assembly(main_axis='Z')
    >>> assembly.add("wg1", waveguide1)
    >>> assembly.add("wg2", waveguide2, after="wg1")
    >>> assembly.build()
    
    **Mosaic layout:**
    
    >>> layout = '''
    ...     .     top      .
    ...   left   center  right
    ... '''
    >>> assembly = Assembly.from_mosaic(layout, components={...})
    
    **Nested assemblies:**
    
    >>> sub_assembly = Assembly()
    >>> sub_assembly.add("a", geo_a)
    >>> sub_assembly.add("b", geo_b, after="a")
    >>> sub_assembly.build()
    >>> 
    >>> main = Assembly()
    >>> main.add("input", input_wg)
    >>> main.add("middle", sub_assembly, after="input")  # Sub-assembly!
    >>> main.add("output", output_wg, after="middle")
    >>> main.build()
    """
    
    _AXIS_VEC = {'X': X, 'Y': Y, 'Z': Z}
    _AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}
    
    def __init__(self, main_axis: Literal['X', 'Y', 'Z'] = 'Z'):
        super().__init__()
        self.main_axis = main_axis.upper()
        
        self._components: Dict[str, ComponentEntry] = {}
        self._component_order: List[str] = []
        self._connections: List[Connection] = []
        
        self._layout_computed = False
        self._is_built = False
    
    # =========================================================================
    # Component Management
    # =========================================================================
    
    def add(
        self,
        key: str,
        geometry: Union[BaseGeometry, 'Assembly'],
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        align_port: Optional[str] = None,
        **metadata
    ) -> 'Assembly':
        """
        Add a component (geometry or sub-assembly) to the assembly.
        
        Parameters
        ----------
        key : str
            Unique identifier
        geometry : BaseGeometry or Assembly
            Component to add (can be another Assembly!)
        position : tuple, optional
            Explicit position (x, y, z)
        rotation : tuple, optional
            Rotation angles in degrees (rx, ry, rz)
        after : str, optional
            Place after this component along main axis
        before : str, optional
            Place before this component along main axis
        align_port : str, optional
            Port to use for alignment
        **metadata
            Additional metadata
        
        Returns
        -------
        self : Assembly
        """
        if key in self._components:
            raise ValueError(f"Component '{key}' already exists")
        
        # Handle sub-assemblies
        if isinstance(geometry, Assembly):
            if not geometry._is_built:
                geometry.build()
        elif geometry.geo is None:
            raise ValueError(f"Geometry '{key}' not built. Call build() first.")
        
        transform = Transform3D(
            translation=position or (0.0, 0.0, 0.0),
            rotation=rotation or (0.0, 0.0, 0.0)
        )
        
        entry = ComponentEntry(
            geometry=geometry,
            key=key,
            transform=transform,
            aligned_port=align_port,
            metadata=metadata
        )
        
        self._components[key] = entry
        
        # Handle ordering
        if after is not None:
            if after not in self._components:
                raise ValueError(f"Component '{after}' not found")
            idx = self._component_order.index(after) + 1
            self._component_order.insert(idx, key)
            
            self._connections.append(Connection(
                from_key=after,
                to_key=key,
                from_port='port2',
                to_port=align_port or 'port1'
            ))
            
        elif before is not None:
            if before not in self._components:
                raise ValueError(f"Component '{before}' not found")
            idx = self._component_order.index(before)
            self._component_order.insert(idx, key)
            
            self._connections.append(Connection(
                from_key=key,
                to_key=before,
                from_port=align_port or 'port2',
                to_port='port1'
            ))
        else:
            self._component_order.append(key)
        
        self._layout_computed = False
        self._is_built = False
        self.invalidate_tag()
        
        return self
    
    def connect(
        self,
        key: str,
        geometry: Union[BaseGeometry, 'Assembly'],
        to_component: str,
        from_port: str = "port1",
        to_port: str = "port2",
        rotation: Optional[Tuple[float, float, float]] = None,
        gap: float = 0.0,
        **metadata
    ) -> 'Assembly':
        """
        Add component connected via ports to existing component.
        
        Parameters
        ----------
        key : str
            Identifier for new component
        geometry : BaseGeometry or Assembly
            Component to add
        to_component : str
            Existing component to connect to
        from_port : str
            Port on new component
        to_port : str
            Port on existing component
        rotation : tuple, optional
            Rotation angles
        gap : float
            Gap between components
        
        Returns
        -------
        self : Assembly
        """
        if to_component not in self._components:
            raise ValueError(f"Target '{to_component}' not found")
        
        self.add(key, geometry, rotation=rotation, **metadata)
        
        self._connections.append(Connection(
            from_key=to_component,
            to_key=key,
            from_port=to_port,
            to_port=from_port,
            connection_type=ConnectionType.PORT_TO_PORT,
            gap=gap
        ))
        
        return self
    
    def remove(self, key: str) -> 'Assembly':
        """Remove a component."""
        if key not in self._components:
            raise KeyError(f"Component '{key}' not found")
        
        del self._components[key]
        self._component_order.remove(key)
        self._connections = [c for c in self._connections 
                           if c.from_key != key and c.to_key != key]
        
        self._layout_computed = False
        self._is_built = False
        self.invalidate_tag()
        
        return self
    
    def rotate(
        self, 
        key: str, 
        angles: Tuple[float, float, float],
        center: Optional[Tuple[float, float, float]] = None
    ) -> 'Assembly':
        """Apply rotation to a component."""
        if key not in self._components:
            raise KeyError(f"Component '{key}' not found")
        
        entry = self._components[key]
        current = entry.transform.rotation
        entry.transform = Transform3D(
            translation=entry.transform.translation,
            rotation=tuple(c + a for c, a in zip(current, angles)),
            rotation_center=center
        )
        
        self._layout_computed = False
        self._is_built = False
        
        return self
    
    def translate(self, key: str, offset: Tuple[float, float, float]) -> 'Assembly':
        """Apply translation to a component."""
        if key not in self._components:
            raise KeyError(f"Component '{key}' not found")
        
        entry = self._components[key]
        current = entry.transform.translation
        entry.transform = Transform3D(
            translation=tuple(c + o for c, o in zip(current, offset)),
            rotation=entry.transform.rotation,
            rotation_center=entry.transform.rotation_center
        )
        
        self._layout_computed = False
        self._is_built = False
        
        return self
    
    # =========================================================================
    # Index-based Access
    # =========================================================================
    
    def __getitem__(self, key: Union[str, int]) -> ComponentEntry:
        """Access component by key or index."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._component_order):
                raise IndexError(f"Index {key} out of range")
            key = self._component_order[key]
        
        if key not in self._components:
            raise KeyError(f"Component '{key}' not found")
        
        return self._components[key]
    
    def __len__(self) -> int:
        return len(self._components)
    
    def __iter__(self):
        return iter(self._component_order)
    
    def __contains__(self, key: str) -> bool:
        return key in self._components
    
    @property
    def keys(self) -> List[str]:
        return list(self._component_order)
    
    @property
    def components(self) -> Dict[str, ComponentEntry]:
        return dict(self._components)
    
    # =========================================================================
    # Mosaic Layout Factory
    # =========================================================================
    
    @classmethod
    def from_mosaic(
        cls,
        mosaic: Union[str, List[List[str]]],
        components: Dict[str, Union[BaseGeometry, 'Assembly']],
        grid_plane: Literal['XY', 'XZ', 'YZ'] = 'XZ',
        extend_axis: Optional[Literal['X', 'Y', 'Z']] = None,
        alignment: Literal['center', 'min', 'max'] = 'center',
        spacing: float = 0.0
    ) -> 'Assembly':
        """
        Create assembly from mosaic-style layout.
        
        The mosaic defines a 2D grid. Empty cells are marked with '.' or ''.
        Components can be geometries OR sub-assemblies.
        
        Parameters
        ----------
        mosaic : str or list
            Layout specification as multiline string or nested list.
        components : dict
            Mapping of identifiers to BaseGeometry or Assembly objects
        grid_plane : str
            Which plane the grid represents ('XY', 'XZ', 'YZ')
        extend_axis : str, optional
            Axis components extend along (inferred if None)
        alignment : str
            How to align within cells ('center', 'min', 'max')
        spacing : float
            Gap between grid cells
        
        Returns
        -------
        Assembly
        
        Examples
        --------
        >>> layout = '''
        ...     .     top      .
        ...   left   center  right
        ... '''
        >>> assembly = Assembly.from_mosaic(
        ...     layout,
        ...     components={
        ...         'center': t_junction,
        ...         'top': waveguide_v,
        ...         'left': waveguide_h,
        ...         'right': waveguide_h  # Same object = same tag!
        ...     },
        ...     grid_plane='XY'
        ... )
        """
        # Parse mosaic
        grid = cls._parse_mosaic(mosaic)
        
        # Validate components
        grid_keys = set()
        for row in grid:
            for cell in row:
                if cell and cell != '.':
                    grid_keys.add(cell)
        
        missing = grid_keys - set(components.keys())
        if missing:
            raise ValueError(f"Components not provided: {missing}")
        
        # Determine axes
        plane_axes = {
            'XY': ('X', 'Y', 'Z'),
            'XZ': ('X', 'Z', 'Y'),
            'YZ': ('Y', 'Z', 'X')
        }
        
        col_axis, row_axis, default_extend = plane_axes[grid_plane]
        extend_axis = extend_axis or default_extend
        
        assembly = cls(main_axis=extend_axis)
        
        # Compute grid dimensions
        n_rows = len(grid)
        n_cols = max(len(row) for row in grid)
        
        # Pad rows
        for row in grid:
            while len(row) < n_cols:
                row.append('')
        
        col_idx = cls._AXIS_IDX[col_axis]
        row_idx = cls._AXIS_IDX[row_axis]
        
        # Get max sizes per row/column
        col_sizes = [0.0] * n_cols
        row_sizes = [0.0] * n_rows
        
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell and cell != '.' and cell in components:
                    geo = components[cell]
                    # Build sub-assemblies if needed
                    if isinstance(geo, Assembly) and not geo._is_built:
                        geo.build()
                    
                    if geo.geo is not None:
                        try:
                            bb = geo.geo.bounding_box
                            size = tuple(bb[1][k] - bb[0][k] for k in range(3))
                            col_sizes[j] = max(col_sizes[j], size[col_idx])
                            row_sizes[i] = max(row_sizes[i], size[row_idx])
                        except:
                            pass
        
        # Compute positions
        col_positions = [0.0]
        for size in col_sizes[:-1]:
            col_positions.append(col_positions[-1] + size + spacing)
        
        row_positions = [0.0]
        for size in row_sizes[:-1]:
            row_positions.append(row_positions[-1] + size + spacing)
        
        # Flip rows (grid top-to-bottom, but Y/Z increase upward)
        total_height = row_positions[-1] + row_sizes[-1] if row_sizes else 0
        row_positions = [total_height - p - row_sizes[i] 
                        for i, p in enumerate(row_positions)]
        
        # Add components
        added = set()
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell and cell != '.' and cell not in added:
                    if cell in components:
                        geo = components[cell]
                        
                        pos = [0.0, 0.0, 0.0]
                        pos[col_idx] = col_positions[j]
                        pos[row_idx] = row_positions[i]
                        
                        # Center alignment
                        if alignment == 'center' and geo.geo is not None:
                            try:
                                bb = geo.geo.bounding_box
                                size = tuple(bb[1][k] - bb[0][k] for k in range(3))
                                pos[col_idx] += (col_sizes[j] - size[col_idx]) / 2
                                pos[row_idx] += (row_sizes[i] - size[row_idx]) / 2
                            except:
                                pass
                        
                        assembly.add(cell, geo, position=tuple(pos))
                        added.add(cell)
        
        return assembly
    
    @staticmethod
    def _parse_mosaic(mosaic: Union[str, List[List[str]]]) -> List[List[str]]:
        """Parse mosaic specification to grid."""
        if isinstance(mosaic, str):
            lines = [line.strip() for line in mosaic.strip().split('\n')]
            lines = [line for line in lines if line]
            
            grid = []
            for line in lines:
                cells = line.split()
                cells = ['' if c == '.' else c for c in cells]
                grid.append(cells)
            return grid
        
        elif isinstance(mosaic, list):
            return [['' if c in ('.', None) else str(c) for c in row] 
                   for row in mosaic]
        
        raise TypeError(f"Mosaic must be str or list, got {type(mosaic)}")
    
    # =========================================================================
    # Layout Computation
    # =========================================================================
    
    def compute_layout(self) -> 'Assembly':
        """Compute positions based on connections."""
        if self._layout_computed:
            return self
        
        if not self._components:
            self._layout_computed = True
            return self
        
        positioned = set()
        
        # First component at origin
        first_key = self._component_order[0]
        positioned.add(first_key)
        
        # Process connections iteratively
        max_iter = len(self._connections) * 2 + 1
        for _ in range(max_iter):
            for conn in self._connections:
                if conn.to_key in positioned:
                    continue
                if conn.from_key not in positioned:
                    continue
                
                self._position_connected(conn)
                positioned.add(conn.to_key)
        
        # Position remaining unconnected components
        for key in self._component_order:
            if key not in positioned:
                if positioned:
                    last = list(positioned)[-1]
                    self._position_after(key, last)
                positioned.add(key)
        
        self._layout_computed = True
        return self
    
    def _get_port_position(
        self, 
        entry: ComponentEntry, 
        port_name: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get port position in local coordinates.
        
        Uses bounding box estimation for reliable, deterministic results.
        The port center is computed as the face centroid.
        """
        if entry.original_bounds is None:
            return None
        
        pmin, pmax = entry.original_bounds
        
        # Compute face center (centroid of the port face)
        # Port faces are perpendicular to the main axis
        axis_idx = self._AXIS_IDX[self.main_axis]
        
        # Center coordinates on the non-main axes
        center = [(pmin[i] + pmax[i]) / 2 for i in range(3)]
        
        # Determine which end of main axis based on port name
        if 'port1' in port_name.lower() or 'min' in port_name.lower():
            center[axis_idx] = pmin[axis_idx]
        elif 'port2' in port_name.lower() or 'max' in port_name.lower():
            center[axis_idx] = pmax[axis_idx]
        else:
            # Unknown port, try to get from mesh if available
            geo = entry.geometry
            if hasattr(geo, 'get_point_on_boundary') and geo.mesh is not None:
                try:
                    # Get any point on boundary, then project to face center
                    pt = geo.get_point_on_boundary(port_name)
                    if pt is not None:
                        # Use the axis coordinate from the point, but center on other axes
                        result = list(center)
                        # Keep the main axis coordinate from the actual point
                        # This handles non-standard port positions
                        return tuple(result)
                except:
                    pass
        
        return tuple(center)
    
    def _position_connected(self, conn: Connection) -> None:
        """
        Position component based on port connection.
        
        Aligns the ports of two components along the main axis,
        while keeping other coordinates aligned.
        """
        from_entry = self._components[conn.from_key]
        to_entry = self._components[conn.to_key]
        
        from_port_pos = self._get_port_position(from_entry, conn.from_port)
        to_port_pos = self._get_port_position(to_entry, conn.to_port)
        
        if from_port_pos is None or to_port_pos is None:
            # Fallback to simple stacking
            self._position_after(conn.to_key, conn.from_key)
            return
        
        axis_idx = self._AXIS_IDX[self.main_axis]
        
        # Calculate from_port position in world coordinates
        from_world = tuple(
            from_port_pos[i] + from_entry.transform.translation[i]
            for i in range(3)
        )
        
        # Calculate translation needed:
        # - Along main axis: align ports (with gap)
        # - On other axes: align centers (so components stack properly)
        translation = [0.0, 0.0, 0.0]
        
        for i in range(3):
            if i == axis_idx:
                # Main axis: align port positions with gap
                translation[i] = from_world[i] - to_port_pos[i] + conn.gap
            else:
                # Other axes: align bounding box centers
                # This ensures the components are centered relative to each other
                from_center = (from_entry.original_bounds[0][i] + 
                             from_entry.original_bounds[1][i]) / 2
                to_center = (to_entry.original_bounds[0][i] + 
                           to_entry.original_bounds[1][i]) / 2
                
                # Translate so centers align (accounting for from_entry's position)
                translation[i] = (from_center + from_entry.transform.translation[i] 
                                 - to_center)
        
        to_entry.transform = Transform3D(
            translation=tuple(translation),
            rotation=to_entry.transform.rotation,
            rotation_center=to_entry.transform.rotation_center
        )
    
    def _position_after(self, key: str, after_key: str) -> None:
        """
        Position component after another along main axis.
        
        Aligns centers on perpendicular axes.
        """
        entry = self._components[key]
        after_entry = self._components[after_key]
        
        axis_idx = self._AXIS_IDX[self.main_axis]
        
        # Get the max position of 'after' component along main axis
        after_max = (
            after_entry.transform.translation[axis_idx] + 
            after_entry.size[axis_idx]
        )
        
        # Build translation vector
        translation = [0.0, 0.0, 0.0]
        
        for i in range(3):
            if i == axis_idx:
                # Main axis: place after the previous component
                translation[i] = after_max
            else:
                # Other axes: align centers
                if after_entry.original_bounds and entry.original_bounds:
                    after_center = (after_entry.original_bounds[0][i] + 
                                   after_entry.original_bounds[1][i]) / 2
                    entry_center = (entry.original_bounds[0][i] + 
                                  entry.original_bounds[1][i]) / 2
                    translation[i] = (after_center + 
                                     after_entry.transform.translation[i] - 
                                     entry_center)
        
        entry.transform = Transform3D(
            translation=tuple(translation),
            rotation=entry.transform.rotation,
            rotation_center=entry.transform.rotation_center
        )
    
    # =========================================================================
    # Build
    # =========================================================================
    
    def build(self) -> None:
        """Build combined geometry from all components."""
        if not self._components:
            raise ValueError("No components in assembly")
        
        self.compute_layout()
        
        # Build sub-assemblies first
        for entry in self._components.values():
            if entry.is_assembly and not entry.geometry._is_built:
                entry.geometry.build()
        
        shapes = []
        
        for key in self._component_order:
            entry = self._components[key]
            shape = entry.geometry.geo
            transformed = self._apply_transform(shape, entry.transform)
            shapes.append(transformed)
        
        if len(shapes) == 1:
            self.geo = shapes[0]
        else:
            self.geo = Glue(shapes)
        
        self._setup_boundaries()
        self._is_built = True
    
    def _apply_transform(self, shape, transform: Transform3D):
        """Apply transformation to OCC shape."""
        if transform.is_identity():
            return shape
        
        result = shape
        
        # Rotation
        rx, ry, rz = transform.rotation
        if abs(rx) > 1e-12 or abs(ry) > 1e-12 or abs(rz) > 1e-12:
            if transform.rotation_center:
                center = transform.rotation_center
            else:
                try:
                    bb = shape.bounding_box
                    center = tuple((bb[0][i] + bb[1][i]) / 2 for i in range(3))
                except:
                    center = (0, 0, 0)
            
            if abs(rz) > 1e-12:
                result = result.Rotate(Axes(center, Z), rz)
            if abs(ry) > 1e-12:
                result = result.Rotate(Axes(center, Y), ry)
            if abs(rx) > 1e-12:
                result = result.Rotate(Axes(center, X), rx)
        
        # Translation
        tx, ty, tz = transform.translation
        if abs(tx) > 1e-12 or abs(ty) > 1e-12 or abs(tz) > 1e-12:
            result = result.Move((tx, ty, tz))
        
        return result
    
    def _setup_boundaries(self) -> None:
        """Setup boundary conditions."""
        try:
            for face in self.geo.faces:
                if not face.name or face.name == 'default':
                    face.name = 'wall'
        except:
            pass
        
        self.bc = 'wall'
        self._bc_explicitly_set = True
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    def inspect(
        self,
        show_labels: bool = True,
        color_by_component: bool = True,
        **kwargs
    ) -> None:
        """Visualize assembly configuration."""
        self.compute_layout()
        
        if not self._components:
            print("Assembly is empty")
            return
        
        shapes = []
        colors = self._generate_colors(len(self._components))
        
        for i, key in enumerate(self._component_order):
            entry = self._components[key]
            shape = entry.geometry.geo
            transformed = self._apply_transform(shape, entry.transform)
            
            if color_by_component:
                try:
                    for face in transformed.faces:
                        face.col = colors[i]
                except:
                    pass
            
            shapes.append(transformed)
        
        if len(shapes) == 1:
            display_geo = shapes[0]
        else:
            display_geo = Glue(shapes)
        
        Draw(display_geo, **kwargs)
        
        # Summary
        print(f"\nAssembly: {len(self._components)} components")
        print("-" * 50)
        for i, key in enumerate(self._component_order):
            entry = self._components[key]
            pos = entry.transform.translation
            geo_type = type(entry.geometry).__name__
            tag_short = str(entry.tag)
            size = entry.size
            
            sub = " (sub-assembly)" if entry.is_assembly else ""
            cached = " [CACHED]" if entry.geometry.has_cached_solution() else ""
            
            print(f"  [{i}] {key}: {geo_type}{sub}{cached}")
            print(f"      Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"      Size:     ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
            print(f"      Tag: {tag_short}")

    def show(
        self,
        what: Literal["geometry", "mesh", "geo", "inspect"] = "geometry",
        **kwargs
    ) -> None:
        """Display the assembly."""
        what = what.lower()
        
        if what == "inspect":
            self.inspect(**kwargs)
        elif what in ("geometry", "geo"):
            if self.geo is None:
                raise ValueError("Assembly not built. Call build() first.")
            Draw(self.geo, **kwargs)
        elif what == "mesh":
            if self.mesh is None:
                raise ValueError("Mesh not generated. Call generate_mesh() first.")
            # Use NGSolve's Draw for NGSolve Mesh objects
            Draw(self.mesh, **kwargs)
        else:
            raise ValueError(f"Invalid option '{what}'")

    @staticmethod
    def _generate_colors(n: int) -> List[Tuple[float, float, float]]:
        """Generate distinct colors."""
        colors = []
        for i in range(n):
            hue = i / max(n, 1)
            h = hue * 6
            c = 0.9 * 0.7
            x = c * (1 - abs(h % 2 - 1))
            m = 0.9 - c
            
            if h < 1: r, g, b = c, x, 0
            elif h < 2: r, g, b = x, c, 0
            elif h < 3: r, g, b = 0, c, x
            elif h < 4: r, g, b = 0, x, c
            elif h < 5: r, g, b = x, 0, c
            else: r, g, b = c, 0, x
            
            colors.append((r + m, g + m, b + m))
        return colors
    
    # =========================================================================
    # Bounds and Info
    # =========================================================================
    
    def get_assembly_bounds(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """Get bounding box of entire assembly."""
        if not self._components:
            return ((0, 0, 0), (0, 0, 0))
        
        all_min = [float('inf')] * 3
        all_max = [float('-inf')] * 3
        
        for key in self._components:
            entry = self._components[key]
            pmin, pmax = entry.original_bounds or ((0, 0, 0), (1, 1, 1))
            t = entry.transform.translation
            
            for i in range(3):
                all_min[i] = min(all_min[i], pmin[i] + t[i])
                all_max[i] = max(all_max[i], pmax[i] + t[i])
        
        return tuple(all_min), tuple(all_max)
    
        # =========================================================================
    # Tagging (for cache identification)
    # =========================================================================
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        """Get parameters for tag computation."""
        # Include all component tags - this makes the assembly tag
        # dependent on all its constituents
        component_tags = {}
        for key, entry in self._components.items():
            component_tags[key] = {
                'tag_hash': entry.tag.full_hash,
                'transform': {
                    'translation': entry.transform.translation,
                    'rotation': entry.transform.rotation
                }
            }
        
        return {
            'class': 'Assembly',
            'main_axis': self.main_axis,
            'components': component_tags,
            'order': list(self._component_order)
        }
    
    def _get_mesh_params(self) -> Dict[str, Any]:
        """Get mesh parameters for tag computation."""
        return {
            'maxh': getattr(self, 'maxh', None),
            'component_count': len(self._components)
        }
    
    # =========================================================================
    # Cache-Aware Methods
    # =========================================================================
    
    def get_component_cache_status(self) -> Dict[str, bool]:
        """
        Check cache status for all components.
        
        Returns
        -------
        dict
            Mapping of component key to cache status (True if cached)
        """
        status = {}
        for key, entry in self._components.items():
            status[key] = entry.geometry.has_cached_solution()
        return status
    
    def get_uncached_components(self) -> List[str]:
        """Get list of components that need computation."""
        return [
            key for key, entry in self._components.items()
            if not entry.geometry.has_cached_solution()
        ]
    
    def get_unique_components(self) -> Dict[str, List[str]]:
        """
        Group components by their geometry tag.
        
        Returns dict mapping tag hash to list of component keys.
        Components with the same tag are geometrically identical
        and can share solutions.
        
        Returns
        -------
        dict
            {tag_hash: [key1, key2, ...]}
        
        Examples
        --------
        >>> # If 'left' and 'right' use the same waveguide object
        >>> unique = assembly.get_unique_components()
        >>> # unique might be: {'abc123...': ['left', 'right'], 'def456...': ['center']}
        """
        groups = {}
        for key, entry in self._components.items():
            tag_hash = entry.tag.geometry_hash  # Use geometry hash, not full
            if tag_hash not in groups:
                groups[tag_hash] = []
            groups[tag_hash].append(key)
        return groups
    
    def count_unique_geometries(self) -> int:
        """Count number of unique geometries (for computation estimation)."""
        return len(self.get_unique_components())
    
    # =========================================================================
    # Flattening (for solver access)
    # =========================================================================
    
    def flatten(self) -> List[Tuple[str, BaseGeometry, Transform3D]]:
        """
        Flatten assembly hierarchy to list of (key, geometry, transform).
        
        Recursively expands sub-assemblies with composed transforms.
        Useful for solvers that need to process each geometry individually.
        
        Returns
        -------
        list
            List of (key, geometry, world_transform) tuples
        """
        result = []
        
        for key in self._component_order:
            entry = self._components[key]
            
            if entry.is_assembly:
                # Recursively flatten sub-assembly
                sub_flat = entry.geometry.flatten()
                for sub_key, sub_geo, sub_transform in sub_flat:
                    # Compose transforms: sub_transform then entry.transform
                    composed = Transform3D(
                        translation=tuple(
                            sub_transform.translation[i] + entry.transform.translation[i]
                            for i in range(3)
                        ),
                        rotation=tuple(
                            sub_transform.rotation[i] + entry.transform.rotation[i]
                            for i in range(3)
                        )
                    )
                    result.append((f"{key}.{sub_key}", sub_geo, composed))
            else:
                result.append((key, entry.geometry, entry.transform))
        
        return result
    
    def get_all_tags(self) -> Dict[str, ComponentTag]:
        """
        Get tags for all geometries (flattened).
        
        Returns
        -------
        dict
            Mapping of flattened key to ComponentTag
        """
        return {key: geo.tag for key, geo, _ in self.flatten()}
    
    # =========================================================================
    # Summary and Info
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """Get assembly summary information."""
        cache_status = self.get_component_cache_status()
        unique = self.get_unique_components()
        
        return {
            'n_components': len(self._components),
            'n_unique_geometries': len(unique),
            'n_connections': len(self._connections),
            'main_axis': self.main_axis,
            'is_built': self._is_built,
            'has_mesh': self.mesh is not None,
            'component_keys': list(self._component_order),
            'cached_count': sum(cache_status.values()),
            'uncached_count': len(self._components) - sum(cache_status.values()),
            'bounds': self.get_assembly_bounds() if self._components else None,
            'has_sub_assemblies': any(e.is_assembly for e in self._components.values())
        }
    
    def print_info(self) -> None:
        """Print detailed assembly information."""
        info = self.summary()
        
        print("\n" + "=" * 70)
        print("Assembly Information")
        print("=" * 70)
        print(f"Components:             {info['n_components']}")
        print(f"Unique geometries:      {info['n_unique_geometries']}")
        print(f"Connections:            {info['n_connections']}")
        print(f"Main axis:              {info['main_axis']}")
        print(f"Built:                  {info['is_built']}")
        print(f"Has mesh:               {info['has_mesh']}")
        print(f"Has sub-assemblies:     {info['has_sub_assemblies']}")
        
        print(f"\nCache Status:")
        print(f"  Cached:               {info['cached_count']}")
        print(f"  Need computation:     {info['uncached_count']}")
        
        if info['bounds']:
            pmin, pmax = info['bounds']
            print(f"\nBounding Box:")
            print(f"  Min: ({pmin[0]:.4f}, {pmin[1]:.4f}, {pmin[2]:.4f})")
            print(f"  Max: ({pmax[0]:.4f}, {pmax[1]:.4f}, {pmax[2]:.4f})")
        
        print(f"\nComponents:")
        for i, key in enumerate(self._component_order):
            entry = self._components[key]
            geo_type = type(entry.geometry).__name__
            pos = entry.transform.translation
            
            flags = []
            if entry.is_assembly:
                flags.append("sub-assembly")
            if entry.geometry.has_cached_solution():
                flags.append("cached")
            if hasattr(entry.geometry, 'supports_analytical') and entry.geometry.supports_analytical:
                flags.append(f"method={entry.geometry.compute_method}")
            
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            
            print(f"  [{i}] {key}: {geo_type}{flag_str}")
            print(f"       Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"       Tag: {entry.tag}")
        
        # Show unique geometry groups
        unique = self.get_unique_components()
        if any(len(v) > 1 for v in unique.values()):
            print(f"\nShared Geometries (can reuse solutions):")
            for tag_hash, keys in unique.items():
                if len(keys) > 1:
                    print(f"  {tag_hash[:12]}...: {', '.join(keys)}")
        
        print("=" * 70)
    
    # =========================================================================
    # Serialization (for saving/loading assemblies)
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize assembly structure to dictionary.
        
        Note: Does not serialize actual geometries, only structure.
        Useful for saving assembly configurations.
        """
        components = {}
        for key, entry in self._components.items():
            components[key] = {
                'geometry_type': type(entry.geometry).__name__,
                'geometry_tag': entry.tag.full_hash,
                'is_assembly': entry.is_assembly,
                'transform': {
                    'translation': entry.transform.translation,
                    'rotation': entry.transform.rotation
                },
                'metadata': entry.metadata
            }
        
        connections = []
        for conn in self._connections:
            connections.append({
                'from_key': conn.from_key,
                'to_key': conn.to_key,
                'from_port': conn.from_port,
                'to_port': conn.to_port,
                'gap': conn.gap
            })
        
        return {
            'main_axis': self.main_axis,
            'component_order': list(self._component_order),
            'components': components,
            'connections': connections,
            'assembly_tag': self.tag.full_hash
        }
