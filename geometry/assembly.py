# assembly.py
"""Assembly class with hierarchical support, multi-solid output, and solution caching."""

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
    base_name: str  # Original name before suffixing
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
    Assembly of geometry components with multi-solid output.
    
    Creates a compound geometry where each component remains a separate solid,
    similar to split geometries. Supports automatic positioning, unified port
    naming, and tracks identical components for solver optimization.
    
    Parameters
    ----------
    main_axis : str
        Primary axis for concatenation ('X', 'Y', or 'Z')
    
    Examples
    --------
    **TESLA 9-cell cavity with identical midcells:**
    
    >>> assembly = Assembly(main_axis='Z')
    >>> assembly.add("endcell_l", endcell_l)
    >>> for i in range(7):
    ...     assembly.add("midcell", midcell, after="endcell_l" if i == 0 else "midcell")
    >>> assembly.add("endcell_r", endcell_r, after="midcell")
    >>> assembly.build()
    >>> 
    >>> # Check what was created
    >>> assembly.print_info()
    >>> # Shows: midcell_1, midcell_2, ..., midcell_7 (all identical)
    
    **Resulting structure:**
    - Solids: endcell_l, midcell_1, midcell_2, ..., midcell_7, endcell_r
    - Ports: port1 (external), port2 (interface), ..., port10 (external)
    """
    
    _AXIS_VEC = {'X': X, 'Y': Y, 'Z': Z}
    _AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}
    
    def __init__(self, main_axis: Literal['X', 'Y', 'Z'] = 'Z'):
        super().__init__()
        self.main_axis = main_axis.upper()
        
        self._components: Dict[str, ComponentEntry] = {}
        self._component_order: List[str] = []
        self._connections: List[Connection] = []
        
        # Track base names for duplicate detection
        self._base_name_counts: Dict[str, int] = {}
        self._base_name_groups: Dict[str, List[str]] = {}  # base_name -> [key1, key2, ...]
        
        self._layout_computed = False
        self._is_built = False
        
        # Port and solid information (populated by build())
        self._port_info: Dict[str, Dict] = {}
        self._solid_info: Dict[str, Dict] = {}
    
    # =========================================================================
    # Component Management (with duplicate name support)
    # =========================================================================
    
    def add(
        self,
        name: str,
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
        
        Duplicate names are allowed and automatically suffixed with _1, _2, etc.
        Components with the same base name are tracked for solver optimization
        (compute once, reuse for identical geometries).
        
        Parameters
        ----------
        name : str
            Component name (duplicates allowed, will be auto-suffixed)
        geometry : BaseGeometry or Assembly
            Component to add (can be another Assembly!)
        position : tuple, optional
            Explicit position (x, y, z)
        rotation : tuple, optional
            Rotation angles in degrees (rx, ry, rz)
        after : str, optional
            Place after this component along main axis (can use base name)
        before : str, optional
            Place before this component along main axis (can use base name)
        align_port : str, optional
            Port to use for alignment
        **metadata
            Additional metadata
        
        Returns
        -------
        self : Assembly
        
        Examples
        --------
        >>> assembly.add("midcell", midcell)           # -> "midcell_1"
        >>> assembly.add("midcell", midcell)           # -> "midcell_2"  
        >>> assembly.add("midcell", midcell)           # -> "midcell_3"
        >>> assembly.get_identical_components()
        {'midcell': ['midcell_1', 'midcell_2', 'midcell_3']}
        """
        base_name = name
        
        # Generate unique key if name already exists
        if name in self._components or name in self._base_name_counts:
            # This is a duplicate name
            if base_name not in self._base_name_counts:
                # First duplicate - rename the original
                self._rename_first_instance(base_name)
            
            # Increment count and generate new key
            self._base_name_counts[base_name] += 1
            key = f"{base_name}_{self._base_name_counts[base_name]}"
        else:
            # First instance of this name
            key = name
            self._base_name_counts[base_name] = 0  # Will become 1 if duplicated
        
        # Track in base name groups
        if base_name not in self._base_name_groups:
            self._base_name_groups[base_name] = []
        self._base_name_groups[base_name].append(key)
        
        # Handle sub-assemblies
        if isinstance(geometry, Assembly):
            if not geometry._is_built:
                geometry.build()
        elif geometry.geo is None:
            raise ValueError(f"Geometry '{name}' not built. Call build() first.")
        
        transform = Transform3D(
            translation=position or (0.0, 0.0, 0.0),
            rotation=rotation or (0.0, 0.0, 0.0)
        )
        
        entry = ComponentEntry(
            geometry=geometry,
            key=key,
            base_name=base_name,
            transform=transform,
            aligned_port=align_port,
            metadata=metadata
        )
        
        self._components[key] = entry
        
        # Handle ordering - resolve base names to actual keys
        resolved_after = self._resolve_component_ref(after) if after else None
        resolved_before = self._resolve_component_ref(before) if before else None
        
        if resolved_after is not None:
            idx = self._component_order.index(resolved_after) + 1
            self._component_order.insert(idx, key)
            
            self._connections.append(Connection(
                from_key=resolved_after,
                to_key=key,
                from_port='port2',
                to_port=align_port or 'port1'
            ))
            
        elif resolved_before is not None:
            idx = self._component_order.index(resolved_before)
            self._component_order.insert(idx, key)
            
            self._connections.append(Connection(
                from_key=key,
                to_key=resolved_before,
                from_port=align_port or 'port2',
                to_port='port1'
            ))
        else:
            self._component_order.append(key)
        
        self._layout_computed = False
        self._is_built = False
        self.invalidate_tag()
        
        return self
    
    def _rename_first_instance(self, base_name: str) -> None:
        """Rename the first instance of a component when duplicates are added."""
        if base_name not in self._components:
            return
        
        # The first instance needs to be renamed to base_name_1
        old_key = base_name
        new_key = f"{base_name}_1"
        
        # Update component entry
        entry = self._components[old_key]
        entry.key = new_key
        
        # Move in dict
        self._components[new_key] = entry
        del self._components[old_key]
        
        # Update order list
        idx = self._component_order.index(old_key)
        self._component_order[idx] = new_key
        
        # Update connections
        for conn in self._connections:
            if conn.from_key == old_key:
                conn.from_key = new_key
            if conn.to_key == old_key:
                conn.to_key = new_key
        
        # Update base name groups
        if base_name in self._base_name_groups:
            self._base_name_groups[base_name] = [
                new_key if k == old_key else k 
                for k in self._base_name_groups[base_name]
            ]
        
        # Set count to 1 (next will be _2)
        self._base_name_counts[base_name] = 1
    
    def _resolve_component_ref(self, ref: str) -> Optional[str]:
        """
        Resolve a component reference (base name or full key) to actual key.
        
        If ref is a base name with multiple instances, returns the last one.
        """
        if ref is None:
            return None
        
        # Direct match
        if ref in self._components:
            return ref
        
        # Try as base name - return last instance
        if ref in self._base_name_groups and self._base_name_groups[ref]:
            return self._base_name_groups[ref][-1]
        
        raise ValueError(f"Component '{ref}' not found")
    
    def get_identical_components(self) -> Dict[str, List[str]]:
        """
        Get groups of components that share the same base name (geometry).
        
        These components are candidates for solver optimization - compute
        the solution once and reuse for all identical instances.
        
        Returns
        -------
        dict
            {base_name: [key1, key2, ...]} for groups with more than one member
        """
        return {
            name: keys for name, keys in self._base_name_groups.items()
            if len(keys) > 1
        }
    
    def get_components_by_base_name(self, base_name: str) -> List[str]:
        """Get all component keys with a given base name."""
        return self._base_name_groups.get(base_name, [])
    
    # =========================================================================
    # Connection and Layout Methods (unchanged logic, updated for new structure)
    # =========================================================================
    
    def connect(
        self,
        name: str,
        geometry: Union[BaseGeometry, 'Assembly'],
        to_component: str,
        from_port: str = "port1",
        to_port: str = "port2",
        rotation: Optional[Tuple[float, float, float]] = None,
        gap: float = 0.0,
        **metadata
    ) -> 'Assembly':
        """Add component connected via ports to existing component."""
        resolved_to = self._resolve_component_ref(to_component)
        
        self.add(name, geometry, rotation=rotation, **metadata)
        
        # Get the key that was just added (might be suffixed)
        new_key = self._component_order[-1]
        
        self._connections.append(Connection(
            from_key=resolved_to,
            to_key=new_key,
            from_port=to_port,
            to_port=from_port,
            connection_type=ConnectionType.PORT_TO_PORT,
            gap=gap
        ))
        
        return self
    
    def remove(self, ref: str) -> 'Assembly':
        """Remove a component by key or base name (removes all instances if base name)."""
        keys_to_remove = []
        
        if ref in self._components:
            keys_to_remove = [ref]
        elif ref in self._base_name_groups:
            keys_to_remove = list(self._base_name_groups[ref])
        else:
            raise KeyError(f"Component '{ref}' not found")
        
        for key in keys_to_remove:
            entry = self._components[key]
            base_name = entry.base_name
            
            del self._components[key]
            self._component_order.remove(key)
            self._connections = [
                c for c in self._connections 
                if c.from_key != key and c.to_key != key
            ]
            
            # Update base name tracking
            if base_name in self._base_name_groups:
                self._base_name_groups[base_name].remove(key)
                if not self._base_name_groups[base_name]:
                    del self._base_name_groups[base_name]
                    del self._base_name_counts[base_name]
        
        self._layout_computed = False
        self._is_built = False
        self.invalidate_tag()
        
        return self

    def rotate(
        self, 
        ref: str, 
        angles: Tuple[float, float, float],
        center: Optional[Tuple[float, float, float]] = None
    ) -> 'Assembly':
        """Apply rotation to a component."""
        key = self._resolve_component_ref(ref)
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
    
    def translate(self, ref: str, offset: Tuple[float, float, float]) -> 'Assembly':
        """Apply translation to a component."""
        key = self._resolve_component_ref(ref)
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
        """Access component by key, base name, or index."""
        if isinstance(key, int):
            if key < 0 or key >= len(self._component_order):
                raise IndexError(f"Index {key} out of range")
            key = self._component_order[key]
        else:
            key = self._resolve_component_ref(key)
        
        return self._components[key]
    
    def __len__(self) -> int:
        return len(self._components)
    
    def __iter__(self):
        return iter(self._component_order)
    
    def __contains__(self, ref: str) -> bool:
        try:
            self._resolve_component_ref(ref)
            return True
        except (KeyError, ValueError):
            return False
    
    @property
    def keys(self) -> List[str]:
        return list(self._component_order)
    
    @property
    def components(self) -> Dict[str, ComponentEntry]:
        return dict(self._components)

    # =========================================================================
    # Layout Computation (unchanged)
    # =========================================================================
    
    def compute_layout(self) -> 'Assembly':
        """Compute positions based on connections."""
        if self._layout_computed:
            return self
        
        if not self._components:
            self._layout_computed = True
            return self
        
        positioned = set()
        first_key = self._component_order[0]
        positioned.add(first_key)
        
        max_iter = len(self._connections) * 2 + 1
        for _ in range(max_iter):
            for conn in self._connections:
                if conn.to_key in positioned:
                    continue
                if conn.from_key not in positioned:
                    continue
                self._position_connected(conn)
                positioned.add(conn.to_key)
        
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
        """Get port position in local coordinates."""
        if entry.original_bounds is None:
            return None
        
        pmin, pmax = entry.original_bounds
        axis_idx = self._AXIS_IDX[self.main_axis]
        center = [(pmin[i] + pmax[i]) / 2 for i in range(3)]
        
        if 'port1' in port_name.lower() or 'min' in port_name.lower():
            center[axis_idx] = pmin[axis_idx]
        elif 'port2' in port_name.lower() or 'max' in port_name.lower():
            center[axis_idx] = pmax[axis_idx]
        
        return tuple(center)
    
    def _position_connected(self, conn: Connection) -> None:
        """Position component based on port connection."""
        from_entry = self._components[conn.from_key]
        to_entry = self._components[conn.to_key]
        
        from_port_pos = self._get_port_position(from_entry, conn.from_port)
        to_port_pos = self._get_port_position(to_entry, conn.to_port)
        
        if from_port_pos is None or to_port_pos is None:
            self._position_after(conn.to_key, conn.from_key)
            return
        
        axis_idx = self._AXIS_IDX[self.main_axis]
        
        from_world = tuple(
            from_port_pos[i] + from_entry.transform.translation[i]
            for i in range(3)
        )
        
        translation = [0.0, 0.0, 0.0]
        
        for i in range(3):
            if i == axis_idx:
                translation[i] = from_world[i] - to_port_pos[i] + conn.gap
            else:
                from_center = (from_entry.original_bounds[0][i] + 
                             from_entry.original_bounds[1][i]) / 2
                to_center = (to_entry.original_bounds[0][i] + 
                           to_entry.original_bounds[1][i]) / 2
                translation[i] = (from_center + from_entry.transform.translation[i] 
                                 - to_center)
        
        to_entry.transform = Transform3D(
            translation=tuple(translation),
            rotation=to_entry.transform.rotation,
            rotation_center=to_entry.transform.rotation_center
        )
    
    def _position_after(self, key: str, after_key: str) -> None:
        """Position component after another along main axis."""
        entry = self._components[key]
        after_entry = self._components[after_key]
        
        axis_idx = self._AXIS_IDX[self.main_axis]
        
        after_max = (
            after_entry.transform.translation[axis_idx] + 
            after_entry.size[axis_idx]
        )
        
        translation = [0.0, 0.0, 0.0]
        
        for i in range(3):
            if i == axis_idx:
                translation[i] = after_max
            else:
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
    # Build - Multi-solid Assembly
    # =========================================================================
    
    def build(self) -> None:
        """
        Build multi-solid geometry from all components.
        
        Creates a compound where each component is a separate solid with:
        - Solid named after component key (e.g., 'midcell_1', 'midcell_2')
        - Sequential port naming (port1, port2, ...) along main axis
        - Interface ports (where solids meet) shared between adjacent components
        - External ports at assembly ends
        - All other faces named 'wall'
        """
        if not self._components:
            raise ValueError("No components in assembly")
        
        self.compute_layout()
        
        # Build sub-assemblies first
        for entry in self._components.values():
            if entry.is_assembly and not entry.geometry._is_built:
                entry.geometry.build()
        
        # Collect transformed shapes
        shapes = []
        shape_keys = []  # Track which key each shape belongs to
        
        for key in self._component_order:
            entry = self._components[key]
            shape = entry.geometry.geo
            transformed = self._apply_transform(shape, entry.transform)
            shapes.append(transformed)
            shape_keys.append(key)
        
        # Create multi-solid compound using Glue
        if len(shapes) == 1:
            self.geo = shapes[0]
        else:
            self.geo = Glue(shapes)
        
        # Name solids and setup boundaries
        self._name_solids(shape_keys)
        self._setup_boundaries()
        
        self._is_built = True
    
    def _name_solids(self, shape_keys: List[str]) -> None:
        """Name each solid in the geometry based on component keys."""
        self._solid_info = {}
        
        try:
            solids = list(self.geo.solids)
            n_solids = len(solids)
            
            if n_solids != len(shape_keys):
                # Mismatch - solids might have been merged or split
                # Fall back to position-based naming
                print(f"Warning: Expected {len(shape_keys)} solids, found {n_solids}")
                self._name_solids_by_position(solids)
                return
            
            # Sort solids by position along main axis to match component order
            axis_idx = self._AXIS_IDX[self.main_axis]
            
            def get_solid_position(solid):
                bb = solid.bounding_box
                return (bb[0][axis_idx] + bb[1][axis_idx]) / 2
            
            solids_with_pos = [(get_solid_position(s), i, s) for i, s in enumerate(solids)]
            solids_with_pos.sort(key=lambda x: x[0])
            
            # Match sorted solids to component order
            for order_idx, (pos, orig_idx, solid) in enumerate(solids_with_pos):
                key = shape_keys[order_idx]
                entry = self._components[key]
                
                # Name the solid
                solid.mat(key)
                
                # Store solid info
                bb = solid.bounding_box
                self._solid_info[key] = {
                    'material': key,
                    'base_name': entry.base_name,
                    'position': pos,
                    'bounds': (tuple(bb[0]), tuple(bb[1])),
                    'is_identical': len(self._base_name_groups.get(entry.base_name, [])) > 1
                }
                
        except AttributeError:
            # Single solid
            key = shape_keys[0]
            entry = self._components[key]
            self.geo.mat(key)
            self._solid_info[key] = {
                'material': key,
                'base_name': entry.base_name,
                'position': 0,
                'bounds': None,
                'is_identical': False
            }
    
    def _name_solids_by_position(self, solids: list) -> None:
        """Fallback: name solids by their position along main axis."""
        axis_idx = self._AXIS_IDX[self.main_axis]
        
        def get_solid_position(solid):
            bb = solid.bounding_box
            return (bb[0][axis_idx] + bb[1][axis_idx]) / 2
        
        solids_sorted = sorted(solids, key=get_solid_position)
        
        for i, solid in enumerate(solids_sorted):
            name = f"cell_{i + 1}"
            solid.mat(name)
            bb = solid.bounding_box
            self._solid_info[name] = {
                'material': name,
                'base_name': name,
                'position': get_solid_position(solid),
                'bounds': (tuple(bb[0]), tuple(bb[1])),
                'is_identical': False
            }
    
    def _setup_boundaries(self) -> None:
        """
        Setup boundary conditions with unified sequential port naming.
        
        - All flat faces perpendicular to main axis are ports
        - Faces at the same position share the same port name (interfaces)
        - Port numbers increase along the positive main axis direction
        - Non-port faces are named 'wall'
        """
        if self.geo is None:
            return
        
        axis_idx = self._AXIS_IDX[self.main_axis]
        tolerance = 1e-6
        
        # Step 1: Name all faces 'wall' first
        self._name_all_faces_wall()
        
        # Step 2: Identify all port faces and their positions
        port_faces = []
        
        try:
            for face in self.geo.faces:
                try:
                    fbb = face.bounding_box
                    face_extent = fbb[1][axis_idx] - fbb[0][axis_idx]
                    
                    if face_extent < tolerance:
                        face_pos = (fbb[0][axis_idx] + fbb[1][axis_idx]) / 2
                        port_faces.append((face_pos, face))
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: Error identifying faces: {e}")
            self.bc = 'default'
            self._bc_explicitly_set = True
            return
        
        if not port_faces:
            print("Warning: No port faces found")
            self.bc = 'default'
            self._bc_explicitly_set = True
            return
        
        # Step 3: Group faces by position
        port_faces.sort(key=lambda x: x[0])
        
        port_groups = []
        for pos, face in port_faces:
            matched = False
            for group in port_groups:
                if abs(group['position'] - pos) < tolerance:
                    group['faces'].append(face)
                    matched = True
                    break
            
            if not matched:
                port_groups.append({
                    'position': pos,
                    'faces': [face]
                })
        
        # Step 4: Sort groups by position and assign sequential port names
        port_groups.sort(key=lambda g: g['position'])
        
        self._port_info = {}
        
        for port_num, group in enumerate(port_groups, start=1):
            port_name = f'port{port_num}'
            
            is_external = len(group['faces']) == 1
            is_interface = len(group['faces']) > 1
            
            for face in group['faces']:
                face.name = port_name
            
            # Find which components connect at this port
            connected_components = self._find_components_at_position(
                group['position'], tolerance
            )
            
            self._port_info[port_name] = {
                'position': group['position'],
                'num_faces': len(group['faces']),
                'type': 'external' if is_external else 'interface',
                'axis': self.main_axis,
                'connected_components': connected_components
            }
        
        # Summary
        n_external = sum(1 for p in self._port_info.values() if p['type'] == 'external')
        n_interface = sum(1 for p in self._port_info.values() if p['type'] == 'interface')
        
        print(f"Port naming complete:")
        print(f"  Total ports: {len(self._port_info)}")
        print(f"  External ports: {n_external}")
        print(f"  Interface ports: {n_interface}")
        
        self.bc = 'default'
        self._bc_explicitly_set = True
    
    def _name_all_faces_wall(self) -> None:
        """Name every face 'wall' as default."""
        if self.geo is None:
            return
        
        try:
            for solid in self.geo.solids:
                for face in solid.faces:
                    face.name = 'default'
        except AttributeError:
            try:
                for face in self.geo.faces:
                    face.name = 'default'
            except AttributeError:
                pass
    
    def _find_components_at_position(
        self, 
        position: float, 
        tolerance: float
    ) -> List[Tuple[str, str]]:
        """Find which components have ports at a given position."""
        axis_idx = self._AXIS_IDX[self.main_axis]
        connected = []
        
        for key in self._component_order:
            entry = self._components[key]
            if entry.original_bounds is None:
                continue
            
            pmin, pmax = entry.original_bounds
            t = entry.transform.translation
            
            world_min = pmin[axis_idx] + t[axis_idx]
            world_max = pmax[axis_idx] + t[axis_idx]
            
            if abs(world_min - position) < tolerance:
                connected.append((key, 'port1'))
            if abs(world_max - position) < tolerance:
                connected.append((key, 'port2'))
        
        return connected
    
    def _apply_transform(self, shape, transform: Transform3D):
        """Apply transformation to OCC shape."""
        if transform.is_identity():
            return shape
        
        result = shape
        
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
        
        tx, ty, tz = transform.translation
        if abs(tx) > 1e-12 or abs(ty) > 1e-12 or abs(tz) > 1e-12:
            result = result.Move((tx, ty, tz))
        
        return result

    # =========================================================================
    # Port and Solid Information
    # =========================================================================
    
    def get_port_info(self) -> Dict[str, Dict]:
        """Get information about all ports in the assembly."""
        return dict(self._port_info) if self._port_info else {}
    
    def get_solid_info(self) -> Dict[str, Dict]:
        """Get information about all solids in the assembly."""
        return dict(self._solid_info) if self._solid_info else {}
    
    def get_external_ports(self) -> List[str]:
        """Get list of external port names."""
        return [name for name, info in self._port_info.items() 
                if info['type'] == 'external']
    
    def get_interface_ports(self) -> List[str]:
        """Get list of interface port names."""
        return [name for name, info in self._port_info.items() 
                if info['type'] == 'interface']
    
    def print_port_info(self) -> None:
        """Print detailed port information."""
        if not self._port_info:
            print("No port information available. Call build() first.")
            return
        
        print("\n" + "=" * 70)
        print("ASSEMBLY PORT MAP")
        print("=" * 70)
        print(f"Main axis: {self.main_axis}")
        print(f"Total ports: {len(self._port_info)}")
        print("-" * 70)
        
        for port_name in sorted(self._port_info.keys(), 
                                key=lambda x: int(x.replace('port', ''))):
            info = self._port_info[port_name]
            port_type = info['type'].upper()
            pos = info['position']
            n_faces = info['num_faces']
            
            type_marker = "◀▶" if info['type'] == 'external' else "◀┃▶"
            
            components = info.get('connected_components', [])
            comp_str = ", ".join(f"{k}.{p}" for k, p in components) if components else "?"
            
            print(f"  {port_name:8s} │ {self.main_axis}={pos:+.6f} │ "
                  f"{port_type:10s} │ {n_faces} face(s) {type_marker}")
            print(f"           │ Components: {comp_str}")
        
        print("=" * 70)
        
        # Visual schematic
        self._print_schematic()
    
    def _print_schematic(self) -> None:
        """Print a visual schematic of the assembly."""
        ports_sorted = sorted(
            self._port_info.items(), 
            key=lambda x: x[1]['position']
        )
        
        print(f"\nSchematic (along {self.main_axis}):\n")
        
        # Build port line
        port_parts = []
        for port_name, info in ports_sorted:
            if info['type'] == 'external':
                port_parts.append(f"║{port_name}║")
            else:
                port_parts.append(f"┃{port_name}┃")
        
        # Build component line
        comp_parts = []
        for key in self._component_order:
            entry = self._components[key]
            if entry.base_name != key:
                # Show base name for duplicates
                comp_parts.append(f"[{key}]")
            else:
                comp_parts.append(f"[{key}]")
        
        print("  Ports:      " + " ─── ".join(port_parts))
        print("  Components: " + " ─── ".join(comp_parts))
        
        # Show identical component groups
        identical = self.get_identical_components()
        if identical:
            print(f"\n  Identical components (compute once, reuse):")
            for base_name, keys in identical.items():
                print(f"    {base_name}: {', '.join(keys)}")
        print()

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
        print("-" * 60)
        for i, key in enumerate(self._component_order):
            entry = self._components[key]
            pos = entry.transform.translation
            geo_type = type(entry.geometry).__name__
            size = entry.size
            
            base_info = f" (={entry.base_name})" if entry.base_name != key else ""
            sub = " [sub-assembly]" if entry.is_assembly else ""
            cached = " [CACHED]" if entry.geometry.has_cached_solution() else ""
            
            print(f"  [{i}] {key}{base_info}: {geo_type}{sub}{cached}")
            print(f"      Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
            print(f"      Size:     ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
        
        # Show identical groups
        identical = self.get_identical_components()
        if identical:
            print(f"\nIdentical components:")
            for base_name, keys in identical.items():
                print(f"  {base_name}: {len(keys)} instances -> compute once")

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
    # Assembly Bounds and Info
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

    def summary(self) -> Dict[str, Any]:
        """Get assembly summary information."""
        identical = self.get_identical_components()
        
        return {
            'n_components': len(self._components),
            'n_unique_geometries': len(self._base_name_groups),
            'n_identical_groups': len(identical),
            'n_connections': len(self._connections),
            'n_ports': len(self._port_info),
            'n_external_ports': len(self.get_external_ports()),
            'n_interface_ports': len(self.get_interface_ports()),
            'main_axis': self.main_axis,
            'is_built': self._is_built,
            'has_mesh': self.mesh is not None,
            'component_keys': list(self._component_order),
            'identical_groups': identical,
            'bounds': self.get_assembly_bounds() if self._components else None,
        }
    
    def print_info(self) -> None:
        """Print detailed assembly information."""
        info = self.summary()
        
        print("\n" + "=" * 70)
        print("ASSEMBLY INFORMATION")
        print("=" * 70)
        print(f"Total components:       {info['n_components']}")
        print(f"Unique geometries:      {info['n_unique_geometries']}")
        print(f"Identical groups:       {info['n_identical_groups']}")
        print(f"Main axis:              {info['main_axis']}")
        print(f"Built:                  {info['is_built']}")
        print(f"Has mesh:               {info['has_mesh']}")
        
        print(f"\nPorts:")
        print(f"  Total:                {info['n_ports']}")
        print(f"  External:             {info['n_external_ports']}")
        print(f"  Interfaces:           {info['n_interface_ports']}")
        
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
            
            base_info = f" (base: {entry.base_name})" if entry.base_name != key else ""
            
            print(f"  [{i}] {key}{base_info}: {geo_type}")
            print(f"       Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        # Identical component groups
        if info['identical_groups']:
            print(f"\nIdentical Component Groups (solver optimization):")
            for base_name, keys in info['identical_groups'].items():
                print(f"  '{base_name}': {len(keys)} instances")
                print(f"    Keys: {', '.join(keys)}")
                print(f"    -> Compute once, reuse {len(keys)-1} times")
        
        print("=" * 70)
        
        # Print port map if built
        if self._is_built and self._port_info:
            self.print_port_info()

    # =========================================================================
    # Tagging and Cache Support
    # =========================================================================
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        """Get parameters for tag computation."""
        component_tags = {}
        for key, entry in self._components.items():
            component_tags[key] = {
                'tag_hash': entry.tag.full_hash,
                'base_name': entry.base_name,
                'transform': {
                    'translation': entry.transform.translation,
                    'rotation': entry.transform.rotation
                }
            }
        
        return {
            'class': 'Assembly',
            'main_axis': self.main_axis,
            'components': component_tags,
            'order': list(self._component_order),
            'identical_groups': self.get_identical_components()
        }
    
    def _get_mesh_params(self) -> Dict[str, Any]:
        """Get mesh parameters for tag computation."""
        return {
            'maxh': getattr(self, 'maxh', None),
            'component_count': len(self._components)
        }
    
    def get_solver_optimization_info(self) -> Dict[str, Any]:
        """
        Get information for solver optimization.
        
        Returns details about which components are identical and can
        share computed solutions.
        
        Returns
        -------
        dict
            {
                'total_components': int,
                'unique_computations': int,
                'reusable_results': int,
                'groups': {
                    base_name: {
                        'keys': [key1, key2, ...],
                        'compute_key': key1,  # Which one to compute
                        'reuse_keys': [key2, ...]  # Which ones reuse the result
                    }
                }
            }
        """
        identical = self.get_identical_components()
        
        # Single-instance components
        unique_keys = [
            keys[0] for name, keys in self._base_name_groups.items()
            if len(keys) == 1
        ]
        
        groups = {}
        total_reuse = 0
        
        for base_name, keys in identical.items():
            groups[base_name] = {
                'keys': keys,
                'compute_key': keys[0],
                'reuse_keys': keys[1:]
            }
            total_reuse += len(keys) - 1
        
        return {
            'total_components': len(self._components),
            'unique_computations': len(self._base_name_groups),
            'reusable_results': total_reuse,
            'efficiency': 1 - (len(self._base_name_groups) / len(self._components)) 
                         if self._components else 0,
            'unique_keys': unique_keys,
            'groups': groups
        }