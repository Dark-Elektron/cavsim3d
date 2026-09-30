# primitives.py (with analytical support and tagging)
"""Primitive geometries with tagging and analytical solution support."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import json

from netgen.occ import Rectangle, X, Y, Z, Cylinder, Axes

from .base import BaseGeometry
from .component_registry import ComputeMethod


def _length_arg(value, alias, name: str, alias_name: str) -> float:
    """The guide length given as ``name`` or as its alias (``L`` / ``length``)."""
    if value is not None and alias is not None:
        raise TypeError(f"give the length as {name}= or {alias_name}=, not both")
    if value is None:
        value = alias
    if value is None:
        raise TypeError(f"missing the length: {name}= (or {alias_name}=), in metres")
    return value


def _replay_after_init(obj: BaseGeometry, history: List[dict]) -> BaseGeometry:
    """Replay the ops recorded AFTER ``__init__`` onto a freshly built ``obj``.

    ``__init__`` already builds and meshes (and records that ``generate_mesh``
    BEFORE its own ``__init__`` entry), so only later operations are replayed.
    """
    ops = [e.get('op') for e in history]
    start = ops.index('__init__') + 1 if '__init__' in ops else len(history)
    for entry in history[start:]:
        obj._replay_common_op(entry)
    return obj


class RectangularWaveguide(BaseGeometry):
    """
    Rectangular waveguide with optional analytical solution.
    
    Parameters
    ----------
    a : float
        Width (x-dimension) [m]
    b : float, optional
        Height (y-dimension) [m]. Default is a/2.
    L : float
        Length (z-dimension) [m]; ``length=`` is accepted too
    maxh : float
        Maximum mesh element size
    compute_method : str or ComputeMethod
        'numeric', 'analytical', or 'semi_analytical'
    
    Notes
    -----
    When `compute_method='analytical'`, the solver uses closed-form
    expressions for TE/TM modes instead of FEM.
    """

    def __init__(
            self,
            a: float,
            L: Optional[float] = None,
            b: Optional[float] = None,
            maxh: float = 0.05,
            compute_method: str = 'numeric',
            *,
            length: Optional[float] = None,
    ):
        super().__init__()
        L = _length_arg(L, length, 'L', 'length')
        self.a = a
        self.b = b if b is not None else a / 2
        self.L = L
        self.maxh = maxh
        
        if isinstance(compute_method, str):
            self._compute_method = ComputeMethod[compute_method.upper()]
        else:
            self._compute_method = compute_method

        self.build()
        self.generate_mesh(maxh=maxh)
        self._record('__init__', a=a, L=L, b=b, maxh=maxh, compute_method=compute_method)

    def build(self) -> None:
        """Build rectangular waveguide geometry."""
        self.geo = Rectangle(self.a, self.b).Face().Extrude(self.L * Z)

        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"
        self.geo.faces.Min(Y).name = "bottom"
        self.geo.faces.Max(Y).name = "top"
        self.geo.faces.Min(X).name = "left"
        self.geo.faces.Max(X).name = "right"

        self.geo.faces.Min(Z).col = (1, 0, 0)
        self.geo.faces.Max(Z).col = (1, 0, 0)

        self.geo.mat('vacuum')
        self.bc = 'left|right|top|bottom'
        self._bc_explicitly_set = True
        self.invalidate_tag()

    @property
    def supports_analytical(self) -> bool:
        return True

    @property
    def cutoff_frequency_TE10(self) -> float:
        """Cutoff frequency for TE10 mode [Hz]."""
        from cavsim3d.core.constants import c0
        return c0 / (2 * self.a)

    @property
    def cutoff_wavenumber_TE10(self) -> float:
        """Cutoff wavenumber for TE10 mode [rad/m]."""
        return np.pi / self.a

    def get_dimensions(self) -> dict:
        return {'a': self.a, 'b': self.b, 'L': self.L}
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        return {
            'class': 'RectangularWaveguide',
            'a': float(self.a),
            'b': float(self.b),
            'L': float(self.L),
            'bc': self.bc
        }
    
    def _get_mesh_params(self) -> Dict[str, Any]:
        return {
            'maxh': self.maxh,
            'nv': self.mesh.nv if self.mesh else None
        }
    
    def get_analytical_modes(self, n_modes: int = 10) -> List[Dict[str, Any]]:
        """
        Get analytical mode information.
        
        Returns list of dicts with 'type', 'indices', 'cutoff_frequency'.
        """
        from cavsim3d.core.constants import c0
        
        modes = []
        for m in range(5):
            for n in range(5):
                if m > 0 or n > 0:  # TE modes
                    fc = (c0 / 2) * np.sqrt((m / self.a)**2 + (n / self.b)**2)
                    modes.append({'type': 'TE', 'indices': (m, n), 'cutoff_frequency': fc})
                
                if m > 0 and n > 0:  # TM modes
                    fc = (c0 / 2) * np.sqrt((m / self.a)**2 + (n / self.b)**2)
                    modes.append({'type': 'TM', 'indices': (m, n), 'cutoff_frequency': fc})
        
        modes.sort(key=lambda x: x['cutoff_frequency'])
        return modes[:n_modes]

    @classmethod
    def _rebuild_from_history(
        cls,
        history: List[dict],
        project_path: Path,
        source_file: Optional[Path] = None,
    ) -> 'RectangularWaveguide':
        """Reconstruct from operation history."""
        # Find initialization parameters
        params = {}
        for entry in history:
            if entry['op'] == '__init__':
                params = entry
                break
        
        # Fallback to reasonable defaults if not found
        a = params.get('a', 0.1)
        L = params.get('L', 0.2)
        b = params.get('b')
        maxh = params.get('maxh', 0.05)
        meth = params.get('compute_method', 'numeric')
        
        obj = cls(a=a, L=L, b=b, maxh=maxh, compute_method=meth)
        return _replay_after_init(obj, history)

    def save_geometry(self, project_path) -> None:
        """Save primitive geometry as STEP + history."""
        project_path = Path(project_path)
        geo_dir = project_path / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)

        # Export STEP file
        if self.geo is not None:
            try:
                self.geo.WriteStep(str(geo_dir / 'cavsim3d.geometry.step'))
            except Exception:
                pass

        meta = {
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'source_link': None,
            'source_filename': 'cavsim3d.geometry.step',
            'source_hash': None,
            'history': self._history,
        }

        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

class CircularWaveguide(BaseGeometry):
    """Circular waveguide with optional analytical solution.

    Parameters
    ----------
    radius : float
        Radius [m]
    length : float
        Length along z [m]; ``L=`` is accepted too
    maxh : float
        Maximum mesh element size [m]
    compute_method : str or ComputeMethod
        'numeric', 'analytical', or 'semi_analytical'
    """

    def __init__(
            self,
            radius: float,
            length: Optional[float] = None,
            maxh: float = 0.05,
            compute_method: str = 'numeric',
            *,
            L: Optional[float] = None,
    ):
        super().__init__()
        length = _length_arg(length, L, 'length', 'L')
        self.radius = radius
        self.length = length
        self.maxh = maxh
        
        if isinstance(compute_method, str):
            self._compute_method = ComputeMethod[compute_method.upper()]
        else:
            self._compute_method = compute_method

        self.build()
        self.generate_mesh(maxh=maxh)
        self._record('__init__', radius=radius, length=length, maxh=maxh, compute_method=compute_method)

    def build(self) -> None:
        self.geo = Cylinder(Axes((0, 0, 0), Z), r=self.radius, h=self.length)

        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"

        for face in self.geo.faces:
            if face.name not in ["port1", "port2"]:
                face.name = "default"

        self.geo.faces.Min(Z).col = (1, 0, 0)
        self.geo.faces.Max(Z).col = (1, 0, 0)

        self.geo.mat('vacuum')
        self.bc = 'default'
        self._bc_explicitly_set = True
        self.invalidate_tag()

    @property
    def supports_analytical(self) -> bool:
        return True

    @property
    def cutoff_frequency_TE11(self) -> float:
        from cavsim3d.core.constants import c0
        p_11 = 1.8412  # First zero of J'_1
        return c0 * p_11 / (2 * np.pi * self.radius)

    @property
    def cutoff_frequency_TM01(self) -> float:
        from cavsim3d.core.constants import c0
        p_01 = 2.4048  # First zero of J_0
        return c0 * p_01 / (2 * np.pi * self.radius)

    def get_dimensions(self) -> dict:
        return {'radius': self.radius, 'length': self.length}
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        return {
            'class': 'CircularWaveguide',
            'radius': float(self.radius),
            'length': float(self.length),
            'bc': self.bc
        }
    
    def _get_mesh_params(self) -> Dict[str, Any]:
        return {'maxh': self.maxh, 'nv': self.mesh.nv if self.mesh else None}

    @classmethod
    def _rebuild_from_history(
        cls,
        history: List[dict],
        project_path: Path,
        source_file=None,
    ) -> 'CircularWaveguide':
        """Reconstruct from operation history."""
        params = {}
        for entry in history:
            if entry['op'] == '__init__':
                params = entry
                break
        
        radius = params.get('radius', 0.05)
        length = params.get('length', 0.2)
        maxh = params.get('maxh', 0.05)
        meth = params.get('compute_method', 'numeric')
        
        obj = cls(radius=radius, length=length, maxh=maxh, compute_method=meth)
        return _replay_after_init(obj, history)

    def save_geometry(self, project_path) -> None:
        """Save primitive geometry as STEP + history."""
        project_path = Path(project_path)
        geo_dir = project_path / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)

        if self.geo is not None:
            try:
                self.geo.WriteStep(str(geo_dir / 'cavsim3d.geometry.step'))
            except Exception:
                pass

        meta = {
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'source_link': None,
            'source_filename': 'cavsim3d.geometry.step',
            'source_hash': None,
            'history': self._history,
        }

        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)


class Box(BaseGeometry):
    """Simple box/cavity geometry.

    Parameters
    ----------
    dimensions : (a, b, L)
        Extents along x, y, z [m].
    port_faces : tuple of str
        The two faces that become ``port1`` / ``port2``, written as
        ``'Min(X)'`` ... ``'Max(Z)'``; the other four are PEC walls.
    maxh : float
        Maximum mesh element size [m].
    """

    _SIDE_NAMES = {('Min', 'X'): 'left', ('Max', 'X'): 'right',
                   ('Min', 'Y'): 'bottom', ('Max', 'Y'): 'top',
                   ('Min', 'Z'): 'front', ('Max', 'Z'): 'back'}

    def __init__(
            self,
            dimensions: tuple,
            port_faces: tuple = ('Min(Z)', 'Max(Z)'),
            maxh: float = 0.05
    ):
        super().__init__()
        self.dimensions = dimensions
        self.port_faces = port_faces
        self.maxh = maxh

        self.build()
        self.generate_mesh(maxh=maxh)
        self._record('__init__', dimensions=dimensions, port_faces=port_faces, maxh=maxh)

    def build(self) -> None:
        from netgen.occ import Box as OCCBox

        import re as _re
        a, b, L = self.dimensions
        self.geo = OCCBox((0, 0, 0), (a, b, L))
        axes = {'X': X, 'Y': Y, 'Z': Z}

        def _face(spec: str):
            m = _re.fullmatch(r'\s*(Min|Max)\(\s*([XYZ])\s*\)\s*', str(spec))
            if not m:
                raise ValueError(f"Box port face {spec!r} must look like 'Min(Z)' or 'Max(X)'.")
            return m.group(1), m.group(2)

        port_keys = [_face(f) for f in self.port_faces]
        if len(port_keys) != 2 or port_keys[0] == port_keys[1]:
            raise ValueError(f"Box needs two distinct port faces, got {self.port_faces}.")

        walls = []
        for (side, ax), name in self._SIDE_NAMES.items():
            faces = getattr(self.geo.faces, side)(axes[ax])
            if (side, ax) in port_keys:
                faces.name = f"port{port_keys.index((side, ax)) + 1}"
                faces.col = (1, 0, 0)
            else:
                faces.name = name
                walls.append(name)

        self.geo.mat('vacuum')
        self.bc = '|'.join(walls)
        self._bc_explicitly_set = True
        self.invalidate_tag()
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        return {
            'class': 'Box',
            'dimensions': tuple(float(d) for d in self.dimensions),
            'port_faces': self.port_faces
        }

    @classmethod
    def _rebuild_from_history(
        cls,
        history: List[dict],
        project_path: Path,
        source_file=None,
    ) -> 'Box':
        """Reconstruct from operation history."""
        params = {}
        for entry in history:
            if entry['op'] == '__init__':
                params = entry
                break
        
        dims = params.get('dimensions', (0.1, 0.1, 0.2))
        port_faces = params.get('port_faces', ('Min(Z)', 'Max(Z)'))
        maxh = params.get('maxh', 0.05)
        
        obj = cls(dimensions=tuple(dims), port_faces=tuple(port_faces), maxh=maxh)
        return _replay_after_init(obj, history)

    def save_geometry(self, project_path) -> None:
        """Save primitive geometry as STEP + history."""
        project_path = Path(project_path)
        geo_dir = project_path / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)

        if self.geo is not None:
            try:
                self.geo.WriteStep(str(geo_dir / 'cavsim3d.geometry.step'))
            except Exception:
                pass

        meta = {
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'source_link': None,
            'source_filename': 'cavsim3d.geometry.step',
            'source_hash': None,
            'history': self._history,
        }

        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)


class Sphere(BaseGeometry):
    """Spherical cavity resonator (closed PEC wall, no ports)."""

    def __init__(
            self,
            radius: float,
            maxh: float = 0.05,
            material: str = 'vacuum'
    ):
        super().__init__()
        self.radius = radius
        self.maxh = maxh
        self.material = material

        self.build()
        self.generate_mesh(maxh=maxh)
        self._record('__init__', radius=radius, maxh=maxh, material=material)

    def build(self) -> None:
        from netgen.occ import Sphere as OCCSphere, Pnt

        self.geo = OCCSphere(Pnt(0, 0, 0), self.radius)
        for f in self.geo.faces:
            f.name = 'wall'
        self.geo.mat(self.material)
        self.bc = 'wall'
        self._bc_explicitly_set = True
        self.invalidate_tag()

    def _get_geometry_params(self) -> Dict[str, Any]:
        return {
            'class': 'Sphere',
            'radius': float(self.radius),
            'material': self.material,
        }

    @classmethod
    def _rebuild_from_history(
        cls,
        history: List[dict],
        project_path: Path,
        source_file=None,
    ) -> 'Sphere':
        """Reconstruct from operation history."""
        params = {}
        for entry in history:
            if entry['op'] == '__init__':
                params = entry
                break
        obj = cls(radius=params.get('radius', 0.1),
                  maxh=params.get('maxh', 0.05),
                  material=params.get('material', 'vacuum'))
        return _replay_after_init(obj, history)

    def save_geometry(self, project_path) -> None:
        """Save primitive geometry as STEP + history."""
        project_path = Path(project_path)
        geo_dir = project_path / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)

        if self.geo is not None:
            try:
                self.geo.WriteStep(str(geo_dir / 'cavsim3d.geometry.step'))
            except Exception:
                pass

        meta = {
            'type': self.__class__.__name__,
            'module': self.__class__.__module__,
            'source_link': None,
            'source_filename': 'cavsim3d.geometry.step',
            'source_hash': None,
            'history': self._history,
        }

        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
