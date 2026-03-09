# primitives.py (with analytical support and tagging)
"""Primitive geometries with tagging and analytical solution support."""

from typing import Optional, Dict, Any, List
import numpy as np

from netgen.occ import Rectangle, X, Y, Z, Cylinder, Axes
from ngsolve import Mesh
from netgen.occ import OCCGeometry

from .base import BaseGeometry
from .component_registry import ComputeMethod


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
        Length (z-dimension) [m]
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
            L: float,
            b: Optional[float] = None,
            maxh: float = 0.05,
            compute_method: str = 'numeric'
    ):
        super().__init__()
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
        from core.constants import c0
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
        from core.constants import c0
        
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


class CircularWaveguide(BaseGeometry):
    """Circular waveguide with optional analytical solution."""

    def __init__(
            self,
            radius: float,
            length: float,
            maxh: float = 0.05,
            compute_method: str = 'numeric'
    ):
        super().__init__()
        self.radius = radius
        self.length = length
        self.maxh = maxh
        
        if isinstance(compute_method, str):
            self._compute_method = ComputeMethod[compute_method.upper()]
        else:
            self._compute_method = compute_method

        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self) -> None:
        self.geo = Cylinder(Axes((0, 0, 0), Z), r=self.radius, h=self.length)

        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"

        for face in self.geo.faces:
            if face.name not in ["port1", "port2"]:
                face.name = "wall"

        self.geo.faces.Min(Z).col = (1, 0, 0)
        self.geo.faces.Max(Z).col = (1, 0, 0)

        self.geo.mat('vacuum')
        self.bc = 'wall'
        self._bc_explicitly_set = True
        self.invalidate_tag()

    @property
    def supports_analytical(self) -> bool:
        return True

    @property
    def cutoff_frequency_TE11(self) -> float:
        from core.constants import c0
        p_11 = 1.8412  # First zero of J'_1
        return c0 * p_11 / (2 * np.pi * self.radius)

    @property
    def cutoff_frequency_TM01(self) -> float:
        from core.constants import c0
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


class Box(BaseGeometry):
    """Simple box/cavity geometry."""

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

    def build(self) -> None:
        from netgen.occ import Box as OCCBox

        a, b, L = self.dimensions
        self.geo = OCCBox((0, 0, 0), (a, b, L))

        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"
        self.geo.faces.Min(Y).name = "bottom"
        self.geo.faces.Max(Y).name = "top"
        self.geo.faces.Min(X).name = "left"
        self.geo.faces.Max(X).name = "right"

        self.geo.mat('vacuum')
        self.bc = 'left|right|top|bottom'
        self._bc_explicitly_set = True
        self.invalidate_tag()
    
    def _get_geometry_params(self) -> Dict[str, Any]:
        return {
            'class': 'Box',
            'dimensions': tuple(float(d) for d in self.dimensions),
            'port_faces': self.port_faces
        }