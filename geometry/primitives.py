"""Primitive geometry definitions."""

from typing import Optional
import numpy as np

from netgen.occ import Rectangle, X, Y, Z
from ngsolve import Mesh
from netgen.occ import OCCGeometry

from .base import BaseGeometry


class RectangularWaveguide(BaseGeometry):
    """
    Rectangular waveguide geometry.

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
    """

    def __init__(
            self,
            a: float,
            L: float,
            b: Optional[float] = None,
            maxh: float = 0.05
    ):
        super().__init__()
        self.a = a
        self.b = b if b is not None else a / 2
        self.L = L
        self.maxh = maxh

        # Build geometry on initialization
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self) -> None:
        """Build rectangular waveguide geometry."""
        self.geo = Rectangle(self.a, self.b).Face().Extrude(self.L * Z)

        # Name faces
        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"
        self.geo.faces.Min(Y).name = "bottom"
        self.geo.faces.Max(Y).name = "top"
        self.geo.faces.Min(X).name = "left"
        self.geo.faces.Max(X).name = "right"

        # Color ports
        self.geo.faces.Min(Z).col = (1, 0, 0)
        self.geo.faces.Max(Z).col = (1, 0, 0)

        # Set material
        self.geo.mat('vacuum')

        # Set boundary conditions (PEC on walls)
        self.bc = 'left|right|top|bottom'

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
        """Return geometry dimensions."""
        return {'a': self.a, 'b': self.b, 'L': self.L}


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
        """Build box geometry."""
        from netgen.occ import Box as OCCBox

        a, b, L = self.dimensions
        self.geo = OCCBox((0, 0, 0), (a, b, L))

        # Name faces based on port_faces configuration
        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"
        self.geo.faces.Min(Y).name = "bottom"
        self.geo.faces.Max(Y).name = "top"
        self.geo.faces.Min(X).name = "left"
        self.geo.faces.Max(X).name = "right"

        self.geo.mat('vacuum')
        self.bc = 'left|right|top|bottom'


class CircularWaveguide(BaseGeometry):
    """
    Circular waveguide geometry.

    Parameters
    ----------
    radius : float
        Waveguide radius [m]
    length : float
        Waveguide length [m]
    maxh : float
        Maximum mesh element size
    """

    def __init__(self, radius: float, length: float, maxh: float = 0.05):
        super().__init__()
        self.radius = radius
        self.length = length
        self.maxh = maxh

        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self) -> None:
        """Build circular waveguide geometry."""
        from netgen.occ import Cylinder, X, Y, Z, Axes

        # Create cylinder along Z axis
        self.geo = Cylinder(
            Axes((0, 0, 0), Z),
            r=self.radius,
            h=self.length
        )

        # Name faces
        self.geo.faces.Min(Z).name = "port1"
        self.geo.faces.Max(Z).name = "port2"

        # The curved surface is the wall
        # In OCC, we need to identify it differently
        for face in self.geo.faces:
            if face.name not in ["port1", "port2"]:
                face.name = "wall"

        # Color ports
        self.geo.faces.Min(Z).col = (1, 0, 0)
        self.geo.faces.Max(Z).col = (1, 0, 0)

        self.geo.mat('vacuum')
        self.bc = 'wall'

    @property
    def cutoff_frequency_TE11(self) -> float:
        """Cutoff frequency for TE11 mode [Hz]."""
        from core.constants import c0
        from analytical.circular_waveguide import CWGAnalytical

        p_11 = CWGAnalytical.BESSEL_ZEROS_DERIVATIVE[(1, 1)]
        return c0 * p_11 / (2 * np.pi * self.radius)

    @property
    def cutoff_frequency_TM01(self) -> float:
        """Cutoff frequency for TM01 mode [Hz]."""
        from core.constants import c0
        from analytical.circular_waveguide import CWGAnalytical

        p_01 = CWGAnalytical.BESSEL_ZEROS[(0, 1)]
        return c0 * p_01 / (2 * np.pi * self.radius)

    def get_dimensions(self) -> dict:
        """Return geometry dimensions."""
        return {'radius': self.radius, 'length': self.length}