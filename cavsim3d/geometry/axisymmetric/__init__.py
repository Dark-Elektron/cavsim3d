"""Bodies of revolution: the cavsim2d models, revolved about the beam axis.

Each model keeps the cavsim2d class name and constructor, so a structure
defined for the 2D code builds its 3D solid here. The meridian is built as a
:class:`Profile` (exact lines, circular and elliptical arcs, splines) and
swept 360 degrees about Z by :func:`revolve`; the beam apertures become the
ports ``port1`` (low z) and ``port2``, and the rest of the surface is the PEC
wall ``'default'``.

Dimensions are in millimetres by default, as in cavsim2d (the RF gun in metres,
its angles in radians); ``unit=`` selects another unit. ``maxh`` is in metres,
as everywhere in cavsim3d. Every constructor also takes its arguments as one
dict, ``config={...}``. A model is built without a mesh: ``generate_mesh()``
makes it, or the first solve with the model's own ``maxh``.

A new body of revolution needs only a subclass of
:class:`AxisymmetricGeometry` that implements ``profile()``.
"""

from .profile import Profile, MaterialRegion, revolve
from .contours import DegenerateGeometry
from .base import AxisymmetricGeometry
from .elliptical import EllipticalCavity, EllipticalCavityFlatTop
from .rfgun import RFGun
from .pillbox import Pillbox
from .spline import SplineCavity
from .beampipe import Beampipe, BLA
from .bellows import Bellows
from .taper import Taper

__all__ = [
    'Profile', 'MaterialRegion', 'revolve', 'DegenerateGeometry',
    'AxisymmetricGeometry',
    'EllipticalCavity', 'EllipticalCavityFlatTop', 'RFGun', 'Pillbox',
    'SplineCavity', 'Beampipe', 'BLA', 'Bellows', 'Taper',
]
