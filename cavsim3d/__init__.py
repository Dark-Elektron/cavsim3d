"""cavsim3d - 3D RF structure analysis.

Note on import order: import :mod:`cavsim3d.core.em_project` (or
:mod:`cavsim3d.solvers.frequency_domain`) before :mod:`cavsim3d.rom` -- the
solver and ROM layers import each other, so entering via ``cavsim3d.rom``
first raises a circular ImportError.
"""

__version__ = "0.1.0"
