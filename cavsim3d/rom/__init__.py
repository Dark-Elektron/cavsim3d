"""Reduced-order modeling tools."""

# The solver layer imports rom.reduction; loading it first lets this package be
# imported on its own (``from cavsim3d.rom import ModelOrderReduction``).
import cavsim3d.solvers  # noqa: F401
from .reduction import ModelOrderReduction
from .structures import ReducedStructure


__all__ = [
    'ModelOrderReduction',
    'ReducedStructure',
]
