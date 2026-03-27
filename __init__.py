"""
CAVSIM - Cavity Simulation Package
Electromagnetic eigenmode analysis and model order reduction for RF cavities.
"""

from .core.constants import mu0, eps0, c0, Z0
from .geometry.primitives import RectangularWaveguide
from .solvers.frequency_domain import FrequencyDomainSolver
from .rom.reduction import ModelOrderReduction
from .solvers.frequency_domain import FrequencyDomainSolver
from .solvers.concatenation import ConcatenatedSystem
from .solvers.results import FOMResult, FOMCollection, ROMCollection

__version__ = "0.1.0"
__all__ = [
    "FrequencyDomainSolver",
    "ModelOrderReduction",
    "ConcatenatedSystem",
    "RWGAnalytical",
]
