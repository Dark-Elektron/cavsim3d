"""Electromagnetic solvers."""

from .base import BaseEMSolver, ParameterConverter
from .frequency_domain import FrequencyDomainSolver
from .concatenation import ConcatenatedSystem, ReducedConcatenatedSystem
from .results import FOMResult, FOMCollection, ROMCollection
from .ports import PortEigenmodeSolver

__all__ = [
    'BaseEMSolver',
    'ParameterConverter',
    'FrequencyDomainSolver',
    'ConcatenatedSystem',
    'ReducedConcatenatedSystem',
    'FOMResult',
    'FOMCollection',
    'ROMCollection',
    'PortEigenmodeSolver',
]