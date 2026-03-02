"""Analytical solutions for waveguides and cavities."""

from .rectangular_waveguide import RWGAnalytical, compare_eigenfrequencies
from .circular_waveguide import CWGAnalytical#, compare_cwg_eigenfrequencies

__all__ = [
    'RWGAnalytical',
    'CWGAnalytical',
    'compare_eigenfrequencies',
    # 'compare_cwg_eigenfrequencies'
]