"""Shared test helper functions for creating mock data."""

import numpy as np
from cavsim3d.solvers.results import FOMResult


def make_frequencies(n=50):
    """Create a linearly spaced frequency array in Hz."""
    return np.linspace(1.0, 3.0, n) * 1e9


def make_z_dict(n=50):
    """Create a mock Z-parameter dictionary."""
    freqs = make_frequencies(n)
    Z11 = np.random.randn(n) + 1j * np.random.randn(n)
    Z12 = np.random.randn(n) + 1j * np.random.randn(n)
    Z21 = Z12.copy()
    Z22 = np.random.randn(n) + 1j * np.random.randn(n)
    return {
        'frequencies': freqs,
        '1(1)1(1)': Z11,
        '1(1)2(1)': Z12,
        '2(1)1(1)': Z21,
        '2(1)2(1)': Z22,
    }


def make_fom(domain='default', n=50, solver_ref=None):
    """Create a mock FOMResult for testing."""
    freqs = make_frequencies(n)
    z = make_z_dict(n)
    s = {k: v * 0.5 for k, v in z.items() if k != 'frequencies'}
    s['frequencies'] = freqs
    return FOMResult(
        domain=domain,
        frequencies=freqs,
        Z_matrix=None,
        S_matrix=None,
        Z_dict=z,
        S_dict=s,
        n_ports=2,
        ports=['port1', 'port2'],
        _solver_ref=solver_ref,
    )
