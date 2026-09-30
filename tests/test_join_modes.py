"""How many modes propagate at a join (for the 'too few port modes' warning)."""
import numpy as np
import pytest

from cavsim3d.core.constants import c0
from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.solvers.concatenation import _n_propagating

R = 0.05


@pytest.mark.parametrize("kR, expected", [
    (1.5, 0),        # below TE11
    (2.0, 2),        # TE11 (both polarisations) only: used to count 0
    (2.3, 2),
    (2.6, 3),        # + TM01
    (3.2, 5),        # + TE21 (both)
])
def test_circular_port_mode_count(kR, expected):
    f = kR * c0 / (2 * np.pi * R)
    assert _n_propagating({"type": "circular", "radius": R}, f) == expected


def test_rectangular_port_mode_count():
    a, b = 0.1, 0.05                  # TE10 at 1.5 GHz, TE20/TE01 at 3 GHz
    geom = {"type": "rectangular", "a": a, "b": b}
    assert _n_propagating(geom, 1.0e9) == 0
    assert _n_propagating(geom, 2.0e9) == 1
    assert _n_propagating(geom, 3.2e9) == 3
