"""Degenerate port modes are numbered alike on every port.

A 0.1 x 0.05 m guide has TE20 and TE01 at the same cutoff.  Each port's size
is fitted from its own mesh face, so the two cutoffs differ in the last digits
from port to port; the mode order must not depend on that.
"""
import numpy as np

from cavsim3d.core.em_project import EMProject


def test_equal_cutoffs_same_order_on_both_ports(tmp_path):
    proj = EMProject("order", base_dir=str(tmp_path), overwrite=True)
    proj.create_primitive("rwg", name="guide", a=0.1, b=0.05, L=0.1, maxh=0.03)
    res = proj.fds.solve(fmin=3.2, fmax=3.4, nsamples=2, nportmodes=3, order=2,
                         solver_type="direct")
    ps = proj.fds.port_solver
    idx = {p: [tuple(ps.port_mode_indices[p][m]) for m in sorted(ps.port_mode_indices[p])]
           for p in ("port1", "port2")}
    assert idx["port1"] == idx["port2"]
    # a uniform guide does not convert one mode into another: with the modes
    # numbered alike, mode j in gives mode j out (a swapped pair would read ~1
    # here; the coarse test mesh leaves ~1e-2 between the degenerate pair)
    S = res["S"]
    for i in range(3):
        for j in range(3):
            if i != j:
                assert np.abs(S[:, 3 + i, j]).max() < 0.1
