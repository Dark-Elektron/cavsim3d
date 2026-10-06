"""Beam impedance of a collimator: one solid, a beam on its axis.

A round beam pipe of radius 100 mm tapers down to 50 mm and back up (the
collimator of E. Gjonaj's 2019 slides).  ``proj.add_beam()`` puts a beam --
a line current of 1 A travelling at the speed of light -- on the axis.  The
solve then returns, next to the S-parameters, the generalised scattering
matrix ``s_tilde = [[S, k], [h, z_b]]``: ``k`` is what the beam sends into
each port mode, ``h`` the beam voltage of each incoming port wave, and
``z_b`` the beam impedance with every port mode matched.  The longitudinal
impedance is ``Z_par = -z_b`` (``fom.beam_impedance()``).

Below the TM01 cut-off of the 100 mm pipe (1.15 GHz) nothing radiates: Z_par
is inductive and close to Yokoya's small-angle estimate
``j k Z0 / (4 pi) int (dr/dz)^2 dz`` (3 % above the computed value).  Above
the cut-off the TM01 wave, faster than the beam, keeps exchanging energy with
it along the pipes up to the port planes, so z_b of a finite model depends on
the pipe lengths there; this example stays below.

The beam impedance is a small effect of a large field: the beam's own field
is almost normal to the walls, and how well the mesh follows the curved walls
matters more than the element order.  Mesh with ``curve_order=4``.

Run:  python examples/beam/collimator_impedance.py
"""
import tempfile
import time
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.core.constants import Z0, c0
from cavsim3d.geometry.axisymmetric.base import AxisymmetricGeometry
from cavsim3d.geometry.axisymmetric.profile import Profile

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_beam_"))

# (z, r) corners of the wall, metres
CORNERS = [(-1.50, 0.10), (-0.95, 0.10), (-0.15, 0.05),
           (0.15, 0.05), (0.95, 0.10), (1.50, 0.10)]


class Collimator(AxisymmetricGeometry):
    """A body of revolution through the (z, r) ``corners`` (metres)."""

    def __init__(self, corners=CORNERS, maxh=0.035):
        super().__init__()
        self.corners = [tuple(map(float, c)) for c in corners]
        self._set_unit('m')
        self._finish_init(maxh, dict(corners=self.corners, maxh=maxh))

    def profile(self) -> Profile:
        (z0, r0), (z1, _r1) = self.corners[0], self.corners[-1]
        prof = Profile('collimator').start(z0, 0.0).line_to(z0, r0, 'PMC')
        for z, r in self.corners[1:]:
            prof.line_to(z, r, 'PEC')
        return prof.line_to(z1, 0.0, 'PMC').close('AXI')


def yokoya(f_hz):
    """j k Z0 / (4 pi) int (dr/dz)^2 dz over the tapers of CORNERS."""
    slopes = sum((r1 - r0) ** 2 / (z1 - z0)
                 for (z0, r0), (z1, r1) in zip(CORNERS, CORNERS[1:]))
    return 1j * (2 * np.pi * np.asarray(f_hz) / c0) * Z0 / (4 * np.pi) * slopes


def main(fmin=0.1, fmax=1.1, nsamples=11, order=3, maxh=0.035, show=True):
    proj = EMProject(name="collimator", base_dir=str(WORK), overwrite=True)
    proj.geometry = Collimator(maxh=maxh)
    proj.generate_mesh(maxh=maxh, curve_order=4)
    proj.add_beam('beam')                       # on the axis: x = y = 0

    t0 = time.time()
    proj.fds.solve(fmin=fmin, fmax=fmax, nsamples=nsamples, nportmodes=3, order=order)
    fom = proj.fds.fom
    print(f"solved in {time.time() - t0:.0f} s, {proj.fds.fes.ndof} unknowns")
    print(f"S~ rows {fom.tilde_labels[0]}\n   columns {fom.tilde_labels[1]}")

    zpar = fom.beam_impedance()                 # matched ports
    f = fom.frequencies
    print(f"\n{'f [GHz]':>8} {'Re Z_par':>10} {'Im Z_par':>10} {'Yokoya':>9}  [Ohm]")
    for fk, zk, yk in zip(f, zpar, yokoya(f)):
        print(f"{fk / 1e9:8.3f} {zk.real:10.4f} {zk.imag:10.4f} {yk.imag:9.4f}")
    # k: the waves the beam sends into the port modes (all evanescent here)
    k = fom.s_tilde_dict
    print(f"\nlargest |k| (beam -> port 1, modes 1-3): "
          f"{max(abs(k[f'b(1)1({m})']).max() for m in (1, 2, 3)):.2e}")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = fom.plot_beam_impedance(label="collimator", marker="o")
        ax.plot(f / 1e9, yokoya(f).imag, "k:", label="Yokoya (small angle)")
        ax.legend()
        ax.set_title("Longitudinal impedance of the collimator, beam on axis")
        plt.show()
    return proj


if __name__ == "__main__":
    main()
