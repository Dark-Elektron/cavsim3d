"""A beam past a lossy ceramic ring: the beam impedance is the absorbed power.

A round pipe (radius 25 mm) carries a ceramic ring at its wall (eps_r = 4,
tan delta = 0.5, inner radius 12 mm, 40 mm long), far from the ports.  The
beam's own field is that of the beam in vacuum; where the material differs,
the solver adds the difference as a volume load (the contrast load), so a
lossy or dielectric part may sit anywhere off the beam line.

Below the cut-off of the pipe nothing leaves through the ports, and with the
ports open the real part of the beam impedance is the power the ring absorbs
per (1 A)^2 / 2:

    Re Z_par = int w eps0 eps_r tan_delta |E|^2 dV,   E = E_s + E_free.

The lossless ring (tan delta = 0) gives Re Z_par = 0 to discretisation
accuracy.

Run:  python examples/beam/lossy_ring.py
"""
import tempfile
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject      # before netgen (pythonocc first)
from cavsim3d.core.constants import eps0
from cavsim3d.geometry.base import BaseGeometry
from netgen.occ import Axes, Cylinder, Glue, Z as AXIS_Z
from ngsolve import InnerProduct, Integrate

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_beam_"))
R, R_IN, L, L_RING = 0.025, 0.012, 0.24, 0.04          # metres


class RingPipe(BaseGeometry):
    """A pipe with a ring of material 'ceramic' at its wall, in the middle."""

    def build(self):
        z1 = (L - L_RING) / 2
        ring = (Cylinder(Axes((0, 0, z1), AXIS_Z), r=R, h=L_RING)
                - Cylinder(Axes((0, 0, z1), AXIS_Z), r=R_IN, h=L_RING))
        core = Cylinder(Axes((0, 0, 0), AXIS_Z), r=R, h=L) - ring
        core.mat('vacuum')
        ring.mat('ceramic')
        self.geo = Glue([core, ring])
        for f in self.geo.faces:
            lo, hi = f.bounding_box
            if hi.z - lo.z < 1e-6:                     # planar faces across the axis
                f.name = ('port1' if abs(lo.z) < 1e-6 else
                          'port2' if abs(lo.z - L) < 1e-6 else 'interface')
            else:                                      # the pipe wall, or the ring's inside
                f.name = 'default' if abs(hi.x - lo.x - 2 * R) < 1e-6 else 'interface'
        self.bc = 'default'


def main(tan_delta=0.5, maxh=0.006, order=3, fmin=0.5, fmax=1.5, nsamples=3):
    results = {}
    for tand in (0.0, tan_delta):
        proj = EMProject(name=f"ring_tand_{tand}", base_dir=str(WORK), overwrite=True)
        geo = RingPipe()
        geo.set_materials({'ceramic': {'eps_r': 4.0, 'tan_delta': tand}})
        proj.geometry = geo
        proj.generate_mesh(maxh=maxh, curve_order=4)
        proj.add_beam('beam')
        proj.fds.solve(fmin=fmin, fmax=fmax, nsamples=nsamples, nportmodes=3, order=order)
        results[tand] = proj

    lossy = results[tan_delta].fds
    zpar = lossy.fom.beam_impedance(ports='open')
    z0 = results[0.0].fds.fom.beam_impedance(ports='open')
    print(f"{'f [GHz]':>8} {'Re Z_par lossless':>18} {'Re Z_par lossy':>15} {'absorbed':>10}  [Ohm]")
    for k, f in enumerate(lossy.frequencies):
        w = 2 * np.pi * f
        E = lossy.fom.beam_field(k)                    # E_s + E_free, 1 A
        absorbed = Integrate(w * eps0 * 4.0 * tan_delta * InnerProduct(E, E).real,
                             lossy.mesh, definedon=lossy.mesh.Materials('ceramic'))
        print(f"{f / 1e9:8.3f} {z0[k].real:18.5f} {zpar[k].real:15.4f} {absorbed:10.4f}")
    return results


if __name__ == "__main__":
    main()
