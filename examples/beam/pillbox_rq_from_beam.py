"""A beam added to a solved pillbox; R/Q from its beam impedance.

1. The pillbox (radius 100 mm, length 100 mm, beam pipes of radius 30 mm) is
   solved without a beam: S-parameters, and from the full-order operator its
   TM010 mode with ``R/Q = V^2 / (w U)``.
2. ``proj.add_beam()`` and the same ``solve()`` again: the port results are
   kept as they are, only the beam columns are solved (with the stored port
   solutions for the beam voltage of the port waves).
3. Below the cut-off of the beam pipes, with every port mode open (magnetic
   walls, as in the eigenproblem), a lossless mode is a pole of the beam
   impedance:  Z_par(w) ~ j (R/Q) w0 / (4 (w0 - w)).  ``R/Q`` read from the
   beam impedance close to the resonance is the eigenmode's: two independent
   routes to the same number.

Run:  python examples/beam/pillbox_rq_from_beam.py
"""
import tempfile
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_beam_"))


def main(maxh=0.015, order=3, offsets=(-0.02, -0.005, -0.001, 0.001, 0.005, 0.02)):
    proj = EMProject(name="pillbox_beam", base_dir=str(WORK), overwrite=True)
    # dims: cell length, cavity radius, aperture / pipe radius, drift, pipe length (mm)
    proj.create_primitive('pillbox', name='cavity', n_cells=1, dims=[100, 100, 30, 0, 100],
                          beampipe='both')
    proj.generate_mesh(maxh=maxh, curve_order=4)

    # 1. ports only; TM010 and its R/Q from the full-order operator
    cfg = dict(fmin=1.0, fmax=1.3, nsamples=4, nportmodes=3, order=order)
    proj.fds.solve(**cfg)
    f0 = proj.fds.fom.get_resonant_frequencies(n_modes=1)[0]
    rq = proj.fds.fom.get_rq(0)['RQ']
    print(f"TM010: f0 = {f0 / 1e9:.5f} GHz, R/Q = {rq:.2f} Ohm (eigenmode)")

    # 2. a beam on the axis, added to the solved model
    S_before = proj.fds.fom._S_matrix.copy()
    proj.add_beam('beam')
    proj.fds.solve(**cfg)                     # the beam columns only
    assert np.array_equal(proj.fds.fom._S_matrix, S_before)
    print(f"beam added: S unchanged, S~ labels {proj.fds.fom.tilde_labels[1]}")

    # 3. R/Q from the pole of Z_par (open ports), close to the resonance
    w0 = 2 * np.pi * f0
    print(f"\n{'f/f0 - 1':>9} {'Z_par open [Ohm]':>22} {'R/Q from Z_par':>15}")
    for off in offsets:
        f = f0 * (1 + off)
        proj.fds.solve(**dict(cfg, fmin=f / 1e9, fmax=f / 1e9, nsamples=1))
        z = proj.fds.fom.beam_impedance(ports='open')[0]
        w = 2 * np.pi * f
        rq_beam = (z * 4 * (w0 - w) / (1j * w0)).real
        print(f"{off:+9.3f} {z.real:10.4f}{z.imag:+12.1f}j {rq_beam:15.2f}")
    print(f"(eigenmode: {rq:.2f} Ohm; the closer to f0, the smaller the other modes' share)")
    return proj


if __name__ == "__main__":
    main()
