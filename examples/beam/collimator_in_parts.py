"""The collimator in two parts: per-part FOMs joined through their S~.

The collimator of ``collimator_impedance.py`` is built from two ``taper``
parts that meet in the middle of the 50 mm flat.  Glued into one mesh, the
two solids are two domains, and ``solve(per_domain=True)`` solves each on its
own, the face between them being a port of both.  With a beam, every part
gets its generalised scattering matrix ``s_tilde``; ``foms.concatenate()``
joins them at the cut (CSC-BEAM, T. Flisgen et al., PRAB 23, 034601 (2020)):
the waves leaving one face enter the other, the beam current is the same in
both parts and their beam voltages add.  No full-order matrices are built,
so the join is immediate; it holds the frequencies of the full-order solve.

The same mesh solved in one piece (``per_domain=False``) gives the reference.
The two agree as far as the modes carried at the cut describe the field
there: the cut is in the 50 mm flat, where the beam's TM0n near field
decays quickly.

Run:  python examples/beam/collimator_in_parts.py
"""
import tempfile
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_beam_"))

# the collimator as two tapers (mm): 100 mm bore -> 50 mm, cone 800 mm long
DOWN = dict(R_left=100, R_right=50, L=1500, straight_left=550, straight_right=150)
UP = dict(R_left=50, R_right=100, L=1500, straight_left=150, straight_right=550)


def build(name, maxh):
    proj = EMProject(name=name, base_dir=str(WORK), overwrite=True)
    proj.create_primitive('taper', name='down', **DOWN)
    proj.create_primitive('taper', name='up', **UP)
    proj.generate_mesh(maxh=maxh, curve_order=4)        # glued: one mesh, two domains
    proj.add_beam('beam')
    return proj


def main(fmin=0.1, fmax=1.1, nsamples=6, order=3, maxh=0.035, nportmodes=3, show=True):
    cfg = dict(fmin=fmin, fmax=fmax, nsamples=nsamples, order=order, nportmodes=nportmodes)

    parts = build("collimator_parts", maxh)
    parts.fds.solve(**cfg, per_domain=True, global_method=None)
    for fom in parts.fds.foms:                          # each part on its own
        print(f"part {fom.domain}: Z_par = {np.round(fom.beam_impedance()[:2], 3)} ...")
    joined = parts.fds.foms.concatenate()               # S~ joined at the cut
    z_join = joined.beam_impedance()

    whole = build("collimator_whole", maxh)
    whole.fds.solve(**cfg, per_domain=False)            # the same mesh in one piece
    z_whole = whole.fds.fom.beam_impedance()

    print(f"\n{'f [GHz]':>8} {'joined':>22} {'one piece':>22}  rel. difference")
    for f, a, b in zip(joined.frequencies, z_join, z_whole):
        print(f"{f / 1e9:8.3f} {a.real:10.4f}{a.imag:+10.4f}j {b.real:10.4f}{b.imag:+10.4f}j"
              f"  {abs(a - b) / abs(b):.1e}")
    # each part alone has a large real part (the beam's own field changes
    # from one pipe radius to the other); joined, the two cancel

    if show:
        import matplotlib.pyplot as plt
        fig, ax = whole.fds.fom.plot_beam_impedance(label="one piece", linewidth=3,
                                                    alpha=0.4)
        joined.plot_beam_impedance(ax=ax, label="parts joined", marker="o",
                                   linestyle="none")
        ax.set_title("Collimator: two parts joined through S~ vs one piece")
        plt.show()
    return parts, whole


if __name__ == "__main__":
    main()
