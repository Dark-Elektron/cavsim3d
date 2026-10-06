"""Three pillbox cells from one: repeated and imported parts joined with the beam.

A pillbox cell (radius 100 mm, 100 mm long) with 30 mm of beam pipe (radius
40 mm) on each side is added three times (``n=3``).  Repeated, the cell is a
coupled part: it is solved once, on its own mesh, with the beam where it runs
through it, and ``proj.fds.foms.concatenate()`` joins the three copies through
their generalised scattering matrices ``s_tilde``.  A copy further along the
axis sees the beam later; its beam column carries the phase of its position.

The same three cells glued into one mesh and solved in one piece give the
reference.  The cut between two copies lies 30 mm from each cavity, where the
beam's TM01 near field has not decayed: it is carried as a port mode
(``nportmodes=3``: TE11 in two polarisations and TM01 of the pipe).

Then a cell solved in another project, without a beam, is imported three times
(``proj.import_project(..., n=3)``): its beam columns are computed in the
importing project from the port solutions stored in the cell's project, which
is never written.

Run:  python examples/beam/cells_coupled.py
"""
import tempfile
import time
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_beam_"))
# cell length, cavity radius, pipe radius, drift, pipe length (mm)
CELL = dict(n_cells=1, dims=[100, 100, 40, 0, 30], beampipe='both')


def main(fmin=1.10, fmax=1.20, nsamples=21, order=3, maxh=0.02, show=True):
    cfg = dict(fmin=fmin, fmax=fmax, nsamples=nsamples, nportmodes=3, order=order)

    # 1. one cell, three copies, joined with the beam
    t0 = time.time()
    chain = EMProject(name="cells_chain", base_dir=str(WORK), overwrite=True)
    chain.create_primitive('pillbox', name='cell', n=3, **CELL)
    chain.add_beam('beam')                         # on the axis
    chain.generate_mesh(maxh=maxh)                 # curve order 4 with a beam
    chain.fds.solve(**cfg)                         # the cell once
    joined = chain.fds.foms.concatenate()          # the copies through their S~
    z_chain = joined.beam_impedance()
    print(f"chain: {time.time() - t0:.0f} s, S~ labels {joined.tilde_labels[1]}")

    # 2. the same three cells glued into one mesh, solved in one piece
    t0 = time.time()
    whole = EMProject(name="cells_whole", base_dir=str(WORK), overwrite=True)
    for name in ('c1', 'c2', 'c3'):
        whole.create_primitive('pillbox', name=name, **CELL)
    whole.add_beam('beam')
    whole.generate_mesh(maxh=maxh)
    whole.fds.solve(**cfg, per_domain=False)
    z_whole = whole.fds.fom.beam_impedance()
    print(f"one piece: {time.time() - t0:.0f} s")

    # 3. a cell solved in another project without a beam, imported three times
    cell = EMProject(name="cell_alone", base_dir=str(WORK), overwrite=True)
    cell.create_primitive('pillbox', name='cell', **CELL)
    cell.generate_mesh(maxh=maxh, curve_order=4)
    cell.fds.solve(**cfg)
    imported = EMProject(name="cells_imported", base_dir=str(WORK), overwrite=True)
    imported.import_project(WORK / "cell_alone", name="cell", n=3)
    imported.add_beam('beam')
    imported.fds.solve(**cfg)                      # beam columns computed here
    z_imported = imported.fds.foms.concatenate().beam_impedance()

    print(f"\n{'f [GHz]':>8} {'chain of copies':>22} {'one piece':>22} {'imported':>22}")
    for f, a, b, c in zip(joined.frequencies, z_chain, z_whole, z_imported):
        print(f"{f / 1e9:8.4f} {a.real:10.3f}{a.imag:+11.3f}j {b.real:10.3f}{b.imag:+11.3f}j"
              f" {c.real:10.3f}{c.imag:+11.3f}j")
    rel = np.abs(z_chain - z_whole) / np.abs(z_whole)
    print(f"\nchain vs one piece: median {np.median(rel):.1e}, max {rel.max():.1e}")
    print(f"imported vs chain:  max |difference| {np.abs(z_imported - z_chain).max():.1e} Ohm")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = whole.fds.fom.plot_beam_impedance(label="one piece", linewidth=3,
                                                    alpha=0.4)
        joined.plot_beam_impedance(ax=ax, label="three copies joined", marker="o",
                                   linestyle="none")
        ax.set_title("Three pillbox cells: copies joined through S~ vs one piece")
        plt.show()
    return chain, whole, imported


if __name__ == "__main__":
    main()
