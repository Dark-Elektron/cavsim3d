"""FOM -> ROM: reduce a full-order model and sweep it cheaply.

The pipeline is staged: solve the full-order model (FOM) on a handful of
frequencies, build a reduced-order model (ROM) from those snapshots, then sweep
the ROM on a dense grid almost for free.  The ROM interpolates/extrapolates the
S-parameters within its training band and passes through the FOM snapshots.

Uses a rectangular waveguide (a clean, matched single-mode line) so the result
is easy to read: |S21| ~ 1, |S11| ~ 0.

Run:  python examples/rom_reduction/fom_to_rom.py
"""
import time
import tempfile
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry import RectangularWaveguide

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_rom_"))


def main(a=0.1, b=0.05, L=0.2, fmin=1.7, fmax=2.9,
         n_fom=6, n_rom=400, tol=1e-9, order=2, maxh=0.02, show=True):
    proj = EMProject(name="rom_demo", base_dir=str(WORK), overwrite=True)
    proj.geometry = RectangularWaveguide(a=a, L=L, b=b, maxh=maxh)

    # --- STAGE 1: full-order model on a few frequencies ---
    t0 = time.time()
    proj.fds.solve(fmin=fmin, fmax=fmax, nsamples=n_fom,
                   config=dict(order=order, nportmodes=1,
                               solver_type="direct", store_snapshots=True))
    t_fom = time.time() - t0
    fom = proj.fds.fom
    print(f"FOM: {n_fom} samples in {t_fom:.2f}s  "
          f"({1e3*t_fom/n_fom:.1f} ms/point)")

    # --- STAGE 2: reduce, then sweep the ROM densely ---
    rom = fom.reduce(tol=tol)
    t0 = time.time()
    rom.solve(fmin=fmin, fmax=fmax, nsamples=n_rom)
    t_rom = time.time() - t0
    print(f"ROM: {n_rom} samples in {t_rom:.2f}s  "
          f"({1e3*t_rom/n_rom:.2f} ms/point)  -> "
          f"~{(t_fom/n_fom)/(t_rom/n_rom):.0f}x cheaper per point")

    Sr = rom.S_dict
    fr = np.asarray(Sr["frequencies"]) / 1e9
    s21 = np.abs(Sr["2(1)1(1)"])
    print(f"ROM sweep: mean |S21| = {s21.mean():.4f} over {n_rom} points")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = fom.plot_s(params=["1(1)1(1)", "2(1)1(1)"], plot_type="db",
                             figsize=(9, 5), label="FOM", marker="o", linestyle="")
        rom.plot_s(params=["1(1)1(1)", "2(1)1(1)"], plot_type="db", ax=ax,
                   label="ROM")
        ax.set_ylim(-70, 3)
        ax.set_title(f"FOM ({n_fom} pts) vs ROM ({n_rom} pts) — rectangular waveguide")
        ax.legend()
        out = Path(__file__).parent / "fom_to_rom.png"
        plt.savefig(out, dpi=130)
        print(f"Saved plot -> {out}")
        plt.show()

    return proj


if __name__ == "__main__":
    main()
