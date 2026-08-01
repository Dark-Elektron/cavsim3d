"""Rectangular waveguide S-parameters — a clean, matched, single-mode line.

A straight length of rectangular waveguide excited on its fundamental TE10 mode
is a matched transmission line: above the TE10 cutoff it transmits perfectly
(|S21| ~ 1, |S11| ~ 0), and the cavsim3d FOM reproduces this exactly.  This is
the canonical sanity check for the frequency-domain solver and its port modes.

For WR-  guide with broad wall a, the TE10 cutoff is fc = c / (2a).

Run:  python examples/rectangular_waveguide/waveguide_sparameters.py
"""
import tempfile
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry import RectangularWaveguide
from cavsim3d.core.constants import c0

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_rwg_"))


def main(a=0.1, b=0.05, L=0.2, fmin=1.7, fmax=2.9, nsamples=25,
         order=2, maxh=0.02, show=True):
    fc = c0 / (2 * a) / 1e9          # TE10 cutoff [GHz]
    print(f"TE10 cutoff fc = {fc:.3f} GHz  (single-mode band ~ {fc:.2f}-{2*fc:.2f} GHz)")

    proj = EMProject(name="rwg_sparams", base_dir=str(WORK), overwrite=True)
    proj.geometry = RectangularWaveguide(a=a, L=L, b=b, maxh=maxh)
    proj.fds.solve(fmin=fmin, fmax=fmax, nsamples=nsamples,
                   config=dict(order=order, nportmodes=1,
                               solver_type="direct", store_snapshots=False))

    S = proj.fds._S_matrix
    freqs = proj.fds.frequencies
    s11 = np.abs(S[:, 0, 0])
    s21 = np.abs(S[:, 1, 0])
    print(f"\n{'f [GHz]':>8} {'|S11|':>8} {'|S21|':>8}")
    for k in range(0, len(freqs), max(1, len(freqs)//8)):
        print(f"{freqs[k]/1e9:8.2f} {s11[k]:8.4f} {s21[k]:8.4f}")
    print(f"\nabove-cutoff transmission: mean |S21| = {s21.mean():.4f} "
          f"(ideal 1.0), max |S11| = {s11.max():.4f} (ideal 0.0)")

    if show:
        import matplotlib.pyplot as plt
        fig, ax = proj.fds.fom.plot_s(params=["1(1)1(1)", "2(1)1(1)"],
                                      plot_type="db", figsize=(9, 5))
        ax.axvline(fc, ls=":", color="grey", label=f"TE10 cutoff {fc:.2f} GHz")
        ax.set_ylim(-60, 3)
        ax.set_title("Rectangular waveguide S-parameters (matched TE10 line)")
        ax.legend()
        out = Path(__file__).parent / "waveguide_sparameters.png"
        plt.savefig(out, dpi=130)
        print(f"Saved plot -> {out}")
        plt.show()

    return proj


if __name__ == "__main__":
    main()
