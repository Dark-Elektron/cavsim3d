"""Microstrip line with quasi-TEM ports — cavsim3d vs CST.

Reproduces the CST ``microstrip_line`` reference model (FR-4 substrate, PEC
strip and ground plane) and exercises the quasi-TEM port boundary condition:

  1. Port-mode check  — effective permittivity eps_eff and power-voltage line
     impedance Z_PV of the fundamental quasi-TEM mode, vs the CST port info.
     This is the VALIDATED deliverable: eps_eff matches analytic microstrip
     theory (~3.3) and the CST ballpark (3.55-3.79); Z_PV ~ 50 ohm (design).
  2. Full S-parameters — a coarse 3D driven sweep, overlaid on the CST S-params.

Notes / caveats
---------------
* The CST model uses lossy FR-4 / copper while cavsim3d here is lossless PEC —
  expect eps_eff real part and |S| trends to differ by small, loss-driven
  offsets.
* The S-parameters do NOT yet match CST cleanly.  A closed rectangular
  waveguide through this same FOM gives |S21|=1.000 flat, so the port machinery
  is sound; but a microstrip is an OPEN structure whose finite air box acts as
  a resonator (PMC-box resonances with natural walls, or a low-frequency cutoff
  with PEC walls), corrupting the 2-port transmission.  A clean microstrip S
  needs absorbing/radiation boundaries on the air box (or a de-embedded /
  tuned box) — a solver-modelling addition, separate from the quasi-TEM port
  boundary condition itself.  The port-mode results in step 1 are unaffected.

Run:  python examples/microstrip/microstrip.py
"""
import tempfile
from pathlib import Path

import numpy as np

# Import order matters: core/em_project before rom.* (see CLAUDE.md).
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry import MicrostripLine

from cst_compare import cst_port_info  # noqa: E402  (local helper)

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_microstrip_"))


def main(nsamples: int = 41, order: int = 2, maxh: float = 1.5e-3, show: bool = True):
    geo = MicrostripLine(maxh=maxh)
    proj = EMProject(name="microstrip", base_dir=str(WORK), overwrite=True)
    proj.geometry = geo

    # ---- full driven sweep with quasi-TEM ports (auto-enabled by the
    #      inhomogeneous substrate+air cross-section; listed here for clarity) ----
    proj.fds.solve(
        fmin=0.1, fmax=10.0, nsamples=nsamples,
        config=dict(order=order, nportmodes=1, qtem_ports=['port1', 'port2'],
                    solver_type='direct', store_snapshots=False),
    )

    ps = proj.fds.port_solver
    print("\n=== Quasi-TEM port modes (cavsim3d, at f_max) ===")
    cst = cst_port_info()
    for p in ('port1', 'port2'):
        ee = ps.port_eps_eff.get(p, {}).get(0)
        zpv = ps.port_line_impedance.get(p, {}).get(0)
        beta = ps.port_beta.get(p, {}).get(0)
        print(f"  {p}: eps_eff={ee.real:6.3f}  beta={beta.real:8.2f} rad/m  "
              f"Z_PV={zpv.real:6.2f} ohm")
    if cst:
        f, ee_cst = cst['eps_eff']
        f, z_cst = cst['line_impedance']
        print(f"  CST @10GHz : eps_eff={ee_cst[-1].real:6.3f}  "
              f"Z_line={z_cst[-1].real:6.2f} ohm")
        print(f"  CST @1GHz  : eps_eff={ee_cst[100].real:6.3f}  "
              f"Z_line={z_cst[100].real:6.2f} ohm")

    # ---- S-parameters vs CST ----
    fom = proj.fds.fom
    freqs = proj.fds.frequencies
    S = proj.fds._S_matrix
    print(f"\nSolved {len(freqs)} points; |S11|(f_max)="
          f"{abs(S[-1,0,0]):.4f}, |S21|(f_max)={abs(S[-1,1,0]):.4f}")

    if show:
        import matplotlib.pyplot as plt
        from cst_compare import overlay_s_parameters
        overlay_s_parameters(freqs, S)
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / "microstrip_s_vs_cst.png", dpi=130)
        print(f"  Saved plot -> {Path(__file__).parent / 'microstrip_s_vs_cst.png'}")
        plt.show()

    return proj


if __name__ == "__main__":
    main()
