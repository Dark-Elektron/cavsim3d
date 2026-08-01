"""Microstrip characteristic impedance & effective permittivity vs strip width.

Sweeps the strip width w and extracts, from the quasi-TEM port mode, the
effective permittivity eps_eff and the power-voltage characteristic (line)
impedance Z_PV.  Wider strips -> lower impedance and higher eps_eff (more field
confined in the substrate).  Marks the ~50 ohm design point.

This exercises the *validated* part of the quasi-TEM port (the mode physics);
see microstrip.py for the S-parameter comparison and its open-box caveat.

Run:  python examples/microstrip/microstrip_impedance_sweep.py
"""
import tempfile
from pathlib import Path

import numpy as np

# cavsim3d (pythonocc) must be imported before ngsolve/netgen
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry import MicrostripLine
from cavsim3d.core.constants import Z0

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_ms_zsweep_"))


def port_mode(width, f0=5e9, maxh=1.5e-3, order=2):
    """Solve the fundamental quasi-TEM port mode for a given strip width.

    Returns (eps_eff, Z_PV, wave_impedance).
    """
    geo = MicrostripLine(w=width, maxh=maxh)
    proj = EMProject(name=f"ms_w{width*1e3:.1f}", base_dir=str(WORK), overwrite=True)
    proj.geometry = geo
    # A 2-point solve is enough to trigger the port eigenmode at k0 = 2*pi*f0/c;
    # we only read the port-mode quantities, not the sweep.
    proj.fds.solve(fmin=f0/1e9, fmax=f0/1e9, nsamples=2,
                   config=dict(order=order, nportmodes=1,
                               qtem_ports=["port1", "port2"],
                               solver_type="direct", store_snapshots=False))
    ps = proj.fds.port_solver
    ee = ps.port_eps_eff["port1"][0].real
    zpv = ps.port_line_impedance["port1"][0].real
    return ee, zpv, Z0 / np.sqrt(ee)


def main(show=True):
    widths = np.array([1.0, 2.0, 3.1, 4.5, 6.0, 8.0]) * 1e-3
    eps_eff, zpv, zwave = np.array([port_mode(w) for w in widths]).T

    print(f"\n{'w [mm]':>7} {'eps_eff':>9} {'Z_PV [ohm]':>11} {'Z_wave [ohm]':>13}")
    print("-" * 44)
    for w, ee, z, zw in zip(widths, eps_eff, zpv, zwave):
        print(f"{w*1e3:7.2f} {ee:9.3f} {z:11.2f} {zw:13.1f}")

    # design point closest to 50 ohm
    i50 = int(np.argmin(np.abs(zpv - 50.0)))
    print(f"\n~50 ohm at w ~ {widths[i50]*1e3:.2f} mm  (Z_PV = {zpv[i50]:.1f} ohm)")

    if show:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        c1 = "tab:blue"
        ax1.plot(widths*1e3, zpv, "o-", color=c1, label="Z_PV (line impedance)")
        ax1.axhline(50, ls=":", color="grey", label="50 ohm")
        ax1.set_xlabel("strip width w [mm]")
        ax1.set_ylabel("Z_PV [ohm]", color=c1)
        ax1.tick_params(axis="y", labelcolor=c1)
        ax2 = ax1.twinx()
        c2 = "tab:red"
        ax2.plot(widths*1e3, eps_eff, "s--", color=c2, label="eps_eff")
        ax2.set_ylabel("eps_eff", color=c2)
        ax2.tick_params(axis="y", labelcolor=c2)
        ax1.set_title("Microstrip (eps_r=4.3, h=1.6mm): impedance & eps_eff vs width")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        out = Path(__file__).parent / "microstrip_impedance_sweep.png"
        plt.savefig(out, dpi=130)
        print(f"Saved plot -> {out}")
        plt.show()


if __name__ == "__main__":
    main()
