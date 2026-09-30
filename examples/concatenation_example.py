"""
Example: Multi-domain concatenation workflow.

Demonstrates:
1. Importing a rectangular waveguide (STEP) and splitting it in two along z
2. FOM: each domain solved on its own, ``proj.fds.solve()``
3. ROM: each domain reduced, and the reduced models joined through their
   ports, ``proj.fds.foms.reduce(tol).concatenate()``
4. Comparison of the joined model with the analytical full-length guide

The project is written to ./simulations, the plot to the working directory.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.analytical.rectangular_waveguide import RWGAnalytical


def main():
    print("=" * 60)
    print("Multi-Domain Concatenation Example")
    print("=" * 60)

    a = 100e-3   # 100 mm width
    L = 200e-3   # 200 mm total length
    maxh = 0.04
    # TE20 and TE01 of this guide start at 3.0 GHz; the joins carry TE10 only
    fmin, fmax = 1.5, 2.9

    # =========================
    # 1. Geometry — split waveguide
    # =========================
    print("\n1. Loading the geometry and splitting it at mid-length...")
    step_path = os.path.join(os.path.dirname(__file__), 'rwg_step', 'rectangular_waveguide.step')
    proj = EMProject(name="concatenation_example", base_dir="./simulations", overwrite=True)
    geo = proj.import_geometry(step_path, name="guide", unit="m", auto_build=False)
    geo.add_splitting_plane_at_z(L / 2)
    geo.split()
    proj.generate_mesh(maxh=maxh)
    print(f"   Domains: {geo.domains}")

    # =========================
    # 2. FOM — each domain on its own
    # =========================
    print("\n2. Running the full-order solver on each domain...")
    proj.fds.solve(fmin=fmin, fmax=fmax, nsamples=30, order=3)
    print(f"   {proj.fds.foms}")

    # =========================
    # 3. ROM — reduce each domain, then join
    # =========================
    print("\n3. Reducing each domain and joining the reduced models...")
    roms = proj.fds.foms.reduce(tol=1e-6)
    concat = roms.concatenate()
    res = concat.solve(fmin=fmin, fmax=fmax, nsamples=200)

    # =========================
    # 4. Analytical reference (full-length waveguide)
    # =========================
    print("\n4. Computing the analytical reference...")
    analytical = RWGAnalytical(a=a, L=L)
    f_ghz = res["frequencies"] / 1e9           # RWGAnalytical takes GHz
    S_ana = analytical.s_parameters(f_ghz)
    Z_ana = analytical.z_parameters(f_ghz)

    # =========================
    # 5. Comparison plot
    # =========================
    print("\n5. Generating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [("S11", res["S"][:, 0, 0], S_ana["S11"]), ("S21", res["S"][:, 1, 0], S_ana["S21"]),
              ("Z11", res["Z"][:, 0, 0], Z_ana["Z11"]), ("Z21", res["Z"][:, 1, 0], Z_ana["Z21"])]
    for ax, (name, joined, exact) in zip(axes.flat, panels):
        ax.plot(f_ghz, 20 * np.log10(np.abs(exact) + 1e-15), '-', label='Analytical', linewidth=2)
        ax.plot(f_ghz, 20 * np.log10(np.abs(joined) + 1e-15), '--', label='Joined ROMs',
                linewidth=1.5)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(f'|{name}| (dB)')
        ax.set_title(f'{name} Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Domain Concatenation: joined ROMs vs Analytical', fontsize=14)
    plt.tight_layout()
    plt.savefig('concatenation_comparison.png', dpi=100, bbox_inches='tight')
    print("   Saved: concatenation_comparison.png")

    # =========================
    # 6. Error against the analytical solution
    # =========================
    print("\n6. Error analysis (joined ROMs vs analytical)...")
    err = np.abs(res["S"][:, 1, 0] - S_ana["S21"])
    print(f"   max |S21 - exact|: {err.max():.2e}")

    print("\n" + "=" * 60)
    print("Concatenation example complete!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
