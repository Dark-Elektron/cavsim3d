"""
Example: Multi-domain concatenation workflow.

Demonstrates:
1. Creating a split (multi-domain) rectangular waveguide geometry
2. FOM: automatic cascade via FrequencyDomainSolver.solve()
3. ROM: automatic concatenation via ModelOrderReduction.solve()
4. Comparison of full-model S-parameters from both methods

For a waveguide split into N cells, the solver detects the compound
structure and automatically cascades the per-domain S-matrices to
produce the full-model 2-port result.
"""

import numpy as np
import matplotlib.pyplot as plt

from geometry.importers import OCCImporter
from geometry.primitives import RectangularWaveguide
from solvers.frequency_domain import FrequencyDomainSolver
from rom.reduction import ModelOrderReduction
from analytical.rectangular_waveguide import RWGAnalytical


def main():
    print("=" * 60)
    print("Multi-Domain Concatenation Example")
    print("=" * 60)

    # =========================
    # 1. Geometry — split waveguide
    # =========================
    # We use a normal RectangularWaveguide but create a split version
    # by importing a CAD file and adding splitting planes.
    # For demonstration, we'll use the CAD file in rwg_step_split/.

    a = 100e-3   # 100 mm width
    L = 200e-3   # 200 mm total length
    maxh = 0.04

    # --- Option A: Use CAD file with splitting planes ---
    import os
    step_path = os.path.join(os.path.dirname(__file__), 'rwg_step_split', 'rectangular_waveguide.step')
    if os.path.exists(step_path):
        print("\n1. Loading geometry with splitting plane...")
        geom = OCCImporter(step_path, unit='mm', auto_build=False, maxh=maxh)
        # Split at the midpoint
        geom.add_splitting_plane_at_z(L / 2)
        geom.split()
        geom.finalize(maxh=maxh)
    else:
        print("\n1. CAD file not found, using single-domain geometry instead.")
        print("   To test multi-domain, place a STEP file at:")
        print(f"   {step_path}")
        geom = RectangularWaveguide(a=a, L=L, maxh=maxh)

    # =========================
    # 2. FOM — Full Order Model
    # =========================
    print("\n2. Running full-order solver...")

    fds = FrequencyDomainSolver(geom, order=3)
    fds.assemble_matrices(nmodes=1)
    fds.print_info()

    # For compound structures, solve() automatically cascades
    results_fom = fds.solve(fmin=1.5, fmax=3.0, nsamples=30, store_snapshots=True)

    print(f"\n   After solve:")
    print(f"   External ports: {fds.ports}")
    print(f"   Z-matrix shape: {results_fom['Z'].shape}")
    print(f"   Number of domains: {fds.n_domains}")
    if fds.is_compound:
        print(f"   All ports (including internal): {fds.all_ports}")

    # =========================
    # 3. ROM — Reduced Order Model
    # =========================
    print("\n3. Building and solving ROM...")

    rom = ModelOrderReduction(fds)
    rom.reduce(tol=1e-6)
    rom.print_info()

    results_rom = rom.solve(fmin=1.5, fmax=3.0, nsamples=100)

    # =========================
    # 4. Analytical reference (full-length waveguide)
    # =========================
    print("\n4. Computing analytical reference...")

    analytical = RWGAnalytical(a=a, L=L)
    frequencies = np.linspace(1.5, 3.0, 200) * 1e9

    # =========================
    # 5. Comparison plot
    # =========================
    print("\n5. Generating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    freq_ghz_fom = fds.frequencies / 1e9
    freq_ghz_rom = rom.frequencies / 1e9
    freq_ghz_ana = frequencies / 1e9

    # S11
    ax = axes[0, 0]
    S_ana = analytical.s_parameters(frequencies)
    ax.plot(freq_ghz_ana, 20 * np.log10(np.abs(S_ana['S11']) + 1e-15),
            '-', label='Analytical', linewidth=2)
    ax.plot(freq_ghz_fom, fds.get_s_db(1, 1), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_s_db(1, 1), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S11| (dB)')
    ax.set_title('S11 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S21
    ax = axes[0, 1]
    ax.plot(freq_ghz_ana, 20 * np.log10(np.abs(S_ana['S21']) + 1e-15),
            '-', label='Analytical', linewidth=2)
    ax.plot(freq_ghz_fom, fds.get_s_db(1, 2), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_s_db(1, 2), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S21| (dB)')
    ax.set_title('S21 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z11
    ax = axes[1, 0]
    Z_ana = analytical.z_parameters(frequencies)
    ax.plot(freq_ghz_ana, 20 * np.log10(np.abs(Z_ana['Z11']) + 1e-15),
            '-', label='Analytical', linewidth=2)
    ax.plot(freq_ghz_fom, fds.get_z_db(1, 1), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_z_db(1, 1), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|Z11| (dB)')
    ax.set_title('Z11 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z21
    ax = axes[1, 1]
    ax.plot(freq_ghz_ana, 20 * np.log10(np.abs(Z_ana['Z21']) + 1e-15),
            '-', label='Analytical', linewidth=2)
    ax.plot(freq_ghz_fom, fds.get_z_db(1, 2), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_z_db(1, 2), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|Z21| (dB)')
    ax.set_title('Z21 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Domain Concatenation: FOM vs ROM vs Analytical', fontsize=14)
    plt.tight_layout()
    plt.savefig('concatenation_comparison.png', dpi=100, bbox_inches='tight')
    print("   Saved: concatenation_comparison.png")

    # =========================
    # 6. Per-domain access
    # =========================
    if fds.is_compound:
        print("\n6. Per-domain results (for debugging)...")
        for domain in fds.domains:
            domain_results = fds.get_domain_results(domain)
            print(f"\n   Domain: {domain}")
            print(f"     Ports: {domain_results['ports']}")
            print(f"     Z shape: {domain_results['Z'].shape}")

    # =========================
    # 7. Error analysis
    # =========================
    print("\n7. Error analysis (ROM vs FOM)...")
    errors = rom.compute_error(fds)
    for key, err in errors.items():
        print(f"   {key}: {err:.4e}")

    print("\n" + "=" * 60)
    print("Concatenation example complete!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
