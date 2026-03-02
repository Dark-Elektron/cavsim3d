"""
Example: Complete rectangular waveguide analysis workflow.

Demonstrates:
1. Geometry creation
2. Frequency-domain analysis (using new unified API)
3. Model order reduction
4. Comparison with analytical solution
"""

import numpy as np
import matplotlib.pyplot as plt

# Import cavsim modules
from geometry.primitives import RectangularWaveguide
from solvers.frequency_domain import FrequencyDomainSolver
from rom.reduction import ModelOrderReduction
from analytical.rectangular_waveguide import RWGAnalytical


def main():
    print("=" * 60)
    print("Rectangular Waveguide Analysis")
    print("=" * 60)

    # =========================
    # 1. Define Geometry
    # =========================
    print("\n1. Creating geometry...")

    a = 100e-3  # Width: 100 mm
    L = 200e-3  # Length: 200 mm
    maxh = 0.04  # Mesh size

    rwg = RectangularWaveguide(a=a, L=L, maxh=maxh)
    print(f"   Dimensions: a={a * 1e3:.0f}mm, b={rwg.b * 1e3:.0f}mm, L={L * 1e3:.0f}mm")
    print(f"   Cutoff frequency (TE10): {rwg.cutoff_frequency_TE10 / 1e9:.3f} GHz")
    print(f"   Mesh DOFs: ~{rwg.mesh.nv} vertices")

    # =========================
    # 2. Analytical Solution
    # =========================
    print("\n2. Computing analytical solution...")

    analytical = RWGAnalytical(a=a, L=L)
    frequencies = np.linspace(1.5, 3.0, 100) * 1e9
    Z_analytical = analytical.z_parameters(frequencies)

    print(f"   Computed Z-parameters at {len(frequencies)} frequency points")

    # =========================
    # 3. Full-Order Numerical Solution
    # =========================
    print("\n3. Running full-order frequency sweep...")

    solver = FrequencyDomainSolver(rwg, order=3)
    solver.assemble_matrices(nmodes=1)

    solver.print_info()

    results_fom = solver.solve(fmin=1.5, fmax=3.0, nsamples=30, store_snapshots=True)
    print(f"   Solved at {len(solver.frequencies)} frequency points")
    print(f"   Ports: {solver.ports}")

    # =========================
    # 4. Model Order Reduction
    # =========================
    print("\n4. Building reduced-order model...")

    rom = ModelOrderReduction(solver)
    rom.reduce(tol=1e-6)

    rom.print_info()

    # Solve ROM over finer frequency grid
    results_rom = rom.solve(fmin=1.5, fmax=3.0, nsamples=100)
    print(f"   Solved ROM at {len(rom.frequencies)} frequency points")

    # =========================
    # 5. Comparison and Plotting
    # =========================
    print("\n5. Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    freq_ghz_fom = solver.frequencies / 1e9
    freq_ghz_rom = rom.frequencies / 1e9
    freq_ghz_ana = frequencies / 1e9

    # Z11 Magnitude
    ax = axes[0, 0]
    ax.plot(freq_ghz_ana, 20 * np.log10(np.abs(Z_analytical['Z11']) + 1e-15),
            '-', label='Analytical', linewidth=2)
    ax.plot(freq_ghz_fom, solver.get_z_db(1, 1), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_z_db(1, 1), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|Z11| (dB)')
    ax.set_title('Z11 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Z21 Magnitude
    ax = axes[0, 1]
    ax.plot(freq_ghz_ana, 20 * np.log10(np.abs(Z_analytical['Z21']) + 1e-15),
            '-', label='Analytical', linewidth=2)
    ax.plot(freq_ghz_fom, solver.get_z_db(1, 2), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_z_db(1, 2), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|Z21| (dB)')
    ax.set_title('Z21 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S11 Magnitude
    ax = axes[1, 0]
    ax.plot(freq_ghz_fom, solver.get_s_db(1, 1), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_s_db(1, 1), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S11| (dB)')
    ax.set_title('S11 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # S21 Magnitude
    ax = axes[1, 1]
    ax.plot(freq_ghz_fom, solver.get_s_db(1, 2), 'o', label='FOM', markersize=4)
    ax.plot(freq_ghz_rom, rom.get_s_db(1, 2), '--', label='ROM', linewidth=1.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S21| (dB)')
    ax.set_title('S21 Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Rectangular Waveguide: FOM vs ROM vs Analytical', fontsize=14)
    plt.tight_layout()
    plt.savefig('rwg_z_parameters.png', dpi=100, bbox_inches='tight')
    print("   Saved: rwg_z_parameters.png")

    # Singular value decay
    rom.plot_singular_values()
    plt.savefig('rwg_singular_values.png', dpi=100, bbox_inches='tight')
    print("   Saved: rwg_singular_values.png")

    # =========================
    # 6. Error Analysis
    # =========================
    print("\n6. Error analysis...")

    # ROM vs FOM
    errors = rom.compute_error(solver)
    print(f"   ROM vs FOM errors:")
    for key, err in errors.items():
        print(f"      {key}: {err:.4e}")

    # Analytical comparison (interpolated to FOM frequencies)
    Z_ana_interp = analytical.z_parameters(solver.frequencies)
    Z11_fom = solver.get_param('Z', '1(1)1(1)')
    Z11_ana = Z_ana_interp['Z11']

    rel_error = np.abs(Z11_fom - Z11_ana) / (np.abs(Z11_ana) + 1e-15)
    print(f"\n   FOM vs Analytical (Z11):")
    print(f"      Mean relative error: {np.mean(rel_error) * 100:.2f}%")
    print(f"      Max relative error:  {np.max(rel_error) * 100:.2f}%")

    # =========================
    # 7. Eigenfrequency Comparison
    # =========================
    print("\n7. Eigenfrequency comparison...")

    freqs_fom = solver.get_resonant_frequencies() / 1e9
    freqs_rom = rom.get_resonant_frequencies() / 1e9

    print(f"\n   FOM eigenfrequencies (GHz, first 10):")
    for i, f in enumerate(sorted(freqs_fom)[:10]):
        print(f"      Mode {i + 1}: {f:.4f} GHz")

    print(f"\n   ROM eigenfrequencies (GHz, first 10):")
    for i, f in enumerate(sorted(freqs_rom)[:10]):
        print(f"      Mode {i + 1}: {f:.4f} GHz")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()