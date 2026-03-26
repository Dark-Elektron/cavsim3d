import numpy as np
import matplotlib.pyplot as plt
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import CircularWaveguide
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.rom.reduction import ModelOrderReduction
from cavsim3d.analytical.circular_waveguide import CWGAnalytical

# 1. Start project
project_name = 'readme_cwg_concat'
proj = EMProject(name=project_name, base_dir='./simulations', overwrite=True)

# 2. Create assembly with two circular waveguides
print("--- [Snippet 1] Geometry Construction ---")
assembly = proj.create_assembly(main_axis='Z')
radius, L_segment = 50e-3, 100e-3
wg1 = CircularWaveguide(radius=radius, length=L_segment)
wg2 = CircularWaveguide(radius=radius, length=L_segment)

assembly.add("cwg1", wg1)
assembly.add("cwg2", wg2, after="cwg1")
assembly.build()

# 3. Visualization
print("--- [Snippet 2] Visualization ---")
print("Drawing geometry...")
assembly.show() 

print("Generating and showing mesh...")
assembly.generate_mesh(maxh=0.03)
assembly.show('mesh')

# 4. Solve Full Order Model (FOM) for subdomains
print("--- [Snippet 3] FOM Solve ---")
fom_config = {
    'nportmodes': 1,
    'order': 2,
    'fmin': 2.0,
    'fmax': 3.0,
    'nsamples': 11,
    'store_snapshots': True
}
# Per-domain solve for concatenation workflow
proj.fds.solve(config=fom_config, per_domain=True)

# 5. Field & Eigenmode Visualization
print("--- [Snippet 4] Field & Eigenmode Visualization ---")
print("Plotting E-field at 2.5 GHz...")
proj.fds.plot_field(freq_idx=5, component='abs')

print("Calculating resonant frequencies...")
proj.fds.calculate_resonant_modes(n_modes=5)
print("Plotting first eigenmode pattern...")
proj.fds.plot_eigenmode(mode_index=0)

# 6. Model Order Reduction
print("--- [Snippet 5] Model Order Reduction ---")
# Reduce each subdomain independently
roms = proj.fds.foms.reduce(tol=1e-6)

# 7. Concatenation & Validation
print("--- [Snippet 6] Concatenation & Validation ---")
# Concatenate reduced subdomains and solve wideband
cs = roms.concatenate()
rom_result = cs.solve(fmin=2.0, fmax=3.0, nsamples=101)

# Analytical Solution comparison
analytical = CWGAnalytical(radius=radius, length=2 * L_segment)
Z_ana = analytical.z_parameters_TE11(rom_result['frequencies'] / 1e9)

# Plot Z11 comparison
plt.figure(figsize=(10, 6))
plt.plot(rom_result['frequencies'] / 1e9, np.abs(rom_result['Z'][:, 0, 0]), 'r-', label='ROM Concatenation')
plt.plot(rom_result['frequencies'] / 1e9, np.abs(Z_ana['Z11']), 'k--', label='Analytical')
plt.xlabel('Frequency (GHz)')
plt.ylabel('|Z11| (Ohms)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Circular Waveguide Concatenation: ROM vs Analytical')
plt.savefig('cwg_comparison.png')
print("Comparison plot saved to cwg_comparison.png")
print("Verification complete.")
