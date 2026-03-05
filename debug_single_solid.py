
import numpy as np
import scipy.linalg as sl
from geometry.primitives import RectangularWaveguide
from solvers.frequency_domain import FrequencyDomainSolver
from rom.reduction import ModelOrderReduction
from analytical.rectangular_waveguide import RWGAnalytical

def debug_solvers():
    print("=== Debugging Single-Solid Solver ===")
    
    # Setup geometry (100mm x 50mm x 200mm)
    a, b, L = 100e-3, 50e-3, 200e-3
    rwg = RectangularWaveguide(a=a, b=b, L=L, maxh=0.04)
    fc = rwg.cutoff_frequency_TE10
    print(f"TE10 Cutoff: {fc/1e9:.4f} GHz")
    
    # Solver setup
    fds = FrequencyDomainSolver(rwg, order=2)
    fds.assemble_matrices(nportmodes=1)
    
    # Frequency sweep (above cutoff)
    fmin, fmax = 1.1 * fc / 1e9, 1.5 * fc / 1e9
    nf = 10
    
    print(f"Solving FOM...")
    results_fom = fds.solve(fmin, fmax, nf, store_snapshots=True)
    Z11_fom = results_fom['Z'][:, 0, 0]
    
    print(f"Solving ROM...")
    rom = ModelOrderReduction(fds)
    rom.reduce(tol=1e-6)
    results_rom = rom.solve(fmin, fmax, nf)
    Z11_rom = results_rom['Z'][:, 0, 0]
    
    print(f"Solving Concatenated FOM (W=I path)...")
    concat_fom = fds.foms.concatenate()
    results_concat = concat_fom.solve(fmin, fmax, nf)
    Z11_concat = results_concat['Z'][:, 0, 0]
    
    # Analytical solution
    ana = RWGAnalytical(a=a, b=b, L=L)
    Z_ana_dict = ana.z_parameters(fds.frequencies)
    Z11_ana = Z_ana_dict['Z11']
    
    print("\nZ11 Comparison (at middle freq):")
    mid = nf // 2
    f_mid = fds.frequencies[mid] / 1e9
    print(f"Frequency: {f_mid:.4f} GHz")
    print(f"  Analytical: {Z11_ana[mid]:.2f}i")
    print(f"  FOM:        {Z11_fom[mid]:.2f}")
    print(f"  ROM:        {Z11_rom[mid]:.2f}")
    print(f"  Concat-FOM: {Z11_concat[mid]:.2e}")
    
    # Check scaling
    print("\nRelative Error vs Analytical:")
    print(f"  FOM: {np.mean(np.abs(Z11_fom - Z11_ana)/np.abs(Z11_ana)):.2%}")
    print(f"  ROM: {np.mean(np.abs(Z11_rom - Z11_ana)/np.abs(Z11_ana)):.2%}")
    
    # Verify B matrix scaling in FOM
    # b_manual = Integrate(E, e) * 1j
    # We suspect mu0 is missing or something
    
def check_reconstruction():
    print("\n=== Checking Field Reconstruction ===")
    a, b, L = 100e-3, 50e-3, 200e-3
    rwg = RectangularWaveguide(a=a, b=b, L=L, maxh=0.06)
    fds = FrequencyDomainSolver(rwg, order=2)
    fds.assemble_matrices(nportmodes=1)
    
    fc = rwg.cutoff_frequency_TE10
    freq = 1.2 * fc
    
    # FOM Solve at one freq
    fds.frequencies = np.array([freq])
    # Use direct solve for clarity
    fds._solve_global_coupled(store_snapshots=True, solver_type='direct')
    E_fom_vec = fds.snapshots['global'][:, 0] # First excitation
    Z_fom = fds._Z_matrix[0, 0, 0]
    
    # ROM Solve
    rom = ModelOrderReduction(fds)
    rom.reduce(tol=1e-6)
    res_rom = rom.solve(freq/1e9, freq/1e9, 1)
    x_r = res_rom['x_r']['default'][:, 0] # First freq, first excitation
    E_rom_recon = rom.reconstruct_field(x_r)
    Z_rom = res_rom['Z'][0, 0, 0]
    
    diff = np.linalg.norm(E_fom_vec - E_rom_recon) / np.linalg.norm(E_fom_vec)
    print(f"Field reconstruction error: {diff:.2e}")
    print(f"Z_fom: {Z_fom:.4f}")
    print(f"Z_rom: {Z_rom:.4f}")
    print(f"Ratio: {Z_rom / (Z_fom + 1e-30):.4f}")

if __name__ == "__main__":
    debug_solvers()
    check_reconstruction()
