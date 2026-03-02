"""
Example: Loading a Simulation and Plotting Results

This script demonstrates how to load a saved FrequencyDomainSolver project
and plot S and Z parameters for FOMs, ROMs, and concatenated systems.
"""

import matplotlib.pyplot as plt
from solvers.frequency_domain import FrequencyDomainSolver
from pathlib import Path

def run_example(project_name="example_simulation"):
    # 1. Load the simulation project
    # This restores everything: mesh, matrices, port modes, and solved results.
    print(f"Loading project: {project_name}...")
    try:
        fds = FrequencyDomainSolver.load(project_name)
    except FileNotFoundError:
        print(f"Project '{project_name}' not found. Please ensure it exists in the 'simulations' directory.")
        return

    # 2. Plot Global FOM Results
    # fds.fom provides the global (coupled) solve results.
    print("\nPlotting Global FOM S-parameters...")
    fig1, ax1 = fds.fom.plot_s(title="Global Coupled FOM - S-parameters")
    
    # 3. Handle Multi-Solid Case
    if fds.is_compound:
        print("\nMulti-solid structure detected.")
        
        # Plot per-domain FOM results
        # fds.foms is a collection of FOMResult objects for each domain.
        print("Plotting Per-Domain FOM S-parameters...")
        fig2, ax2 = fds.foms.plot_s(title="Per-Domain FOM - S-parameters")
        
        # Concatenate FOMs directly (W=I)
        # Note: This can be memory-intensive for large meshes.
        print("Concatenating FOMs...")
        # try:
        #     concat_fom = fds.foms.concatenate()
        #     concat_fom.solve(fds.frequencies[0]/1e9, fds.frequencies[-1]/1e9, len(fds.frequencies))
        #     concat_fom.plot_s(title="Concatenated FOM - S-parameters")
        # except Exception as e:
        #     print(f"FOM concatenation skipped or failed: {e}")

    # 4. Model Order Reduction (ROM)
    print("\nPerforming Model Order Reduction...")
    # Reduce the global FOM to create a ROM
    rom = fds.fom.reduce(tol=1e-4)
    
    # Solve the ROM over the same frequency range
    fmin, fmax = fds.frequencies[0] / 1e9, fds.frequencies[-1] / 1e9
    nsamples = len(fds.frequencies)
    rom.solve(fmin, fmax, nsamples)
    
    # Overlay ROM results on the FOM plot
    print("Overlaying ROM results on FOM plot...")
    rom.plot_s(ax=ax1, label="ROM", linestyle="--")
    ax1.legend()
    fig1.canvas.draw()

    # 5. Multi-Solid ROM & Concatenation
    if fds.is_compound:
        print("\nReducing and Concatenating ROMs...")
        # Reduce each domain and return a ROMCollection
        roms = fds.foms.reduce(tol=1e-4)
        
        # Concatenate the reduced models
        # This uses Kirchhoff coupling at internal ports.
        concat_rom = roms.concatenate()
        
        # Solve the concatenated system
        concat_rom.solve(fmin, fmax, nsamples)
        
        # Plot the concatenated ROM result
        print("Plotting Concatenated ROM S-parameters...")
        concat_rom.plot_s(title="Concatenated ROM - S-parameters")

    # 6. Z-Parameter Plotting
    print("\nPlotting Z-parameters for the global ROM...")
    rom.plot_z(title="Global ROM - Z-parameters", plot_type="mag") # can use 'db', 'mag', 'phase'

    print("\nShowing all plots. Close windows to finish.")
    plt.show()

if __name__ == "__main__":
    # Change this to your actual project name
    MY_PROJECT = "my_simulation_result" 
    
    # run_example(MY_PROJECT)
    print("Example script ready. Modify 'MY_PROJECT' in the script to point to your data.")
