"""
Example: Loading a Simulation and Plotting Results

This script opens a saved project and plots S and Z parameters of its
full-order model (FOM), its reduced model (ROM) and, for a model of several
domains, the joined model.
"""

import matplotlib.pyplot as plt

from cavsim3d.core.em_project import EMProject


def run_example(project_name="example_simulation", base_dir="./simulations"):
    # 1. Load the simulation project
    # This restores everything: mesh, matrices, port modes, and solved results.
    print(f"Loading project: {project_name}...")
    try:
        proj = EMProject.load(project_name, base_dir=base_dir)
    except FileNotFoundError:
        print(f"Project '{project_name}' not found in '{base_dir}'.")
        return
    fds = proj.fds
    fmin, fmax = fds.frequencies[0] / 1e9, fds.frequencies[-1] / 1e9
    nsamples = len(fds.frequencies)

    if not fds.is_compound:
        # 2. Single domain: the FOM, and a ROM of it on the same frequencies
        print("\nPlotting FOM S-parameters...")
        fig1, ax1 = fds.fom.plot_s(title="FOM - S-parameters")

        print("\nPerforming Model Order Reduction...")
        rom = fds.fom.reduce(tol=1e-4)
        rom.solve(fmin=fmin, fmax=fmax, nsamples=nsamples)

        print("Overlaying ROM results on FOM plot...")
        rom.plot_s(ax=ax1, label="ROM", linestyle="--")
        ax1.legend()
        fig1.canvas.draw()

        print("\nPlotting Z-parameters for the ROM...")
        rom.plot_z(title="ROM - Z-parameters", plot_type="mag")  # can use 'db', 'mag', 'phase'
    else:
        # 3. Several domains: the FOM of each domain, reduced and joined
        print("\nMulti-solid structure detected.")
        print("Plotting Per-Domain FOM S-parameters...")
        fds.foms.plot_s(title="Per-Domain FOM - S-parameters")

        print("\nReducing and Concatenating ROMs...")
        roms = fds.foms.reduce(tol=1e-4)
        # Join the reduced models through their internal ports
        concat_rom = roms.concatenate()
        concat_rom.solve(fmin=fmin, fmax=fmax, nsamples=nsamples)

        print("Plotting Concatenated ROM S-parameters...")
        concat_rom.plot_s(title="Concatenated ROM - S-parameters")

    print("\nShowing all plots. Close windows to finish.")
    plt.show()


if __name__ == "__main__":
    # Change this to your actual project name (and folder)
    MY_PROJECT = "my_simulation_result"

    # run_example(MY_PROJECT, base_dir="./simulations")
    print("Example script ready. Modify 'MY_PROJECT' in the script to point to your data.")
