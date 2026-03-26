# Getting Started

This guide walks you through a simple simulation.

## Prerequisites

- Python 3.9+ with a [conda](https://docs.conda.io/en/latest/) environment
- [NGSolve](https://ngsolve.org/) and [PythonOCC](https://github.com/tpaviot/pythonocc-core) installed

## Installation

```bash
git clone https://github.com/Dark-Elektron/cavsim3d.git
cd cavsim3d
pip install -e .
```

## Your First Simulation

This example creates a rectangular waveguide, runs an FEM frequency sweep, reduces it via proper orthogonal decomposition (POD), and plots S-parameters.

### Step 1: Create the Geometry

```python
from core.em_project import EMProject
from geometry.primitives import RectangularWaveguide

# Create a project (results will be saved here)
proj = EMProject(name='my_first_sim', base_dir='./simulations')

# Define a rectangular waveguide: 100 mm wide, 200 mm long
wg = RectangularWaveguide(a=0.1, L=0.2, maxh=0.04)

# Add the geometry to the project
proj.geometry = wg

# # Alternatively, the geometry can be created directly in the project
# proj.create_primitive('rwg', a=0.1, L=0.2, maxh=0.04)
```

### Step 2: Solve the Full-Order Model (FOM)

```python
# Run an FEM frequency sweep from 1.5 to 3.0 GHz with 30 sample points
results = proj.fds.solve(fmin=1.5, fmax=3.0, nsamples=30)
```

This assembles the stiffness ($\mathbf{K}$), mass ($\mathbf{M}$), and port excitation ($\mathbf{B}$) matrices, solves
a linear system at each frequency point, and computes the S- and Z-parameter matrices.

### Step 3: Reduce to a ROM

```python
# Create a Reduced Order Model using POD (Proper Orthogonal Decomposition)
rom = proj.fds.fom.reduce(tol=1e-6)

# Solve the ROM over a much finer frequency grid (fast — milliseconds)
rom.solve(fmin=1.5, fmax=3.0, nsamples=500)
```

### Step 4: Plot Results

```python
# Plot S-parameters
rom.plot_s(plot_type='db')

# Plot Z-parameters
rom.plot_z(plot_type='db')
```

## Next Steps

- [**Pathway 1: Single Solid**](tutorials/pathway1_single_solid.md) — Full walkthrough with analytical comparison
- [**Pathway 3: FOM Concatenation**](tutorials/pathway3_fom_concatenation.md) — Multi-domain simulation
- [**Pathway 4: ROM Concatenation**](tutorials/pathway4_rom_concatenation.md) — Most efficient multi-component workflow
- [**Importing CAD Files**](tutorials/importing_cad.md) — Using STEP/IGES geometry
- [**Architecture Overview**](architecture.md) — Understanding the analysis pathways
