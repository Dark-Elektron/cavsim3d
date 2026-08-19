# Getting Started

This guide walks you through a simple simulation.

## 📦 Installation
### Prerequisites

- Python 3.9-3.12
- Conda environment

To install `cavsim3d` from source, clone the repository and install it using `pip` in editable or normal mode:

```bash
git clone https://github.com/Dark-Elektron/cavsim3d
cd cavsim3d
pip install --upgrade pip
conda install -c conda-forge -y pythonocc-core pythreejs ipywidgets ipykernel --no-update-deps
pip install -e .
```

## Your First Simulation

This example creates a rectangular waveguide, inspects it, runs an FEM frequency sweep, reduces it via proper orthogonal decomposition (POD), and plots S-parameters. To keep the installation folder clean, navigate to a folder where you want your
simulation saved and either create a `getting_started.py` file or a Jupyter Notebook in order to follow the example.

### Step 1: Create the Geometry

```python
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide

# Create a project (results will be saved here)
proj = EMProject(name='my_first_sim', base_dir='./simulations')

# Define a rectangular waveguide: 100 mm wide, 200 mm long
wg = RectangularWaveguide(a=0.1, L=0.2, maxh=0.04)

# Add the geometry to the project
proj.geometry = wg

# # Alternatively, the geometry can be created directly in the project
# proj.create_primitive('rwg', a=0.1, L=0.2, maxh=0.04)
```

### Step 2: Inspect the Geometry

Before committing to a solve, check that the geometry and its mesh are what you expect.
Primitives build and mesh themselves on construction (using the `maxh` you passed), so
nothing extra is needed here.

```python
# Interactive 3D view of the CAD solid
proj.geo.show()

# ...or view the mesh the solver will actually use
proj.geo.show('mesh')

# Text summary: mesh size, port boundaries, geometry tag
proj.geo.print_info()
```

```text
======================================================================
RectangularWaveguide Geometry Information
======================================================================
Geometry type:          RectangularWaveguide
Compute method:         numeric
Supports analytical:    True
Boundary condition:     left|right|top|bottom

Component Tag:
  Full:                 RectangularWaveguide:a9dccab0
  Geometry hash:        a9dccab0e52c9952...

Cache status:           NOT CACHED

Mesh generated:         True
  Vertices:             48
  Elements:             90
  Ports:                ['port1', 'port2']
======================================================================
```

The two `port*` faces are the boundaries the solver excites, so this waveguide yields a
2 x 2 S-matrix. If the mesh looks too coarse, recreate the geometry with a smaller `maxh`.

!!! note
    `show()` renders through NGSolve's WebGUI, so it displays inside Jupyter (or the
    rendered notebook tutorials). Called from a plain `.py` script it opens no viewer —
    use `print_info()` there instead.

### Step 3: Solve the Full-Order Model (FOM)

```python
# Run an FEM frequency sweep from 1.5 to 3.0 GHz with 30 sample points
results = proj.fds.solve(fmin=1.5, fmax=3.0, nsamples=30, rerun=True)
```

This assembles the stiffness ($\mathbf{K}$), mass ($\mathbf{M}$), and port excitation ($\mathbf{B}$) matrices, solves
a linear system at each frequency point, and computes the S- and Z-parameter matrices.

### Step 4: Reduce to a ROM

```python
# Create a Reduced Order Model using POD (Proper Orthogonal Decomposition)
rom = proj.fds.fom.reduce(tol=1e-6)

# Solve the ROM over a much finer frequency grid (fast — milliseconds)
rom.solve(fmin=1.5, fmax=3.0, nsamples=500, rerun=True)
```

### Step 5: Plot Results

```python
# Plot S-parameters
rom.plot_s(plot_type='db', show=True)

# Plot Z-parameters
rom.plot_z(plot_type='db', show=True)
```

## Next Steps

- [**Pathway 1: Single Solid**](tutorials/pathway1_single_solid.ipynb) — Full walkthrough with analytical comparison
- [**Pathway 3: FOM Concatenation**](tutorials/pathway3_fom_concatenation.ipynb) — Multi-domain simulation
- [**Pathway 4: ROM Concatenation**](tutorials/pathway4_rom_concatenation.ipynb) — Most efficient multi-component workflow
- [**Importing CAD Files**](tutorials/importing_cad.ipynb) — Using STEP/IGES geometry
- [**Architecture Overview**](architecture.md) — Understanding the analysis pathways
