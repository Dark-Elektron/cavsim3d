# cavsim3d

[![License: LGPL](https://img.shields.io/badge/License-LGPL-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![Documentation Status](https://img.shields.io/badge/docs-GitHub%20Pages-brightgreen)](https://dark-elektron.github.io/cavsim3d/)

<p align="center">
  <img src="docs/assets/cavsim3d_logo_square.svg" alt="icon" width="128">
</p>
<h1 align="center">cavsim3d</h1>
`cavsim3d` is a Python library for **3D Electromagnetic Simulation** and **Model Order Reduction (MOR)** of RF structures. Built on the [NGSolve](https://ngsolve.org) finite element engine, it provides a streamlined workflow for analyzing complex cavity systems, waveguides, and multi-component assemblies.

## 📖 Documentation
Comprehensive tutorials and API documentation are available at:
**[https://dark-elektron.github.io/cavsim3d/](https://dark-elektron.github.io/cavsim3d/)**

## 🚀 Core Capabilities
*   **High-Order FEM**: Leverages NGSolve capabilities for 3D Maxwell solutions.
*   **Model Order Reduction (MOR)**: Accelerates frequency sweeps and eigenmode analysis using Proper Orthogonal Decomposition (POD).
*   **Component-Based Assembly**: Construct complex geometries by concatenating and aligning subdomains.
*   **Domain Decomposition**: Solve massive structures by breaking them into manageable subdomains and recombining via Kirchhoff coupling.

## 📦 Installation
### Prerequisites

- Python 3.9+
- Conda environment

To install `cavsim3d` from source, clone the repository and install it using `pip` in editable or normal mode:

```bash
git clone https://github.com/Dark-Elektron/cavsim3d
cd cavsim3d
pip install --upgrade pip
conda install -y pythonocc-core pythreejs ipywidgets --no-update-deps
pip install -e .
```

## 🛠️ Walkthrough: Circular Waveguide ROM Concatenation
This example demonstrates constructing two circular waveguide segments, solving them as independent subdomains, reducing their order, and concatenating them to compare against an analytical solution.

### 1. Geometry Construction & Mesh
Define two 50mm radius waveguide segments and assemble them sequentially.

```python
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import CircularWaveguide

# Initialize project and assembly
proj = EMProject(name='cwg_concat_example', overwrite=True)
assembly = proj.create_assembly(main_axis='Z')

# Create two segments
radius, L = 50e-3, 100e-3
wg1 = CircularWaveguide(radius=radius, length=L)
wg2 = CircularWaveguide(radius=radius, length=L)

# Add and align sequentially
assembly.add("segment1", wg1)
assembly.add("segment2", wg2, after="segment1")
assembly.build()

# Visualize the geometry
assembly.show() 

# Generate and visualize the mesh
assembly.generate_mesh(maxh=0.03)
assembly.show('mesh')
```

### 2. Full-Order Model (FOM) Solve
Solve for the per-domain frequency response and store field snapshots for reduction.

```python
# Solve subdomains from 2.0 to 3.0 GHz (above cutoff)
fom_config = {
    'nportmodes': 3,
    'order': 3,
    'fmin': 1e-3,
    'fmax': 3.0,
    'nsamples': 30,
    'solver_type': 'direct',
    # 'verbose': True
}
fom_result = proj.fds.solve(config=fom_config)
```

### 3. ROM Concatenation
Reduce the subdomains to a low-rank basis and concatenate them into a unified system.

```python
# Reduce subdomains via POD
roms = proj.fds.foms.reduce(tol=1e-6)

# Concatenate into a unified system and solve on more sample points
concat = roms.concatenate()
concat_result = concat.solve(fmin=1e-3, fmax=3.0, nsamples=1000)
```

### 4. Validation
Compare the concatenated ROM results against the analytical solution for the combined length.

```python
from cavsim3d.analytical.circular_waveguide import CWGAnalytical
import matplotlib.pyplot as plt

# [Snippet 1] Compare with Analytical Solution
analytical = CWGAnalytical(radius=radius, length=2 * L, 
                            freq_range=(fom_config['fmin'], fom_config['fmax']))

# Visualize comparison
which = [['1(1)1(1)'], ['1(1)2(1)']]

fig, axs = plt.subplot_mosaic(
    [[1, 2], [3, 4]], figsize=(10, 8), layout='constrained'
)

for idx, wh in enumerate(which):
    # Magnitude
    analytical.plot_s(wh, ax=axs[idx + 1])
    concat.plot_s(wh, ax=axs[idx + 1])
    # Phase
    analytical.plot_s(wh, plot_type='phase', ax=axs[idx + 3])
    concat.plot_s(wh, plot_type='phase', ax=axs[idx + 3])

fig.suptitle('Concat vs Analytical — S-Parameters', fontsize=14)
plt.show()
```

### 5. ROM Eigenmode Analysis
Eigenvalues can also be calculated directly from the concatenated reduced order models, allowing for rapid resonant mode identification and field visualization of the entire assembly.

```python
import numpy as np
# Compute the eigenmodes of the concatenated ROM
evals, evecs = concat.get_eigenmodes()
freqs = np.sqrt(evals)/(2*np.pi) * 1e-9
print(freqs)
# Plot the 3D field pattern of the an eigenmode
concat.plot_eigenmode(6)
```

## 📚 References
[1] T. Flisgen, J. Heller, T. Galek, L. Shi, N. Joshi, N. Baboi, R. M. Jones und U. van Rienen, *Eigenmode compendium of the third harmonic module of the European X-ray Free Electron Laser*, Phys. Rev. Accel. Beams 20, 042002, 2017, doi: [https://doi.org/10.1103/PhysRevAccelBeams.20.042002](https://doi.org/10.1103/PhysRevAccelBeams.20.042002)

[2] T. Wittig, R. Schuhmann und T. Weiland, *Model order reduction for large systems in computational electromagnetics*, Linear algebra and its applications, vol. 415, no. 2-3, pp. 499-530, 2006

[3] J. Schöberl, *C++ 11 implementation of finite elements in NGSolve*, Institute for Analysis and Scientific Computing, Vienna University of Technology, 30, 2014, [ngsolve.org/_static/ngs-cpp11.pdf](https://ngsolve.org/_static/ngs-cpp11.pdf)

---
*More tutorials and examples can be found at the [GitHub Pages](https://dark-elektron.github.io/cavsim3d/).*
