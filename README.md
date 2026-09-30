<p align="center">
  <img src="docs/assets/cavsim3d_logo_square.svg" alt="icon" width="128">
</p>
<h1 align="center">cavsim3d</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/Dark-Elektron/cavsim3d/actions/workflows/tests.yml/badge.svg)](https://github.com/Dark-Elektron/cavsim3d/actions/workflows/tests.yml)
[![Documentation Status](https://img.shields.io/badge/docs-GitHub%20Pages-brightgreen)](https://dark-elektron.github.io/cavsim3d/)

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

- Python 3.9-3.13
- Conda (for `pythonocc-core`, which is published on conda-forge only; `pythreejs` serves its notebook viewer)

Create an environment, then install `cavsim3d` from source:

```bash
conda create -n cavsim3d -c conda-forge python=3.11 "pythonocc-core=7.9" pythreejs
conda activate cavsim3d
git clone https://github.com/Dark-Elektron/cavsim3d
cd cavsim3d
pip install -e .
```

Keep `pythonocc-core` at 7.9: NGSolve loads its own OpenCASCADE 7.8 into the same process,
and on Windows pythonocc-core 8 (OpenCASCADE 8) cannot run next to it.

`pip install -e ".[dev]"` adds the test and lint tools, `pip install -e ".[docs]"` the documentation tools.

### Running the tests

```bash
python -m pytest tests/ -q                 # full suite, about 10 minutes
python -m pytest tests/ -q -m "not slow"   # without the heavy 3D solves
```

## 🛠️ Walkthrough: Circular Waveguide ROM Concatenation
This example builds a circular waveguide from two segments, solves each segment on its own, reduces each one, joins the reduced models, and compares the result with the analytical solution. All lengths are in metres, `maxh` too.

### 1. Geometry and mesh
Each call adds a part to the project; parts are chained in the order they are added, along +z.

```python
from cavsim3d.core.em_project import EMProject

proj = EMProject(name="cwg_concat_example", base_dir="./simulations", overwrite=True)

radius, L = 50e-3, 100e-3
proj.create_primitive("cwg", name="segment1", radius=radius, length=L, maxh=0.03)
proj.create_primitive("cwg", name="segment2", radius=radius, length=L, maxh=0.03)
proj.generate_mesh(maxh=0.03)          # prints the layout: two parts glued along +z

proj.geo.show("mesh")
```

### 2. Full-order model (FOM)
Solve each segment from 1 to 3 GHz, with three port modes (TE11 in two polarisations, and TM01).

```python
res = proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=21, nportmodes=3)
```

### 3. Reduce and concatenate
Reduce each segment by POD, join the reduced models, and sweep the joined model at 1000 frequencies.

```python
roms = proj.fds.foms.reduce(tol=1e-6)
concat = roms.concatenate()
concat_result = concat.solve(fmin=1.0, fmax=3.0, nsamples=1000)
```

### 4. Validation
Compare with the analytical solution for the whole 200 mm guide: magnitude (dB) and phase of S11 and S21.

```python
from cavsim3d.analytical import CWGAnalytical
import matplotlib.pyplot as plt

analytical = CWGAnalytical(radius=radius, length=2 * L, freq_range=(1.0, 3.0))

fig, axs = plt.subplot_mosaic([["11 db", "21 db"], ["11 phase", "21 phase"]],
                              figsize=(12, 7), layout="constrained")
for ij in ("11", "21"):
    label = f"{ij[1]}(1){ij[0]}(1)"       # excitation first: '1(1)2(1)' is S21
    for kind in ("db", "phase"):
        ax = axs[f"{ij} {kind}"]
        analytical.plot_s([label], plot_type=kind, ax=ax, label="exact")
        concat.plot_s([label], plot_type=kind, ax=ax, ls="--", label="joined ROMs",
                      title=f"S{ij}")
plt.show()
```

### 5. Resonances of the joined model
The resonant frequencies of the joined model inside the band, with an index for each, and the field of one of them:

```python
idx, f_ghz = concat.chain_eigenfrequencies(fmin_ghz=1.0, fmax_ghz=3.0)
print(f_ghz)
concat.plot_eigenmode(int(idx[0]))
```

The [tutorials](https://dark-elektron.github.io/cavsim3d/tutorials/) cover every step in detail, from a first waveguide to repeated and imported parts.

## 📝 Citing
If you use `cavsim3d` in your work, please cite it: GitHub's **Cite this repository** button (from [`CITATION.cff`](CITATION.cff)) gives the reference in BibTeX and APA. Changes between versions are listed in the [changelog](CHANGELOG.md).

## 📚 References
[1] T. Flisgen, J. Heller, T. Galek, L. Shi, N. Joshi, N. Baboi, R. M. Jones und U. van Rienen, *Eigenmode compendium of the third harmonic module of the European X-ray Free Electron Laser*, Phys. Rev. Accel. Beams 20, 042002, 2017, doi: [https://doi.org/10.1103/PhysRevAccelBeams.20.042002](https://doi.org/10.1103/PhysRevAccelBeams.20.042002)

[2] T. Wittig, R. Schuhmann und T. Weiland, *Model order reduction for large systems in computational electromagnetics*, Linear algebra and its applications, vol. 415, no. 2-3, pp. 499-530, 2006

[3] J. Schöberl, *C++ 11 implementation of finite elements in NGSolve*, Institute for Analysis and Scientific Computing, Vienna University of Technology, 30, 2014, [ngsolve.org/_static/ngs-cpp11.pdf](https://ngsolve.org/_static/ngs-cpp11.pdf)

---
*More tutorials and examples can be found at the [GitHub Pages](https://dark-elektron.github.io/cavsim3d/).*
