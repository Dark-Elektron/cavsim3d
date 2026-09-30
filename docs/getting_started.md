# Getting Started

This page installs cavsim3d, checks that it works with a one-minute simulation, and points
to where to go next.

## Install

You need [conda](https://docs.conda.io) (Anaconda or Miniconda) and git. In a terminal:

```bash
conda create -n cavsim3d python=3.11 -y
conda activate cavsim3d
conda install -c conda-forge -y "pythonocc-core=7.9" pythreejs ipywidgets jupyterlab
git clone https://github.com/Dark-Elektron/cavsim3d
cd cavsim3d
pip install -e .
```

`pip install -e .` installs the code with its dependencies (NGSolve among them) in editable
mode, so a `git pull` updates it. `pythonocc-core` (for CAD import) is only available from
conda-forge, which is why it is installed first. Keep it at version 7.9: NGSolve loads its own
OpenCASCADE 7.8 into the same process, and pythonocc-core 8 (OpenCASCADE 8) cannot run next
to it.

## Check the installation

Start Jupyter (`jupyter lab`) in an empty folder, open a notebook, and run:

```python
from cavsim3d.core.em_project import EMProject

proj = EMProject(name="install_check", base_dir="./simulations", overwrite=True)
proj.create_primitive("rwg", name="guide", a=0.1, b=0.05, L=0.2, maxh=0.03)
res = proj.fds.solve(fmin=1.6, fmax=2.9, nsamples=5, solver_type="direct")

print(abs(res["S"][:, 1, 0]))
```

After the solver's log, the last line shows five values of $|S_{21}|$, all equal to 1 to
seven decimals or more:

```text
[1.         1.         1.         1.         0.99999996]
```

A uniform, lossless waveguide transmits everything, so this is the expected result. Then
draw the mesh:

```python
proj.geo.show("mesh")
```

An interactive 3D view of a meshed box appears below the cell. If it stays empty, the
Jupyter widget extensions are missing: install `webgui-jupyter-widgets` and restart Jupyter.

## Where to go next

- **Learn the code step by step**: the [tutorials](tutorials/index.md), starting with
  [Your first simulation](tutorials/basics/first_simulation.ipynb). Each one builds a small
  model and checks it against the exact answer.
- **Get a specific job done**: the [how-to guides](how-to/index.md), for example
  [importing a CAD file](how-to/import_cad.md) or
  [comparing with CST](how-to/compare_with_cst.md).
- **Understand how it works**: [Explanation](explanation/index.md), starting with
  [How a model is solved in pieces](explanation/architecture.md).
- **Look something up**: the [reference](reference/index.md) and the API pages.
