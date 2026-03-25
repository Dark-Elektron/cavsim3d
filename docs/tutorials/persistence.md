# Tutorial: Persistence — Save, Load, and Rerun Control

**cavsim3d** provides a hierarchical persistence system that saves geometry, mesh, and solver results to disk. This enables:

- **Resuming** work without recomputation
- **Sharing** projects with colleagues
- **Version control** of simulation configurations

## Saving a Project

```python
from core.em_project import EMProject
from geometry.primitives import RectangularWaveguide

# Create and solve
proj = EMProject(name='my_project', base_dir='./simulations')
proj.geometry = RectangularWaveguide(a=0.1, L=0.2, maxh=0.04)
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30)

# Save everything
proj.save()
```

This creates the following directory structure:

```
simulations/my_project/
├── metadata.json            # Project config
├── geometry/
│   ├── history.json         # Geometry construction history
│   └── geometry.step        # Exported STEP file
├── mesh/
│   └── mesh.pkl             # Serialised NGSolve mesh
└── fds/
    ├── config.json           # Solver configuration
    ├── port_modes/
    │   └── port_modes.pkl    # Port eigenmode data
    └── fom/
        ├── metadata.json
        ├── matrices/         # K, M, B matrices (HDF5)
        ├── z/                # Z-parameter matrices
        ├── s/                # S-parameter matrices
        └── snapshots/        # Field snapshots + frequencies
```

## Loading a Project

```python
proj = EMProject(name='my_project', base_dir='./simulations')
# ^ Automatically detects existing project and loads it
```

On load, the project reconstructs:

- **Geometry** from `history.json` (replays construction steps)
- **Mesh** from `mesh.pkl`
- **Solver state** including all result matrices

## Rerun Control

The key feature: **`rerun=False`** prevents re-computation when results already exist.

```python
# This returns instantly — loads cached results from disk
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30, rerun=False)
```

```python
# This forces a fresh computation (overwrites cached results)
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30, rerun=True)
```

!!! warning "Configuration Change Detection"
    If you change the frequency range or solver parameters, **cavsim3d** will warn you that the loaded results may be invalid:
    ```
    WARNING: Simulation configuration has changed since last save/load:
      - fmin: 1.0 -> 1.5
    Existing results may be invalid. Use rerun=True to recompute.
    ```

## Source Link Management

When using imported CAD files, the project tracks the **original source path**. If the source file is moved or modified, you'll be prompted:

- **Update**: Re-import the geometry from the new/modified source
- **Keep Local**: Use the project-local copy (in `geometry/components/`)
- **Break Link**: Stop tracking the source file entirely

## Assembly Persistence

For assemblies, each sub-component's source file is copied into the project:

```
geometry/
├── history.json
├── assembly.step           # Full fused assembly
└── components/
    ├── cell1.iges          # Local copy of cell 1
    └── cell2.iges          # Local copy of cell 2
```

This ensures the project is self-contained and portable.
