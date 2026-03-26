# cavsim3d

**cavsim3d** is a 3D electromagnetic cavity simulation and model-order reduction library built on [NGSolve](https://ngsolve.org/) and [PythonOCC](https://github.com/tpaviot/pythonocc-core).

## What It Does

- Simulate RF cavities, waveguides, and accelerator components
- Compute S-parameters, Z-parameters, and eigenfrequencies
- Accelerate wideband analysis with Proper Orthogonal Decomposition (POD)
- Handle multi-component assemblies with automatic concatenation

## Analysis Pathways

cavsim3d supports four distinct simulation workflows, from simple single-component model analysis to advanced hierarchical ROM concatenation of multiple models. See the [Architecture](architecture.md) page for details.

| Pathway | Input | Method | Best For |
|---------|-------|--------|----------|
| 1 | Single solid | FDS → FOM → ROM | Single component models |
| 2 | Multi-solid assembly | Global FDS → FOM → ROM | Small assemblies |
| 3 | Multi-solid | Per-domain FDS → FOMs → FOMs Concatenation -> ROM | Small assemblies |
| 4 | Multi-solid | Per-domain FDS → FOMs → ROMs → ROMs Concatenation -> ROM | Large assemblies, repeated components, maximum efficiency |

FDS - Frequency Domain Solver
FOM(s) - Full Order Model(s)
ROM(s) - Reduced Order Model(s)

The overarching workflow is as follows:
The frequency domain solver (FDS) solves the Maxwell's equations for a given frequency range and material properties. The full order model (FOM) is a representation of the physical system that is used to solve the frequency domain problem. The reduced order model (ROM) is a reduced order model of the FOM that is used to solve the frequency domain problem over a much finer frequency grid (fast — milliseconds). A m


## Quick Links

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg } **Getting Started**

    ---

    Run your first simulation in 5 minutes.

    [:octicons-arrow-right-24: Getting Started](getting_started.md)

-   :material-sitemap:{ .lg } **Architecture**

    ---

    Understand the four analysis pathways.

    [:octicons-arrow-right-24: Architecture](architecture.md)

-   :material-school:{ .lg } **Tutorials**

    ---

    Step-by-step guides for every workflow.

    [:octicons-arrow-right-24: Tutorials](tutorials/pathway1_single_solid.md)

-   :material-function-variant:{ .lg } **Mathematical Theory**

    ---

    The physics and numerics behind the solver.

    [:octicons-arrow-right-24: Theory](theory.md)

-   :material-api:{ .lg } **API Reference**

    ---

    Auto-generated from source docstrings.

    [:octicons-arrow-right-24: API Reference](api/project.md)

</div>
