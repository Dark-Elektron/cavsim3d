<p align="center">
  <img src="assets/cavsim3d_logo_square.svg" alt="icon" width="128">
</p>
<h1 align="center">cavsim3d</h1>

**cavsim3d** is a 3D electromagnetic simulation and model-order reduction library for radio frequency (RF) components. It is built on [NGSolve](https://ngsolve.org/) and [PythonOCC](https://github.com/tpaviot/pythonocc-core).

## What It Does

- Simulate RF cavities, waveguides, and accelerator components
- Compute S-parameters, Z-parameters, and eigenfrequencies
- Accelerate electromagnetic field analysis through Model Order Reduction (MOR)
- Handle multi-component assemblies with automatic concatenation

## Analysis Pathways

cavsim3d supports four distinct simulation workflows, from simple single-component model analysis to advanced hierarchical ROM concatenation of multiple models. See the [Architecture](architecture.md) page for details.

| Pathway | Input | Method | Best For |
|---------|-------|--------|----------|
| 1 | Single solid | FDS → FOM → ROM | Single component models |
| 2 | Multi-solid assembly | Global FDS → FOM → ROM | Small assemblies |
| 3 | Multi-solid | Per-domain FDS → FOMs → FOMs Concatenation → ROM | Small assemblies, mostly for comparison |
| 4 | Multi-solid | Per-domain FDS → FOMs → ROMs → ROMs Concatenation → ROM | Large assemblies, repeated components, maximum efficiency |

FDS - Frequency Domain Solver,
FOM(s) - Full Order Model(s),
ROM(s) - Reduced Order Model(s)

The overall workflow is as follows:

The frequency domain solver (FDS) computes solutions to Maxwell’s equations over a specified frequency range for given material properties. This is performed using the full order model (FOM), which represents the physical system in its complete form.

Input geometries may originate either as a single assembly or as multiple individual models. A single assembly exported from external software can be decomposed into smaller sub-models, which are analysed independently. Alternatively, individual models can be directly imported into the code and concatenated to form a multi-solid assembly.

Each sub-model or solid is first solved independently at the full order level. Subsequently, a reduced order model (ROM) is constructed for each component, significantly lowering the computational complexity. These reduced models are then concatenated to form a coupled system representation. If required, an additional reduction step can be applied to the concatenated system to further compress the model.

The resulting ROM enables the frequency domain problem to be evaluated over a much finer frequency grid at very low computational cost.


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

    [:octicons-arrow-right-24: Tutorials](tutorials/pathway1_single_solid.ipynb)

-   :material-function-variant:{ .lg } **Mathematical Theory**

    ---

    The physics and numerics behind the solver.

    [:octicons-arrow-right-24: Theory](theory.md)

-   :material-api:{ .lg } **API Reference**

    ---

    Auto-generated from source docstrings.

    [:octicons-arrow-right-24: API Reference](api/project.md)

</div>
