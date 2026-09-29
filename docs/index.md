<p align="center">
  <img src="assets/cavsim3d_logo_square.svg" alt="icon" width="128">
</p>
<h1 align="center">cavsim3d</h1>

**cavsim3d** is a 3D electromagnetic simulation and model-order reduction library for radio
frequency (RF) components, built on [NGSolve](https://ngsolve.org/) and
[PythonOCC](https://github.com/tpaviot/pythonocc-core).

## What it does

- Computes S-parameters, Z-parameters, resonant frequencies and fields of RF cavities,
  waveguides and accelerator components, from primitives or imported CAD models.
- Reduces a full-order finite-element model to a model with tens of unknowns that sweeps
  thousands of frequencies in a fraction of a second.
- Solves large structures in pieces: each part (or each domain of a CAD model) is solved and
  reduced on its own, and the reduced parts are joined into one model. A repeated part is
  solved once; a part solved in an earlier project is reused.

## Ways to solve a model

| Model | Route | Tutorial |
|---|---|---|
| One part | full-order sweep → reduced model | [Build a reduced-order model](tutorials/basics/reduced_order_model.ipynb) |
| Several parts, one mesh | solved in one piece → reduced model | [Join parts into one model](tutorials/multi_part/combine_parts.ipynb), §4 |
| Several parts or domains | each solved → each reduced → joined | [Cut a model into domains](tutorials/models/splitting_cad.ipynb), [Join parts into one model](tutorials/multi_part/combine_parts.ipynb) |
| Repeated or imported parts | unique parts solved once → reduced → copies joined | [Repeat a section](tutorials/multi_part/repeated_sections.ipynb), [Reuse a solved project](tutorials/multi_part/reuse_projects.ipynb) |

[How a model is solved in pieces](explanation/architecture.md) explains the stages and when
to use which route.

## Where to start

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg } **Getting Started**

    ---

    Install the code and run a first check.

    [:octicons-arrow-right-24: Getting Started](getting_started.md)

-   :material-school:{ .lg } **Tutorials**

    ---

    Step-by-step lessons, from a first waveguide to joined, repeated and imported parts.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-hammer-wrench:{ .lg } **How-to guides**

    ---

    Recipes for specific jobs: CAD import, materials, port modes, CST comparison.

    [:octicons-arrow-right-24: How-to guides](how-to/index.md)

-   :material-lightbulb-on:{ .lg } **Explanation**

    ---

    How the pipeline, the reduction, the ports and the joins work.

    [:octicons-arrow-right-24: Explanation](explanation/index.md)

-   :material-api:{ .lg } **Reference**

    ---

    solve() options, results, project files, and the API.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>
