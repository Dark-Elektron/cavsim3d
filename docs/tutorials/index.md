# Tutorials

The tutorials are lessons: each one builds a small, complete model, step by step, and
shows the result of every step. They are written to be run in order, top to bottom, in a
Jupyter notebook. Every notebook sets up its own project, so any one of them also runs on
its own.

Most tutorials use rectangular waveguides. Their S-parameters, impedances and resonances
are known in closed form, so every result can be checked against the exact answer, and a
full run takes a minute or two on a laptop.

## The sections

**[Basics](basics/index.md)**: the first three lessons. A frequency sweep of a single
part, a reduced-order model of it, and its resonances and fields. Start here.

**[Building models](models/index.md)**: where models come from. Cavities built from their
parameters, CAD files, cutting a model into domains, and filling it with lossy material.

**[Models of several parts](multi_part/index.md)**: chaining parts into one model, and
the three ways of solving it: in one piece, per part and then concatenated, or per part,
reduced, and then concatenated. Repeating a part, and reusing a project solved earlier.

**[Ports and port modes](ports/index.md)**: more than one mode per port, coaxial (TEM)
ports, and microstrip (quasi-TEM) ports.

**[Applications](applications/index.md)**: complete studies on real structures, such as a
TESLA 9-cell cavity chain and a two-cavity module benchmarked against CST Studio Suite.

## Before you start

Install the code as described in [Getting Started](../getting_started.md). Each tutorial
writes its project to a `simulations/` folder next to the notebook.

The tutorials show one way through each task. For a specific job ("how do I import a STEP
file in millimetres?") see the [how-to guides](../how-to/index.md). For the reasons behind
the design see [Explanation](../explanation/index.md), and for every parameter see the
[reference](../reference/index.md).
