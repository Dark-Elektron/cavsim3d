# How-to guides

Recipes for specific jobs, for readers who already know the basics from the
[tutorials](../tutorials/index.md). Each guide goes straight to the steps and links to the
[reference](../reference/index.md) for every option.

## Building models

- [Import a CAD file](import_cad.md): units, ports on the right faces, cutting planes.
- [Assign materials](materials.md): permittivity, losses, conductivity, PEC solids.
- [Chain and arrange parts](chain_parts.md): order, repeats, replacing, axis, glued or
  coupled meshing.
- [Choose the mesh size and element order](mesh_and_order.md): `maxh` in metres, local
  refinement, convergence checks.

## Ports

- [Set up port modes](port_modes.md): mode counts per port, numeric modes, TEM reference
  impedance.
- [Model a microstrip line](microstrip.md): quasi-TEM ports, effective permittivity and
  line impedance.

## Solving

- [Run large models](large_models.md): direct or iterative solver, memory, long sweeps.
- [Control recomputation and resume a stopped sweep](rerun_and_resume.md).
- [Build a reliable reduced model](reduce_well.md): samples, tolerance, checks, spikes.
- [Get the figures of merit of resonances](resonance_figures.md): loaded and external Q per
  port, R/Q, wall Q and geometry factor, peak fields, transverse kick, field flatness.
- [Compute the beam impedance](beam_impedance.md): a beam through the structure, its
  impedance and coupling to the port modes, parts joined.

## Results and projects

- [Compare results with CST Studio Suite](compare_with_cst.md).
- [Export results](export_results.md): arrays, Touchstone files, resonances, plots.
- [Reuse and share projects](reuse_and_share.md): import solved projects, reference or
  copy, self-contained projects.
