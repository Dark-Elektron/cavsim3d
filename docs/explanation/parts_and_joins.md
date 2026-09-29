# Parts, joins and netlists

A project holds a list of parts, and a model of several parts has to be meshed, solved and
joined. This page explains how parts are laid out, the two ways they are joined (glued and
coupled), how imported projects take part, and what is checked at every join.

## A project is a list of parts

Parts are added one at a time:

- `proj.create_primitive(kind, name=...)`: a built-in primitive, either a waveguide or a
  body of revolution such as an elliptical cavity;
- `proj.import_geometry(path, name=...)`: a CAD file;
- `proj.import_project(path, name=...)`: a project solved earlier;
- `proj.add(name, part)`: any geometry object, including a sub-assembly.

Each part has a name, and the name is its identity: adding a part under a name the project
already has **replaces** that part. Re-running a notebook cell therefore does not double
the model. `n=4` on a part repeats it four times in a row.

With one part, the project's geometry is that part. With two or more, the parts are
chained, in the order they were added, along the project's main axis (`proj.main_axis`, Z
unless set). The chain is held by an `Assembly`, which only records the parts, their order,
their repeat counts and their connections; it never computes anything. Meshing prints the
chain: the axis, where each part sits, and how the parts are meshed.

## Glued and coupled parts

There are two ways to turn a chain into a model.

**Glued.** The parts are fused into one conformal mesh: elements match across the faces
where parts touch, and those faces become internal ports. This is the strategy for plain
geometry parts that each appear once. The model can then be solved in one piece, or per
part and joined (see [How a model is solved in pieces](architecture.md)).

**Coupled.** Each unique part is meshed and solved on its own, as if it were a project of
its own, and the copies are joined afterwards through the modes of the ports that face each
other along the axis. This is the strategy as soon as a part is repeated (`n=` > 1) or
imported from another project: a repeated part is solved once whatever the number of
copies, and an imported part keeps the mesh it was solved on. Because coupled parts share
nothing but port modes, they are joined through the ports that face each other along the
axis, not by port name.

The mesh summary names the strategy and the reason; `asm.set_mesh_strategy("glued")` or
`"coupled"` overrides it. `flip=True`, which turns a part end for end, is available for
glued parts only: a coupled part must be solved in the orientation it is used.

## Imported projects: reference or copy

An imported project enters the chain as a coupled part, in one of two modes:

- **Reference** (the default). Nothing is copied; the part's reduced model is read from the
  source project where it lies. The importing project records the source and a fingerprint
  of its results, and warns on reopening if the source has moved or changed.
  `proj.localize()` turns references into copies at any time.
- **Copy** (`mode="copy"`). The source's results are copied into the importing project,
  into the same folders and under the same file names a computed part would have. The copy
  is a snapshot: it keeps working if the source is deleted.

Either way, **the source project is never written**. When `solve()` runs, it prints a plan
with one line per unique part: *reuse* (the saved reduced model fits the request),
*reduce* (a full-order model exists but no reduced one), or *recompute* (the part's saved
results do not cover the request: a wider band, more port modes). Anything computed for an
imported part is computed into the importing project. In a script, where no one can confirm
the plan, a full-order recompute of an imported part needs `rerun=True`.

## What is checked at a join

Compatibility at a join is a checked condition, not a matter of where the parts came from:

- **Port-mode counts** must match on the two faces of a join, or joining raises an error.
- **Mode fingerprints** must correspond one to one: mode type, indices, cutoff wavenumber
  (so the cross-section dimensions) and polarisation. A mismatch raises an error.
- **Propagating modes**: if a join carries fewer modes than propagate below the top of the
  band, joining warns and names the port and the count needed, since the modes left out
  are reflected at the join.
- **Training bands** of reduced parts must overlap: disjoint bands raise an error, and
  sweeping outside the common band warns.

A multi-solid project cannot be imported as a single part: a part is one domain. Add its
solids individually instead.

## Related

- Tutorials: [Join parts into one model](../tutorials/multi_part/combine_parts.ipynb),
  [Repeat a section](../tutorials/multi_part/repeated_sections.ipynb),
  [Reuse a solved project](../tutorials/multi_part/reuse_projects.ipynb).
- [How to chain and arrange parts](../how-to/chain_parts.md);
  [How to reuse and share projects](../how-to/reuse_and_share.md).
- [Ports and port modes](ports.md).
