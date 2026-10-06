# The project folder

A project is a folder on disk, and everything an analysis produces is saved into it as soon
as it is produced: the parts, the mesh, the full-order results, the reduced models and the
joined models. This page explains how that folder is organised and why, and how the saved
results decide whether a later request is answered from disk or computed again.

## Saved as you go

There is no "save" step in a normal workflow. Adding a part saves the geometry; meshing
saves the mesh; `solve()`, `reduce()` and `concatenate()` each save their result when they
finish. Reopening the project with `EMProject(name, base_dir)` restores all of it. Every
stage therefore acts as a cache for the next session: a sweep that took an hour is read
back in seconds.

The same property makes results portable. A reduced model saved in one project can be
imported into another ([Reuse a solved project](../tutorials/multi_part/reuse_projects.ipynb))
without the solver that produced it.

## The layout

```text
my_project/
├── project.json            project settings (order, main axis, part name, ...)
├── timing.json             wall time of each stage
├── geometry/               how the parts were built (history.json), STEP copies,
│   └── components/         one STEP file per part of a chain
├── mesh/                   the mesh (mesh.pkl) and FE space; per part: mesh_<part>.pkl
└── fds/                    everything the frequency-domain solver computed
    ├── config.json         the request the results belong to
    ├── solve.log
    ├── port_modes/
    ├── imports.json        imported projects (chains)
    ├── sections.json       parts of a chain solved here (chains)
    ├── checkpoint/         finished samples of a running sweep (temporary)
    ├── fom/                a single part ...
    │   ├── matrices/ eigenmodes/ s/ z/ snapshots/
    │   └── rom/            ... and its reduced model (same five folders)
    └── foms/               several parts or domains ...
        ├── matrices/ eigenmodes/ s/ z/ snapshots/
        └── roms/           ... their reduced models ...
            └── concat/     ... and the joined model (+ rom/ for a further reduction)
```

The complete list of files is in the [project folder reference](../reference/project_folder.md).

Three rules keep the layout predictable:

- **Every stage folder has the same five subfolders**: `matrices/`, `eigenmodes/`, `s/`,
  `z/`, `snapshots/`, plus the folder of the next stage nested inside it
  (`fom/rom/`, `foms/roms/concat/`). What a stage is can be read from its path. A beam
  adds `z_tilde/`, `s_tilde/` and `snapshots_beam/` next to them, and leaves the port
  files as they are.
- **Parts are told apart by file name, not by folder.** A model of several parts keeps one
  `foms/` tree, with a file per part and quantity: `matrices/K_inlet.h5`,
  `s/s_slab.h5`, and so on. There are no per-part subfolders.
- **One mesh folder and one geometry folder per project**, at the top level. Per-part
  meshes and geometry are files inside them (`mesh_<part>.pkl`,
  `geometry/components/<part>.step`).

A part imported by copy from another project ends up in exactly the same files a computed
part would, renamed after the part in the new project. On disk, the two are
indistinguishable.

## When saved results are reused

`fds/config.json` records the request the results belong to: frequency range and samples,
element order and kind, port modes and port settings, materials, and a signature of the
geometry. When `solve()` is called again (in the same session or after reopening), it
compares the new request with that record:

- **same request**: the stored results are returned; nothing is computed;
- **different request**: `solve()` prints what changed and recomputes. Recomputing a
  full-order model also discards the reduced and joined models built from it, because
  they would no longer match.

`rerun=True` and `rerun=False` override this in either direction
([How to control recomputation](../how-to/rerun_and_resume.md)).

Changing the parts or the mesh invalidates everything downstream: the code asks before
deleting existing results (or proceeds with `force=True`).

## Interrupted sweeps

A long full-order sweep writes each finished frequency sample to `fds/checkpoint/`. It is
the one temporary folder: when the sweep completes and its results are saved, the
checkpoint is deleted. If the sweep is interrupted, the samples already there are reused by
the next `solve()` with the same request, as long as the mesh, order, port modes and
materials are unchanged.

## Related

- [How to control recomputation and resume a stopped sweep](../how-to/rerun_and_resume.md)
- [Project folder reference](../reference/project_folder.md)
- [How a model is solved in pieces](architecture.md)
