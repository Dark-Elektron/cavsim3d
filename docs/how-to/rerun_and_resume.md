# How to control recomputation and resume a stopped sweep

Every stage saves its results as it finishes. This page shows how to reopen a project, when
`solve()` reuses those results, how to force or prevent a recomputation, and how to finish a
sweep that was interrupted.

## Reopen a saved project

Create the project with the same `name` and `base_dir`, **without** `overwrite=True`:

```python
from cavsim3d.core.em_project import EMProject

proj = EMProject(name="my_project", base_dir="./simulations")
```

The output starts with `Project 'my_project' exists. Loading...`. The geometry, mesh, solver
results, reduced models and concatenated models are all available again, for example
`proj.fds.fom`, `proj.fds.fom.rom` or `proj.fds.foms.roms.concat`.

!!! warning
    `overwrite=True` deletes the project folder, results included. Use it only to start a
    project from scratch. A folder that is not a cavsim3d project is never deleted: the call
    raises `FileExistsError` instead.

## Let `solve()` decide (the default)

With the default `rerun=None`, `solve()` compares the request with the stored results:

```python
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30)   # same request: returns the stored results
proj.fds.solve(fmin=1.5, fmax=3.0, nsamples=30)   # different request: recomputes
```

A change of frequency range, number of samples, element order or kind, port-mode counts,
port settings, materials or geometry counts as a different request. Before recomputing,
`solve()` prints what changed:

```text
Simulation configuration has changed since the results were saved:
  - fmin: 1.0 -> 1.5
The request differs from the stored results -> recomputing (pass rerun=False to keep the stored ones).
```

The same rule applies to `rom.solve()` and `concat.solve()`. Recomputing a full-order sweep
also discards the reduced and concatenated models built from it.

## Force a recomputation

```python
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30, rerun=True)
```

`rerun=True` also discards the samples of an interrupted sweep (see below).

## Keep the stored results whatever the request

```python
res = proj.fds.solve(fmin=1.5, fmax=3.0, nsamples=30, rerun=False)
```

If the request differs, `solve()` warns that the returned results do not match it.

## Chains of repeated or imported parts

For a chain, `solve()` prints a plan with one line per part. A part solved before with the
same settings and the same geometry is reused, in the same session or after reopening:

```text
Solve plan:
  cell  geometry             reuse     its full-order results from an earlier solve
```

The reduced models can be rebuilt with another tolerance at any time; they are reduced from
the stored full-order results, and a part already reduced with the same `tol` is reused:

```python
roms = proj.fds.foms.reduce(tol=1e-6)
roms = proj.fds.foms.reduce(tol=1e-9)     # again, tighter
concat = roms.concatenate()
```

After reopening, `proj.fds.foms`, `proj.fds.foms.roms` and `proj.fds.foms.roms.concat`
(with its last sweep) are available without solving again.

## Resume an interrupted sweep

A full-order sweep writes every finished frequency sample to `fds/checkpoint/` as it goes. If
the sweep stops (a stopped kernel, a crash, a closed laptop), call `solve()` again with the
same request:

```python
proj = EMProject(name="my_project", base_dir="./simulations")   # after a kernel restart
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30)                 # computes the missing samples only
```

A stored sample is reused only for the same frequencies, mesh, element order, port modes and
materials; the solver type may change in between. For a project of several parts, parts that
had finished are read back and the interrupted part continues. The checkpoint folder is
removed once the complete results are saved.

## Troubleshooting

- **A reopened project recomputes although nothing changed**: something in the request
  differs from the stored one: read the list of changes `solve()` prints. A common cause is
  passing `order` or `nportmodes` in one call and not in the other.
- **`Could not fully delete existing project`**: another program (or a second notebook) holds
  a file of the project open. Close it and create the project again.

**See also:** [The project folder](../explanation/project_layout.md) for what is stored where;
[solve() options](../reference/solve_options.md) for every option.
