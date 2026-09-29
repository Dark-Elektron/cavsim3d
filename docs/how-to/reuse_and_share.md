# How to reuse and share projects

A solved and reduced project can be used as a part of another project, and a project can be
made self-contained for sharing or archiving. This page shows both.

## Use a solved project as a part

```python
part = proj.import_project("../simulations/coupler", name="coupler")
print(part)
```

```text
ImportedModel('coupler', mode='reference', has=rom, ports=['port1', 'port2'], band=[0.5, 2.9] GHz)
```

The handle shows what the source holds (`rom`, `fom` or only `geometry`), its ports and the
band its reduced model was trained on. `n=2` uses it twice. Add other parts before or after
it as usual, then run the normal pipeline:

```python
proj.fds.solve(fmin=0.5, fmax=2.9, nsamples=25)          # prints the solve plan
concat = proj.fds.foms.reduce(tol=1e-6).concatenate()
concat.solve(fmin=0.5, fmax=2.9, nsamples=2000)
```

## Read the solve plan

`solve()` prints one line per unique part before computing:

```text
Solve plan:
  section  imported (reference) reuse     its reduced model fits the request
  stub     geometry             compute   full-order solve, 25 samples
```

- `reuse`: the saved reduced model covers the request; nothing is computed for it.
- `reduce`: a full-order model exists; it is reduced in this project.
- `recompute`: the saved results do not cover the request (band, port modes). The part is
  solved again from the source's geometry, **in this project**.

The source project is never written. In a script, a `recompute` of an imported part runs
only with `rerun=True`:

```python
proj.fds.solve(fmin=0.5, fmax=3.5, nsamples=31, rerun=True)
```

## Copy instead of reference

```python
proj.import_project("../simulations/coupler", name="coupler", mode="copy")
```

A copy is independent of the source: it keeps working if the source is moved or deleted,
and it does not follow later changes to the source.

## Make a project self-contained before sharing

```python
n = proj.localize()          # copies every referenced part into the project
print(n, "part(s) copied")
```

Then share or archive the whole project folder. It can be opened elsewhere with
`EMProject(name, base_dir)`.

## When the source has changed

On reopening, a project warns if a referenced source moved or was solved again since it was
imported. Solve again to pick up the change, or `localize()` to keep the version you have.

**See also:** [Reuse a solved project](../tutorials/multi_part/reuse_projects.ipynb)
(tutorial); [Parts, joins and netlists](../explanation/parts_and_joins.md).
