# How to run large models

A cavity or module with hundreds of thousands of unknowns needs the right linear solver,
a watch on memory, and a sweep that survives interruptions. This page covers those
settings.

## Choose the linear solver

`solver_type` is a `solve()` option:

| Value | Solver | Use for |
|---|---|---|
| `"auto"` (default) | direct when its factorisation fits in 60 % of the free memory, iterative otherwise | most models |
| `"direct"` | sparse factorisation (PARDISO) | small and medium models; hard-to-converge cases |
| `"iterative"` | COCG with a BDDC preconditioner, finished by GMRES if it stalls | large models; low memory |

```python
proj.fds.solve(fmin=1.0, fmax=1.8, nsamples=12, solver_type="direct")
```

`"auto"` prints its choice next to the memory estimate and the free memory
(`Auto solver: ... unknowns, a factorisation needs about ... GB of ... GB free`).
A factorisation serves every right-hand side of a sample (each port mode, each beam) and
does not slow down near resonances, so it is preferred whenever it fits; its memory grows
faster than the model (about the 1.4th power of the unknowns).

## Tune the iterative solver

```python
proj.fds.solve(fmin=1.0, fmax=1.8, nsamples=12, solver_type="iterative",
               iterative_opts={"method": "gmres", "maxsteps": 1000, "tol": 1e-10})
```

The defaults are `method="cocg"`, `precond="bddc"`, `maxsteps=500`, `tol=1e-8`. `tol` is
relative to the right-hand side. After a sweep, plot how each sample converged:

```python
proj.fds.fom.plot_residual(per_excitation=True)      # proj.fds.foms.plot_residual() for several parts
```

A sample that stops at `maxsteps` without reaching `tol` is not converged: raise
`maxsteps`, or switch to `"direct"` for that model. A COCG solve that GMRES had to finish
is reported in the progress output (`verbose=True`) as "finished by GMRES".

## Save memory

- Keep the default first-kind elements (`nedelec="first"`): about a third fewer unknowns
  than `"second"` at order 2.
- With `nedelec="second"` the direct solver stores and factorises the symmetric matrix as
  symmetric (same speed, half the memory); with `"first"` the full matrix factorises
  faster.
- Lower `order` and refine the mesh only where the field varies fast
  ([How to choose the mesh size and element order](mesh_and_order.md)).
- Cut the model into parts or domains and solve them one at a time
  ([Cut a model into domains](../tutorials/models/splitting_cad.ipynb)). A part repeated
  with `n=` is solved once.

## Spend the full-order samples where they matter

A reduced model interpolates between the full-order samples. Use few samples (10 to 30) over
the band you need, and let the reduced model fill in the rest:

```python
proj.fds.solve(fmin=1.0, fmax=1.8, nsamples=12)
rom = proj.fds.fom.reduce(tol=1e-9)
rom.solve(fmin=1.0, fmax=1.8, nsamples=2000)
```

See [How to build a reliable reduced model](reduce_well.md).

## Follow a long sweep and survive interruptions

```python
proj.fds.solve(fmin=1.0, fmax=1.8, nsamples=12, verbose=True)   # per-sample progress
```

Every finished sample is written to `fds/checkpoint/`. After a crash or a stopped kernel,
reopen the project and call `solve()` with the same request: only the missing samples are
computed ([How to control recomputation](rerun_and_resume.md)). The log of every solve is
kept in `fds/solve.log`; `proj.fds.fom.print_log()` prints it.

**See also:** [solve() options](../reference/solve_options.md).
