# solve() options

Options of `proj.fds.solve()` (the full-order sweep) and of `solve()` on reduced and joined
models. Every option can be passed as a keyword or inside `config={...}`; keywords win.

```python
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30, order=2, nportmodes=[2, 1])
proj.fds.solve(config=dict(fmin=1.0, fmax=3.0, nsamples=30, order=2))
```

A name that is not an option raises `TypeError` and suggests the closest one
(`n_port_modes=2` → "did you mean 'nportmodes'?"). The reduced and joined models accept a
config written for the full-order solve: its full-order options have no effect there.

## Full-order solve: `proj.fds.solve()`

### Sweep

| Option | Type | Default | Meaning |
|---|---|---|---|
| `fmin`, `fmax` | float | required | band, in GHz; finite, `0 < fmin <= fmax` |
| `nsamples` | int | 100 | number of equally spaced frequencies, `fmin` and `fmax` included (a whole number >= 1) |
| `rerun` | None, bool | `None` | `None`: reuse stored results for the same request, recompute when it changed. `True`: always recompute (also ignores an interrupted sweep's samples). `False`: keep stored results whatever the request |
| `verbose` | None, bool | `None` | `True`: per-sample progress in the output, for this solve. `None`: the console setting of `cavsim3d.utils.printing.set_verbosity` |

### Discretisation

| Option | Type | Default | Meaning |
|---|---|---|---|
| `order` | int | 3 | polynomial order of the finite elements |
| `nedelec` | `"first"`, `"second"` | `"first"` | Nédélec element kind; projects saved without the setting keep `"second"` |

### Ports

| Option | Type | Default | Meaning |
|---|---|---|---|
| `nportmodes` | int, list, dict | 1 | modes per port: one count for all, a list in `port_map()` order, or `{port: count, "default": count}` |
| `mode_source` | `"analytic"`, `"numeric"` | `"analytic"` | port modes of external ports: closed form (rectangular, circular, coaxial) or a 2D eigenproblem on the port face |
| `mode_source_internal` | `"analytic"`, `"numeric"` | `"analytic"` | the same for internal ports between parts |
| `impedance_reference` | `"line"`, `"wave"` | `"line"` | reference impedance of TEM ports; TE/TM modes always use their wave impedance |
| `qtem_ports` | list of str | auto | ports solved as quasi-TEM (inhomogeneous cross-section, e.g. microstrip); detected automatically when a port face has several materials |
| `qtem_conductor_bbnd` | str | from the geometry | conductor edges on the port plane, e.g. `"microstrip_edges|ground_edges"` |
| `qtem_voltage_path` | – | from the geometry | integration path of the line voltage for the power-voltage impedance |

### Linear solver

| Option | Type | Default | Meaning |
|---|---|---|---|
| `solver_type` | `"iterative"`, `"direct"`, `"auto"` | `"iterative"` | GMRES + preconditioner, sparse LU, or iterative above 400 000 unknowns |
| `iterative_opts` | dict | `{"precond": "bddc", "maxsteps": 500, "tol": 1e-6, "printrates": False}` | settings of the iterative solver; given keys replace the defaults |

### Several parts or domains

| Option | Type | Default | Meaning |
|---|---|---|---|
| `per_domain` | bool | `True` | solve each part / domain on its own (the input of `foms.reduce()`); `False`: solve the whole mesh as one system. Ignored for a single part |
| `global_method` | `"coupled"`, None | `"coupled"` | `"coupled"` with `per_domain=False` solves the whole mesh; `None` keeps only per-domain results |
| `store_snapshots` | bool | `True` | keep the field solutions; needed by `reduce()` and field plots |
| `compute_s_params` | bool | `True` | compute S from Z |

## Reduced and joined models: `rom.solve()`, `roms.solve()`, `concat.solve()`

| Option | Type | Default | Meaning |
|---|---|---|---|
| `fmin`, `fmax`, `nsamples` | | as above | band in GHz and number of samples |
| `rerun` | None, bool | `None` | as above |
| `solver_type` | str | `"auto"` | dense solver choice for the small reduced system |
| `verbose` | None, bool | `None` | as above |
| `compute_s_params` | bool | `True` | (joined models) compute S from Z |

## Return value

A dictionary; see [Results](results.md).
