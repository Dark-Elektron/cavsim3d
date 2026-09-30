# How to export results

This page shows how to get S- and Z-parameters, resonances and timings out of a project:
as arrays, as a Touchstone file, or as plots.

## Get the arrays

Every `solve()` returns them; the result objects hold them too:

```python
res = rom.solve(fmin=0.5, fmax=4.5, nsamples=2000)
f_hz = res["frequencies"]            # Hz
S = res["S"]                         # [frequency, response, excitation]
Z = res["Z"]
S21 = S[:, 1, 0]                     # or rom.S_dict["1(1)2(1)"] (excitation first)
```

See [Results](../reference/results.md) for the layout and the labels.

## Save them to a file

```python
import numpy as np

np.savez("results.npz", f=f_hz, S=S, Z=Z)
np.savetxt("s21.csv", np.column_stack([f_hz / 1e9, S21.real, S21.imag]),
           delimiter=",", header="f_GHz,re_S21,im_S21")
```

## Write a Touchstone file

From the full-order solver:

```python
path = proj.fds.export_touchstone("my_model", format="MA")      # writes my_model.s2p
```

- `format`: `"MA"` (magnitude, angle), `"DB"` (dB, angle) or `"RI"` (real, imaginary).
- `z0=50.0` (default) renormalises every port to 50 Ω, so the file's `R 50` is exact and a
  circuit simulator reads it correctly. Any positive `z0` works the same way.
- `z0=None` writes S as solved, each port referred to its own impedance (the values
  `fom.plot_s` shows). Touchstone v1 holds a single reference, so the option line then says
  `R 50` only nominally: the true references are listed in the header comments, and the
  call warns.

## Get the resonances

```python
f_res = proj.fds.fom.get_resonant_frequencies(n_modes=10)       # Hz
idx, f_ghz = concat.chain_eigenfrequencies(fmin_ghz=1.0, fmax_ghz=1.6)   # joined model, GHz
```

## Save a plot

```python
fig, ax = rom.plot_s(["1(1)1(1)", "1(1)2(1)"])
fig.savefig("s_parameters.png", dpi=200)
```

## Get the timings

```python
proj.timing_summary()          # prints stage times, reduction and speed-up
```

The same numbers are saved in `timing.json` in the project folder.

**See also:** [Results](../reference/results.md); [Project folder](../reference/project_folder.md)
for the files a project already writes.
