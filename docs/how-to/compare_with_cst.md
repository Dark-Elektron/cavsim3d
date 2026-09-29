# How to compare results with CST Studio Suite

This page shows how to load S- and Z-parameters exported from CST Studio Suite and compare
them with a full-order, reduced or joined model.

## Export from CST

In CST, export the S-parameters (and, if wanted, the Z-matrix and the port impedances) as
text files: *Post-Processing → Export → Plot data (ASCII)* for each curve, or a result
template that writes them all. Put them in a folder called `Export` inside a folder for the
model:

```text
cst_reference/my_model/
└── Export/
    ├── S-Parameters_S1,1.txt        (or S-Parameters_S1(1),1(1).txt with modes)
    ├── S-Parameters_S2,1.txt
    ├── Z Matrix_Z1,1.txt
    └── ...
```

## Load the results

```python
from cavsim3d.analytical.cst_result import CSTResult

cst = CSTResult("cst_reference/my_model")        # the folder that contains Export/
```

The loader prints the band, the number of ports and modes per port, and how many S- and
Z-parameters it found. `cst.frequencies` is in Hz, `cst.S_matrix` and `cst.Z_matrix` are
`[frequency, response, excitation]`.

## Plot both on the same axes

`CSTResult` has the same `plot_s` and `plot_z` as the result objects:

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplot_mosaic([["11 db", "21 db"], ["11 phase", "21 phase"]],
                              figsize=(12, 7), layout="constrained")
for ij in ("11", "21"):
    label = f"{ij[1]}(1){ij[0]}(1)"          # excitation first: '1(1)2(1)' is S21
    for kind in ("db", "phase"):
        ax = axs[f"{ij} {kind}"]
        cst.plot_s([label], plot_type=kind, ax=ax, lw=3, label="CST")
        rom.plot_s([label], plot_type=kind, ax=ax, ls="--", label="ROM", title=f"S{ij}")
plt.show()
```

## Compare many port modes at once

For models with several ports and modes, `cavsim3d.analysis` works on whole matrices and
handles the different key conventions (the code names the excitation first, CST the
response first):

```python
from cavsim3d.analysis import network_matrix, plot_matrix, band_difference

labels = ["1(1)", "2(1)", "3(1)"]
S_cst = network_matrix(cst, labels)            # [frequency, response, excitation]
S_rom = network_matrix(concat, labels)
fig, axs = plot_matrix({"CST": S_cst, "ROM": S_rom}, freq=cst.frequencies / 1e9)
```

To keep only some port modes of a model (the others open-circuited), use
`keep_port_modes(model, labels)`.

## Put numbers on the agreement

Both models must be on the same frequency grid. Interpolate the CST result to the model's
frequencies, or solve the reduced model on CST's grid:

```python
cst_on_grid = cst.interpolate_to(rom.frequencies)          # frequencies in Hz
S_cst = network_matrix(cst_on_grid, labels)
print(band_difference(S_rom, S_cst, rom.frequencies / 1e9, edges=[0.5, 1.0, 1.5]))
```

`band_difference` returns the mean difference of the magnitudes per frequency band.

## Match CST's port references

For TEM ports, compare with `impedance_reference="line"` (the default), which is also CST's
default. CST reports TE/TM ports against their wave impedance, as the code does.

**See also:** [Coaxial line with a dielectric window](../tutorials/tem_transmission_line.ipynb)
and [C3794 two-cavity module vs CST](../tutorials/benchmarks/c3794_cavity_module.ipynb);
[Analysis & comparison API](../api/analysis.md).
