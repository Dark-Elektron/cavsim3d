# Analysis & Comparison

Utilities for comparing the results of several models: solve results at any stage (FOM, ROM,
concatenated system), a `CSTResult`, or plain arrays indexed `[frequency, response, excitation]`.

```python
from cavsim3d.analysis import keep_port_modes, network_matrix, plot_matrix, band_difference

S, Z = keep_port_modes(concat, ["1(1)", "2(1)", "3(1)"])   # other modes open-circuited
S_ref = network_matrix(cst, ["1(1)", "2(1)", "3(1)"])       # CST keys read response-first
fig, axs = plot_matrix({"CST": S_ref, "FEM": S}, freq=f_ghz)
band_difference(S, S_ref, f_ghz, edges=[0.1, 0.5, 1.0])
```

The convergence of an iterative full-order solve is plotted by the result itself:
`proj.fds.fom.plot_residual(per_excitation=True)`.

## Network parameters

::: cavsim3d.analysis.network
    options:
      members:
        - network_matrix
        - keep_port_modes
        - port_mode_labels
        - plot_matrix
        - plot_entries
        - band_difference
      show_root_heading: false

## Beam current spectra

The spectral lines of a bunch train, for the HOM power of a structure
(`get_hom_power()`, see [How to compute the beam impedance](../how-to/beam_impedance.md#compute-the-hom-power-per-port)).

::: cavsim3d.analysis.beam
    options:
      members:
        - bunch_train_spectrum
      show_root_heading: false

## Eigenmode spectra

::: cavsim3d.analysis.eigenmodes
    options:
      members:
        - compare_spectra
        - cluster_frequencies
        - SpectrumComparison
      show_root_heading: false
