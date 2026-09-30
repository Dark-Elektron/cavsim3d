# How to build a reliable reduced model

A reduced-order model (ROM) is only as good as the full-order samples it was built from.
This page shows how to choose the samples and the tolerance, how to check the result, and
what to do when a reduced or joined model shows features the full model does not have.

## Sample the band you need

Train the ROM on the band you will use it in, with a few extra percent on either side:

```python
proj.fds.solve(fmin=0.9, fmax=1.9, nsamples=31)      # to use the ROM on 1.0 - 1.8 GHz
rom = proj.fds.fom.reduce(tol=1e-6)
rom.solve(fmin=1.0, fmax=1.8, nsamples=2000)
```

A ROM does not know anything outside the band it was trained on: its results there, and
its resonances there, can be wrong without warning. Its resonance list therefore stops
10 % beyond the band's edges (`get_resonant_frequencies(fmin=..., fmax=...)` lists others).
When reduced parts trained on different bands are joined, disjoint bands raise an error and
sweeping outside the common band warns.

## Choose the tolerance

`tol` is relative to the largest singular value of the snapshots. Start with `1e-6`; go to
`1e-9` for high-Q resonances or when the joined model is compared to a reference at the
$10^{-4}$ level. `max_rank` caps the size:

```python
rom = proj.fds.fom.reduce(tol=1e-9, max_rank=80)
roms = proj.fds.foms.reduce(tol=1e-9)                # one ROM per part / domain
```

## Check the reduction

```python
print(rom.reduced_dimensions)                          # unknowns per domain
fig, ax = rom.plot_singular_values()                   # the red line marks the cut
```

The singular values should fall by several orders of magnitude before the cut. If they
level off above `tol`, the sweep has too few samples for the band: add samples.

Then compare the ROM with the full-order samples it came from:

```python
import numpy as np

fom_res = proj.fds.solve(fmin=0.9, fmax=1.9, nsamples=31)    # returns the stored results
rom_res = rom.solve(fmin=0.9, fmax=1.9, nsamples=31)          # the same frequencies
print(np.max(np.abs(rom_res["S"] - fom_res["S"])))
```

## Fix spikes that the full model does not have

A narrow spike or dip in a reduced or joined model, between two full-order samples, often
sits at a resonance of one part on its own (with magnetic walls at its ports). Near such a
frequency the part's response changes quickly and the samples miss it. Either:

- add full-order samples around the spike (a denser sweep, or a band that ends before it), or
- lower `tol` so more of the snapshot information is kept.

To list a part's own resonances, solve that part on its own in a project and call
`proj.fds.fom.get_resonant_frequencies(n_modes=10)`.

## Reduce a joined model further

A joined model can itself be reduced, for example before many repeated sweeps:

```python
concat = proj.fds.foms.reduce(tol=1e-9).concatenate()
concat.solve(fmin=1.0, fmax=1.8, nsamples=200)     # its snapshots train the next reduction
concat_rom = concat.reduce(tol=1e-10)              # stored as concat.rom
concat_rom.solve(fmin=1.0, fmax=1.8, nsamples=5000)
```

**See also:** [How model order reduction works](../explanation/model_reduction.md);
[Build a reduced-order model](../tutorials/basics/reduced_order_model.ipynb) (tutorial).
