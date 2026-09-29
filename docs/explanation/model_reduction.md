# How model order reduction works

A full-order sweep solves a large finite-element system at every frequency. Yet across a
band, the field of a structure varies in only a few independent ways: the solutions at
different frequencies are strongly alike. Model order reduction uses this to replace the
large system with a small one that gives the same answers inside the band. This page
explains what a reduced model is, what its tolerance controls, and where it can be trusted.

## Snapshots and the reduced basis

Each full-order solve stores its field solutions, one per frequency and port excitation:
the **snapshots**. The reduction (Proper Orthogonal Decomposition) computes a singular value
decomposition of the snapshot matrix. The left singular vectors are field patterns ordered
by how much of the snapshots they explain; the singular values measure that. The reduced
basis keeps the patterns whose singular value, relative to the largest, is above `tol`.

The system matrices are then projected onto this basis. The result has as many unknowns as
there are kept patterns, typically tens, against tens or hundreds of thousands for the
full-order system, and it can be solved at thousands of frequencies in a fraction of a
second. Fields can be rebuilt from the reduced solution through the same basis.

## What the tolerance means

The singular values of a well-sampled band fall by many orders of magnitude and then level
off at the round-off of the solver. `tol` places the cut:

- too loose (`1e-1`, `1e-2`), and patterns that matter are dropped; the model is
  wrong everywhere, not just slightly off;
- in the steep part of the curve (`1e-6` to `1e-9`), the reduced model is as accurate as
  the full-order samples it came from, and adds no error of its own;
- below the level-off, the extra patterns are numerical noise and only cost size.

The [reduced-order model tutorial](../tutorials/basics/reduced_order_model.ipynb) shows all
three on one waveguide.

## Inside and outside the training band

A reduced model interpolates between its snapshots. Inside the band they cover, and with
enough samples to resolve the band's features, it reproduces the full-order model. Outside
that band it has no information: its S-parameters and its resonances there can be wrong
without any sign of it. The resonances of a reduced model include spurious values below
and above the band, and it misses modes that none of its port excitations produced.

When reduced parts are joined, their training bands are checked: parts trained on bands
that do not overlap cannot be joined, and sweeping the joined model outside the common band
warns.

Features narrower than the sample spacing need care. A part on its own has resonances
(with magnetic walls at its ports) at which its response changes quickly; if such a
resonance falls between two samples, a joined model can show a narrow spike near it that
the full-order model does not have. More samples there, or a smaller `tol`, remove it.

## Reducing parts, then joining them

The reduced models of several parts are joined through the port modes of the faces they
share, by requiring the modal voltages and currents to match on both sides. This is done on
the reduced system matrices, not by cascading S-parameters, so evanescent modes carried at
a join couple correctly. The joined model is itself small, and can be reduced once more.

Because each part is reduced on its own, a part that does not change keeps its reduced
model, and a repeated part is reduced once. That is what makes long chains of cavities
cheap.

## Losses

A lossy structure has complex snapshots. Their real and imaginary parts are both kept as
snapshots, so the reduced basis stays real, and the loss matrices are projected like the
others. Everything above applies unchanged.

## Related

- [How to build a reliable reduced model](../how-to/reduce_well.md)
- [How a model is solved in pieces](architecture.md)
