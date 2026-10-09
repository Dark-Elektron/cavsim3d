# How to get the figures of merit of resonances

This page shows how to get the loaded and external Q of each resonance, and the R/Q, wall
Q, geometry factor, peak surface fields and the other cavity figures of merit of an
eigenmode. They work on the full-order model, the reduced model and a joined model
(`concat`); the external Q needs a reduced or joined model.

## External and loaded Q

Reduce the full-order model, then ask the reduced model for the resonances in a band:

```python
rom = proj.fds.fom.reduce(tol=1e-9)
q = rom.get_external_q(fmin=1.8, fmax=2.6)        # GHz

for k, f in enumerate(q["frequencies"]):          # Hz
    print(f"{f / 1e9:.5f} GHz  Q_L = {q['Q_L'][k]:.0f}  "
          f"Qext(port1) = {q['Qext']['port1'][k]:.0f}")
```

The result holds, for each loaded resonance:

- `frequencies` and `Q_L`: frequency and loaded Q, every port terminated in its reference
  impedance, as for the S-parameters;
- `Qext`: the external Q of each port, all its modes together, and `Qext_mode` per port
  mode, keyed like `"port1(2)"`;
- `mode_index` and `f_closed`: the resonance of the closed problem (magnetic walls at the
  ports) it belongs to, a different one for each, for `get_eigenmode()`, `get_rq()` and
  `get_figures_of_merit()`.

A port mode that is evanescent at the resonance takes no power: its `Qext` is `inf`.
Wall and dielectric losses are not included here (they are in the unloaded Q below).
Strongly damped resonances of feed lines also appear, with a Q_L near 1; select the modes
of interest by `Q_L` or by frequency. Why the loaded problem is solved, rather than the
closed one, is explained in [Ports and port modes](../explanation/ports.md#loaded-resonances).

## All figures of merit of one mode

`get_figures_of_merit()` takes the index of an eigenmode in the list
`get_resonant_frequencies()` returns (for a reduced model, the modes near its training band):

```python
f_res = rom.get_resonant_frequencies()
fm = rom.get_figures_of_merit(0)                   # beam along z, on the axis
print(fm["R/Q [Ohm]"], fm["G [Ohm]"], fm["Epk/Eacc []"], fm["Bpk/Eacc [mT/MV/m]"])
```

The keys and their units are those of cavsim2d; [Results](../reference/results.md#figures-of-merit)
lists them. Voltages, fields and losses are for the mode scaled to a stored energy of
1 J; the ratios (R/Q, G, Epk/Eacc, ...) do not depend on that scale. For several modes,
loop over the indices; the list of dictionaries goes directly into a `pandas.DataFrame`:

```python
for i in range(4):
    fm = rom.get_figures_of_merit(i)
    print(f"{fm['freq [MHz]']:9.3f} MHz  R/Q = {fm['R/Q [Ohm]']:7.2f} Ohm  "
          f"G = {fm['G [Ohm]']:6.1f} Ohm")
```

### Set the wall material

The wall Q, `Ploss`, `Rsh` and the unloaded `Q` use the surface resistance of the walls;
the geometry factor `G` does not depend on it.

- `conductivity`: wall conductivity in S/m; the default is copper, 5.96e7.
- `surface_resistance`: a fixed surface resistance in ohm instead, for a superconducting
  wall for example: `surface_resistance=10e-9`.
- `walls`: the conducting boundaries, as a boundary-name pattern (`"default|coupler"`).
  The default is the solver's `bc`. Port faces are never walls: they are magnetic walls in
  the eigenproblem.

### Set the beam

- `axis`, `offset`, `span`: the beam line, as for `get_rq()` below.
- `beta`: the particle velocity over the speed of light (default 1).
- `active_length`: the length in metres that `Eacc` is normalised to. An elliptical cavity
  gives its own, 2 L n_cells (the cavsim2d convention); otherwise the default is the length
  of the beam line, beam pipes included.

### Multi-cell cavities: field flatness and cell coupling

Give the number of cells to get the field flatness `ff [%]`, and the 0 and π modes of the
passband to get the cell-to-cell coupling:

```python
fm = rom.get_figures_of_merit(i_pi, n_cells=9)    # adds "ff [%]"
kcc = rom.get_cell_coupling(i_0, i_pi)            # %, 2 (f_pi - f_0) / (f_pi + f_0)
```

An elliptical cavity gives `n_cells` itself.

### Dipole modes: transverse R/Q and kick factor

`Vt`, `Et`, `R/Q_t` and `k_kick` are the transverse kick at `offset`, from the gradient of
the longitudinal voltage (Panofsky–Wenzel). For a dipole mode they are the values cavsim2d
reports for m = 1; for a monopole mode on the axis they are close to zero. `kick_step`
sets the transverse step of the gradient in metres; by default it is 2 % of the smallest
transverse extent, reduced until the shifted lines stay inside the beam aperture.

The same values per plane are `Vt_x`, `Vt_y`, `R/Q_t_x`, `R/Q_t_y`, `k_kick_x` and
`k_kick_y` (for a beam along z; along x the planes are y and z). The couplers split a
dipole pair into two polarisations at angles of their own, and the thresholds of the two
planes differ, so compare each plane with its own threshold. `R/Q_t_x + R/Q_t_y` equals
`R/Q_t`.

### Lossy materials

With a loss tangent or a conductivity on a material (see [Assign materials](materials.md)),
the result adds `Q_wall`, `Q_diel` and `Pdiel`, and `Q` is the unloaded Q of walls and
materials together, 1/Q = 1/Q_wall + 1/Q_diel. With several materials, `U_frac_<name>`
is each material's share of the electric energy and `Epk_<name>` the peak field inside it.

### Peak fields

A peak field is the largest value of the discrete field, not an integral, so it converges
more slowly than R/Q, G or Q: with `order=2` it can be a few percent off, with `order=3`
or a finer mesh on the walls it settles. Check it the way you check any result, by
refining (see [Choose the mesh size and element order](mesh_and_order.md)).

The field at a re-entrant edge (an iris without rounding, the joint of a beam pipe and an
end wall) is singular. A peak field that sits on such an edge grows as the mesh is refined
and does not converge; round the edge in the geometry to get a converged value.

## R/Q only

`get_rq()` returns the frequency, the voltage, the stored energy and R/Q, without the
surface integrals:

```python
rq = rom.get_rq(0)                                  # beam along z, on the axis
print(rq["frequency"], rq["RQ"])                    # Hz, ohm
```

R/Q is $V^2/(\omega U)$: $V$ is the voltage a particle at the speed of light gains along a
line parallel to the beam axis, $U$ the stored energy. It is the convention of cavsim2d,
twice the circuit definition $V^2/(2\omega U)$.

- `axis`: the beam direction, `"X"`, `"Y"` or `"Z"`.
- `offset`: the line's transverse position in metres, in the order of the two other axes
  (x, y for a beam along z). A dipole mode has no voltage on the axis: give an offset, in
  both transverse directions to catch both polarisations, or use the transverse R/Q above.
- `span`: start and end of the line along the axis in metres; by default the whole model.
  Parts of the line outside the mesh, inside a conductor, contribute nothing.

`rq["V"]` is the magnitude of the voltage and `rq["V_complex"]` the voltage with its
phase. The phase refers to the model's own coordinate along the axis and to the mode as
the last spectrum gave it, so the voltages of one mode on several lines can be combined.
A mode that mixes multipoles (above the beam-pipe cutoff, where the couplers mix them)
splits into its parts this way:

```python
d = 0.02                                            # m
vp, vm = (rom.get_rq(i, offset=(x, 0.0))["V_complex"] for x in (d, -d))
v_dipole_x = (vp - vm) / 2                          # odd in x
v_even = (vp + vm) / 2                              # monopole and quadrupole parts
```

## On a joined model

The same calls work on the joined model of several parts:

```python
concat = proj.fds.foms.reduce(tol=1e-9).concatenate()
q = concat.get_external_q(fmin=1.2, fmax=1.4)
f_res = concat.get_resonant_frequencies()
fm = concat.get_figures_of_merit(0)
```

The parts of a glued model share one mesh. The parts of a coupled model (repeated or
imported) each have their own mesh; they are laid end to end along the beam axis in the
order of the parts list, and `offset` is taken in each part's own coordinates, so the
parts must share their transverse axis.

**See also:** [Find resonances and look at fields](../tutorials/basics/resonances_and_fields.ipynb)
(tutorial); [Cavity figures of merit: validation](../tutorials/benchmarks/figures_of_merit.ipynb)
(benchmark); [Ports and port modes](../explanation/ports.md#loaded-resonances) (explanation);
[Results](../reference/results.md#figures-of-merit) (reference).
