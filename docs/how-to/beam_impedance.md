# How to compute the beam impedance

When you need the longitudinal impedance of a structure for a beam travelling through it,
or the coupling between the beam and the port modes.

## Add a beam and solve

A beam is a line current of 1 A travelling along `proj.main_axis` at the speed of light.
Add it before meshing (then `generate_mesh()` curves the mesh to order 4, see below) or
after; the same `solve()` then adds its column and row:

```python
proj.create_primitive('taper', name='taper', R_left=50, R_right=25, L=100,
                      straight_left=30, straight_right=30)
proj.add_beam('beam')                          # on the axis: x = y = 0
proj.generate_mesh(maxh=0.008)                 # curve order 4 with a beam
proj.fds.solve(fmin=0.5, fmax=1.5, nsamples=11, nportmodes=3, order=3)

fom = proj.fds.fom
zpar = fom.beam_impedance()                    # Z_par = -z_b in ohm, per frequency
fom.plot_beam_impedance()
```

`fom.tilde_labels` lists the beam as `b(1)` after the port modes. `fom.s_tilde` is the
generalised scattering matrix `[[S, k], [h, z_b]]`, `fom.z_tilde` the same with every port
mode open.

## Place the beam, and read voltages along other lines

Give the transverse position in metres (`x=`, `y=` for the main axis Z). A second line
without current reads the beam's field elsewhere:

```python
proj.add_beam('beam', x=2e-3)                  # the same name replaces the beam
proj.add_beam_path('probe', x=3e-3)            # a voltage path, no current
proj.fds.solve(fmin=0.5, fmax=1.5, nsamples=11, nportmodes=3, order=3)
z_probe = proj.fds.fom.beam_impedance('beam', path='probe')
```

`proj.beams` and `proj.beam_paths` list what is defined; `proj.remove_beam(name)` and
`proj.remove_beam_path(name)` remove it, and the next `solve()` drops its results.

## Compute the transverse impedance

`add_transverse_beams(d)` adds two beams per transverse plane, at $\pm d$ from a centre
(the axis unless `x=`, `y=` are given). After a solve, `transverse_impedance(plane)` gives
$Z_\perp$ in Ω/m:

```python
proj.add_transverse_beams(0.01)                # dipole_x+, dipole_x-, dipole_y+, dipole_y-
proj.fds.solve(fmin=0.5, fmax=1.5, nsamples=11, nportmodes=3, order=3)
zx = proj.fds.fom.transverse_impedance('x')
zy = proj.fds.fom.transverse_impedance('y')    # also on rom and concat
```

Each beam's field is read on both lines of its plane, and with $Z(w, s)$ the longitudinal
impedance of the beam at $s$ read on the line at $w$,

$$
Z_\perp = \frac{c}{\omega}\,
\frac{Z(+,+) - Z(+,-) - Z(-,+) + Z(-,-)}{(2d)^2},
$$

the mixed derivative $\partial^2 Z_\parallel/\partial u_s\,\partial u_w$ (Panofsky–Wenzel). The
double difference keeps the part that is odd in both offsets, the dipole part: the
monopole and quadrupole parts drop out, and so does the kick a coupler gives a monopole
mode, which is odd in one offset only.

- Take $d$ small next to the aperture radius $a$, a quarter of it or less: a sextupole
  part is left at about $(d/a)^4$.
- The double difference of four similar numbers amplifies their discretisation error.
  Mesh finer than for $Z_\parallel$, and check $Z_\perp$ on two meshes.
- Near a dipole mode below cut-off, with open ports, $Z_\perp \approx j\frac{\omega}{c}
  (R/Q)_t\,\frac{\omega_0}{4(\omega_0 - \omega)}$, with the mode's `R/Q_t_x [Ohm]` from
  `get_figures_of_merit()`.

## Read the coupling to the port modes

```python
k = fom.s_tilde_dict['b(1)2(1)']               # wave sent into port 2, mode 1
h = fom.s_tilde_dict['1(1)b(1)']               # beam voltage of a wave from port 1, mode 1
z_open = fom.beam_impedance(ports='open')      # every port mode a magnetic wall
```

Keys are excitation first, as for S. With matched ports (`z_b`, the default) every port
mode is terminated in its reference impedance: the waves the beam excites leave without
reflection. With open ports (`z_oc`) every port mode is open-circuited: no port-mode
current, so for the port modes each port face is a magnetic wall, as in the eigenproblem
(`get_resonant_frequencies()`). The beam passes through the faces either way. With open
ports, a lossless structure below cut-off has a purely reactive beam impedance whose poles
are the resonances of the eigenproblem.

## Compute the HOM power per port

`get_hom_power()` gives the power a beam leaves in each port mode. Each frequency of the
result is one spectral line of the beam current, so solve at the lines first. For a train
of equal bunches, `bunch_train_spectrum()` gives the lines and their amplitudes:

```python
from cavsim3d.analysis import bunch_train_spectrum

f, I = bunch_train_spectrum(charge=1e-9, spacing=25e-9, sigma_t=30e-12, fmax=2.9)
rom.solve(frequencies=f, store_snapshots=False)     # GHz; also concat.solve()
P = rom.get_hom_power(I)                            # A, peak amplitude per line
P['P_port']                                         # {'1': W, '2': W, ...}
P['P_mode']                                         # {'1(1)': W, ...}, per port mode
```

With every port mode matched, the beam sends the wave $k\,I_p$ into a port mode, which
carries $\tfrac12 |k|^2 |I_p|^2\,\mathrm{Re}\,Z_{ref}/|Z_{ref}|$; an evanescent mode carries
none. Port numbers are those of the labels: `'2(1)'` is mode 1 of port 2. The current is
$i(t) = \sum_p \mathrm{Re}\{I_p e^{j\omega_p t}\}$, so `I` holds peak amplitudes, not rms.

- Carry every propagating mode of each port (`nportmodes`): a mode that is left out sees a
  magnetic wall, and the power it would take is reflected into the others.
- With lossless walls, the port powers add up to the power the beam loses,
  $\tfrac12\mathrm{Re}\,Z_\parallel |I_p|^2$ per line. A difference of more than a few per cent
  means the mesh is too coarse for $\mathrm{Re}\,Z_\parallel$.
- A narrow resonance takes power only when a line falls within its width. The lines of a
  real machine are fixed by the bunch spacing, so compute them at those frequencies, not
  on a grid.

## Add a beam to a solved project

```python
proj = EMProject(name='solved_cavity', base_dir='./simulations')
proj.add_beam('beam')
proj.fds.solve(fmin=1.0, fmax=1.3, nsamples=4, nportmodes=3, order=3)
```

With the request of the stored results, only the beam columns are solved; the port
results are kept as they are. This needs the stored port solutions
(`store_snapshots=True`, the default); otherwise everything is solved again.

## Join glued parts solved one by one

Solve a model of several glued parts part by part and join the parts' generalised
scattering matrices at the faces between them:

```python
proj.fds.solve(fmin=0.5, fmax=1.5, nsamples=11, nportmodes=3, order=3,
               per_domain=True, global_method=None)
joined = proj.fds.foms.concatenate()
zpar = joined.beam_impedance()
```

The join holds the frequencies of the full-order solve. It is exact for the modes carried
at the cut: put the cut in a uniform stretch of pipe, away from discontinuities, and carry
the modes the beam excites there (the TM0n modes for a beam on the axis of a round pipe).

## Join repeated or imported parts

Parts that are repeated (`n=`) or imported from another project are solved on their own
meshes and joined through their port modes. With a beam, the same two calls join them
with the beam:

```python
proj.add('cell', cell, n=3)                    # or proj.import_project(path, name='cell', n=3)
proj.generate_mesh(maxh=0.01)
proj.add_beam('beam')
proj.fds.solve(fmin=1.0, fmax=1.3, nsamples=31, nportmodes=3, order=3)
chain = proj.fds.foms.concatenate()            # the copies joined through their S~
zpar = chain.beam_impedance()
```

- The beam's position is given in the frame of the first part. Every next part sits with
  the face it is joined by centred on the face it joins, and each unique part is solved
  once, with the beam where it runs through that part.
- A copy placed further along the axis sees the beam later: its beam column carries the
  phase of its position. The beam impedance of the chain does not depend on where the
  first part sits.
- A part imported from a project solved without a beam gets its beam columns computed in
  this project, from the port solutions stored there; that project is never written. Its
  sweep must have the requested frequencies; otherwise the part is solved again here
  (`rerun=True` in a script), and the solve plan says so.
- The joined model holds the frequencies of the full-order solve. For other frequencies,
  join reduced models (next section).

## Reduce a model with the beam

A reduced model carries the beam when the full-order sweep kept its field snapshots
(`store_snapshots=True`, the default). It then gives $\tilde{S}$ and the beam impedance at
any frequency of its band, in a few milliseconds per frequency:

```python
proj.fds.solve(fmin=1.0, fmax=2.9, nsamples=39, nportmodes=3, order=3)
rom = proj.fds.fom.reduce(tol=1e-6)
rom.solve(fmin=1.0, fmax=2.9, nsamples=1901)
zpar = rom.beam_impedance()
```

Parts joined with the beam work the same way, at any frequencies:

```python
concat = proj.fds.foms.reduce(tol=1e-6).concatenate()
concat.solve(fmin=1.0, fmax=2.9, nsamples=1901)
zpar = concat.beam_impedance()
```

- The beam's phase runs along the structure, so the beam column changes with frequency at
  least every $v_b/L$ for a structure of length $L$. Sample the full-order sweep more
  finely than $v_b/(2L)$: about 120 MHz for a 1.3 m long cavity.
- A reduced model holds from 10 % of its band's width below the band to 10 % above it, and
  refuses a sweep that goes further.
- Judge the rank by the beam impedance, not by the singular values alone: the field near
  the walls dominates the snapshots, while the beam reads $E_z$ on its own line.
- An imported part needs a reduced model with the beam in its own project:
  `proj.fds.fom.reduce(tol)` there, after a solve with the beam.
- A narrow resonance (a loaded Q of 10^6 is a line 1 kHz wide at 1 GHz) falls between the
  points of any grid. Place points on it: `get_external_q()` gives the loaded resonances,
  and `rom.solve(frequencies=...)` (or `concat.solve`) takes any frequencies in GHz.
- `store_snapshots=False` keeps S, Z and $\tilde{S}$ only, not the reduced solution of every
  frequency: use it for long sweeps of large joined models.

## Get accurate beam results

- Curve the mesh to order 4. With a beam defined, `generate_mesh()` does so unless
  `curve_order=` is given; a mesh made before the beam was added (or an imported part's)
  keeps its curving, and the solve warns if curved walls are meshed to a lower order. The
  beam's own field is almost normal to the walls, and the facets of a coarser curving
  tilt it into a spurious datum.
- Use order 3, or a finer mesh than for the port modes: the beam's field is strongest at the
  walls closest to it. Check Z_par on two meshes.
- An off-axis beam needs a finer mesh than an on-axis one.
- The beam line need not be part of the mesh.

## Troubleshooting

- **`ValueError: Beam ... leaves the model ... through a wall`**: the beam line hits a
  wall. A beam must enter and leave through port faces across the main axis.
- **`NotImplementedError` naming `beta` or a material**: only a beam at the speed of light
  (`beta=1`) running through vacuum is implemented; materials off the beam line are fine.
- **Negative Re Z_par above a cut-off, with matched ports**: the propagating wave, faster
  than the beam, keeps exchanging energy with it along the pipe up to the port planes, so
  `z_b` of a model depends on the length of its pipes. Compare models with the same pipes.
- **`RuntimeError` from `joined.solve()`**: a model joined through scattering matrices
  holds the full-order frequencies only; join reduced models
  (`proj.fds.foms.reduce(tol).concatenate()`) for other frequencies.
- **`ValueError: The reduced model's beam data hold from ... to ... GHz`**: the sweep
  reaches beyond the band of the snapshots plus 10 % on each side. Solve the full-order
  model over a band that covers it, and reduce again.
- **A warning that the reduced model carries no beam**: the full-order sweep kept no field
  snapshots (`store_snapshots=False`); solve again with them.
- **A warning that parts have no reduced beam column**, and no beam in the joined model: a
  part's reduced model was made without the beam (or, for an imported part, its project
  has none); solve with the beam and reduce again.
- **`NotImplementedError: The copies of part ... see the beams at different places`**: the
  copies of a part are shifted across the axis against each other (their joined faces are
  not centred on one line), so each copy would need a solve of its own. Align the parts'
  joined faces on one axis.
- **`RuntimeError: Part ... has no beam results to join`**: the parts were solved before
  the beam was added; run `proj.fds.solve(...)` again.

**See also:** [Results](../reference/results.md#beams) for every beam quantity and label;
[§9 Beam excitation](../theory/beam.md) for the formulation;
[§10](../theory/beam_reduction.md) for reduced models with the beam.
