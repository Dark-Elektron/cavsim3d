# How to compute the beam impedance

When you need the longitudinal impedance of a structure for a beam travelling through it,
or the coupling between the beam and the port modes.

## Add a beam and solve

A beam is a line current of 1 A travelling along `proj.main_axis` at the speed of light.
Add it before or after meshing; the same `solve()` then adds its column and row:

```python
proj.create_primitive('taper', name='taper', R_left=50, R_right=25, L=100,
                      straight_left=30, straight_right=30)
proj.generate_mesh(maxh=0.008, curve_order=4)
proj.add_beam('beam')                          # on the axis: x = y = 0
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
without current reads the beam's field elsewhere, for example to estimate a transverse
kick:

```python
proj.add_beam('beam', x=2e-3)                  # the same name replaces the beam
proj.add_beam_path('probe', x=3e-3)            # a voltage path, no current
proj.fds.solve(fmin=0.5, fmax=1.5, nsamples=11, nportmodes=3, order=3)
z_probe = proj.fds.fom.beam_impedance('beam', path='probe')
```

`proj.beams` and `proj.beam_paths` list what is defined; `proj.remove_beam(name)` and
`proj.remove_beam_path(name)` remove it, and the next `solve()` drops its results.

## Read the coupling to the port modes

```python
k = fom.s_tilde_dict['b(1)2(1)']               # wave sent into port 2, mode 1
h = fom.s_tilde_dict['1(1)b(1)']               # beam voltage of a wave from port 1, mode 1
z_open = fom.beam_impedance(ports='open')      # every port mode a magnetic wall
```

Keys are excitation first, as for S. With matched ports (`z_b`) every port mode is
terminated in its reference impedance; with open ports (`z_oc`) a lossless structure below
cut-off has a purely reactive impedance, and its resonances are the poles of the
eigenproblem (`get_resonant_frequencies()`).

## Add a beam to a solved project

```python
proj = EMProject(name='solved_cavity', base_dir='./simulations')
proj.add_beam('beam')
proj.fds.solve(fmin=1.0, fmax=1.3, nsamples=4, nportmodes=3, order=3)
```

With the request of the stored results, only the beam columns are solved; the port
results are kept as they are. This needs the stored port solutions
(`store_snapshots=True`, the default); otherwise everything is solved again.

## Join parts solved one by one

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
Coupled parts (imported or repeated) and reduced models do not carry the beam yet.

## Get accurate beam results

- Mesh curved walls with `curve_order=4`. The beam's own field is almost normal to the
  walls, and the facets of a coarser curving tilt it into a spurious datum; the solve
  warns about it.
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
  holds the full-order frequencies only; solve the parts at the frequencies you need.

**See also:** [Results](../reference/results.md#beams) for every beam quantity and label;
[§9 Beam excitation](../theory/beam.md) for the formulation.
