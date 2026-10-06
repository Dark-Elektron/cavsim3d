# Beam excitation

A beam (a line current of 1 A travelling at the speed of light along the main
axis) is added to a project with `proj.add_beam(...)`. The same
`proj.fds.solve(...)` then adds one column per beam and one row per voltage
path to the port results: the generalised matrices

- `fom.s_tilde = [[S, k], [h, z_b]]`: `k` are the waves the beam sends into the
  port modes, `h` the beam voltage of each incoming wave, `z_b` the beam
  impedance with every port mode matched;
- `fom.z_tilde = [[Z, k_Z], [h_Z, z_oc]]`: the same with every port mode open.

`fom.beam_impedance()` is the longitudinal impedance `Z_par = -z_b`
(`ports='open'`: `-z_oc`). Labels: port modes `1(1)`, `2(1)`, ..., beams and
paths `b(1)`, `b(2)`, ...; dictionary keys are excitation first, as for S:
`b(1)b(1)` is `z_b`, `b(1)2(1)` is `k` from the beam into port 2 mode 1,
`1(1)b(1)` is `h`.

| file | what it shows |
|---|---|
| `collimator_impedance.py` | one solid, beam on the axis: `Z_par` of a collimator below the pipe cut-off, against Yokoya's small-angle estimate |
| `collimator_in_parts.py` | two glued parts solved one by one, joined through their `s_tilde` (`foms.concatenate()`), against the same mesh solved in one piece |
| `pillbox_rq_from_beam.py` | a beam added to a solved pillbox (only the beam columns are solved); R/Q from the pole of `Z_par`, against the eigenmode's R/Q |
| `lossy_ring.py` | a lossy ceramic ring at the pipe wall: `Re Z_par` is the absorbed power; lossless, it vanishes |
| `cells_coupled.py` | one pillbox cell repeated three times (`n=3`), solved once and joined with the beam, against the three cells in one piece; the same cell imported from a project solved without a beam |

Typical output (`pillbox_rq_from_beam.py`):

```text
TM010: f0 = 1.16485 GHz, R/Q = 162.56 Ohm (eigenmode)
beam added: S unchanged, S~ labels ['1(1)', '1(2)', '1(3)', '2(1)', '2(2)', '2(3)', 'b(1)']

 f/f0 - 1     Z_par open [Ohm]  R/Q from Z_par
   -0.020    -0.0325     +2040.0j          163.20
   -0.001    -0.6000    +40649.3j          162.60
   +0.001     0.5943    -40630.8j          162.52
```

Notes:

- The beam line need not be part of the mesh; the beam voltage is integrated
  piece by piece through the elements the line crosses.
- The beam impedance is a small effect of a large field: the beam's own field
  is almost normal to the walls, and how well the mesh follows curved walls
  matters most. With a beam defined, `generate_mesh()` curves the mesh to
  order 4; the solve warns if curved walls are meshed to a lower order.
- A beam must enter and leave through port faces across the main axis; only
  `beta = 1` (the speed of light) and a beam in vacuum are implemented.
  Materials off the beam line are fine (`lossy_ring.py`).
- Above the cut-off of a propagating mode, the matched `z_b` of a model
  depends on the length of its pipes: the wave and the beam keep exchanging
  energy along the pipe up to the port planes.
- Joined models (`foms.concatenate()` with a beam, for glued parts and for
  repeated or imported ones) hold the frequencies of the full-order solve;
  reduced models do not carry the beam yet.
- Repeated or imported parts are each solved once with the beam where it runs
  through them, in their own frame; the beam is given in the frame of the
  first part. A part imported from a project solved without a beam gets its
  beam columns computed in the importing project.

Run, for example:

```bash
python examples/beam/collimator_impedance.py
```
