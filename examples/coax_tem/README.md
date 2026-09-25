# Coaxial line — TEM ports against closed form

An air-filled coaxial line is the cleanest TEM benchmark available: every
quantity has an exact answer, so it isolates the port machinery from anything
geometry- or reference-data-dependent.

```
Z0   = (eta0 / 2pi) * ln(b/a)          beta = k0 = omega/c
S11  = 0                               S21  = exp(-j*beta*L)
Z11  = -j*Z0*cot(beta*L)               Z21  = -j*Z0/sin(beta*L)
```

Geometry is built inline from `netgen.occ` (two cylinders), so these scripts
need no STEP file and no external reference data.

## Files

| file | what it checks | result |
|---|---|---|
| `coax_tem_analytic.py` | analytic TEM port path: S-parameters vs closed form | **passes** — see below |
| `coax_z_reference.py` | which reference impedance Z is expressed in | explains the CST Z offset |
| `coax_qtem_singularity.py` | the qTEM eigenproblem on a homogeneous cross-section | diagnoses + fixes the singularity |

## 1. The TEM port path is correct

`coax_tem_analytic.py`, 1–10 GHz, a=1 mm, b=2.3 mm, L=50 mm:

```
port1/port2: geometry=coaxial, mode0 type=TEM, kc=0
mean |S11| = 4.96e-04     (exact 0)
mean |S21| = 1.000000     (exact 1)
phase error vs exp(-j*k0*L): mean +0.22 deg, max 0.38 deg
```

Detection, the analytic TEM mode, the port basis and the Z->S conversion are all
sound. A homogeneous coax never touches the Arnoldi solver — it uses the
closed-form modes from `_generate_coaxial_modes`.

## 2. Z is referenced to the WAVE impedance, not the line impedance

This is the reason cavsim3d S-parameters can agree with CST while the
Z-parameters sit at a constant offset. `coax_z_reference.py` measures it:

```
reference used for S  = 376.730 ohm      (modal wave impedance)
Z0 analytic (line)    =  49.940 ohm
|Z21|cav / |Z21|exact = 7.49 .. 7.57     (constant across frequency)
376.730 / 49.940      = 7.543
```

For a **uniform** through-line there is no impedance step, so S is essentially a
pure phase whichever common reference is used — S is insensitive to the choice.
Z scales linearly with it, so it is not.

To compare Z against a tool that references the line impedance (CST does),
renormalise through S:

```python
from cavsim3d.solvers.base import ParameterConverter
Z_cmp = ParameterConverter.s_to_z(S_cavsim3d, Z0_line)
```

which reproduces the exact line Z to **0.45-0.9%**. (Near beta*L = n*pi/2 the
analytic cot / 1/sin are singular, so relative errors there are the formula
blowing up, not solver error.)

Note `get_port_wave_impedance` already returns the power-voltage line impedance
`Z_PV` for **quasi-TEM** ports, specifically to match CST. TE/TM/TEM ports fall
through to the wave impedance.

## 3. The qTEM eigenproblem: singularity, cause and fix

The quasi-TEM solver exists for *inhomogeneous* cross-sections, but handed a
homogeneous one it must reproduce the exact TEM answer — which makes this coax a
yardstick for a path that otherwise has none. `coax_qtem_singularity.py` sweeps
mesh density and shift offset.

Two independent defects were found and fixed in `_solve_port_qtem`:

**Structurally singular matrix.** `fesEt = HCurl(mesh, definedon=port_region)`
carries DOFs over the whole 3-D mesh while only the port face is assembled, so
the off-face DOFs are identically zero rows in both `a` and `m`. `FreeDofs()`
removes only Dirichlet DOFs, so the solver was handed a singular block —
**8082 empty rows out of 9028 "free" DOFs** on this coax port. Singular for any
shift, which is why changing the shift alone never helped. Fixed by
intersecting with `fes.GetDofs(port_region)`: 762 DOFs, none empty.

**The shift sat on the eigenvalue.** For a homogeneous cross-section the mode is
at exactly `beta^2 = k0^2*eps_r`, so `shift = k0^2*eps_max` lands on it and
`(a - shift*m)` is singular by construction:

| shift | modes returned |
|---|---|
| coincident | 3 spurious modes, `eps_eff` = 1.05 / 1.11 / 1.14 (5-14% error) |
| offset x1.15 | 1 physical mode, `eps_eff` = 1.000000 (0.000% error) |

The spurious cluster is also what made mode ORDER look unstable: the
descending-beta sort was faithfully ranking numerical artefacts. The offset is
`SHIFT_OFFSET` in `cavsim3d/solvers/ports.py`.

Verified exact at 4x and 16x finer meshes.
