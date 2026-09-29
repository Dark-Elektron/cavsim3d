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
| `coax_z_reference.py` | which reference impedance Z is expressed in | Z matches the exact line Z (CST's reference) |
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

## 2. Z is referenced to the LINE impedance of the TEM mode, as in CST

The port modes are normalised to their wave impedance, and Z is then rescaled to
the reference CST reports: the **line impedance for a TEM mode** (and the
power-voltage line impedance `Z_PV` for a quasi-TEM one), the **wave impedance
for every TE/TM mode**, the TE11 modes of a coaxial port included. Rescaling Z
and the reference together leaves S unchanged, so S does not depend on the
choice; Z does. `coax_z_reference.py`:

```
Z0 analytic (line)    = 49.940 ohm
reference used for S  = 49.937 ohm       (line impedance, radii fitted from the mesh)
|Z21|cav / |Z21|exact = 0.995 .. 1.003
```

Near beta*L = n*pi/2 (1.5, 3.0, 4.5 GHz here) the analytic cot / 1/sin are
singular, so relative errors there are the formula blowing up, not solver error.

`impedance_reference="wave"` in the solve config refers TEM modes to the wave
impedance instead (376.730 ohm in air); Z then comes out 376.730 / 49.940 = 7.54
times larger.

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
