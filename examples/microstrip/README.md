# Microstrip line — quasi-TEM port examples

A microstrip line (FR-4 substrate, PEC strip + ground plane) driven with the
**quasi-TEM port** boundary condition — an inhomogeneous (substrate + air)
cross-section solved with a mixed `HCurl(Et) × H1(Ez)` eigenproblem.

Geometry: [`cavsim3d.geometry.MicrostripLine`](../../cavsim3d/geometry/microstrip.py).
Default dimensions reproduce the CST `microstrip_line` reference model
(L=40, W=20, w=3.1, h=1.6, t=0.5 mm; eps_r=4.3).

## Files

| file | what it shows | status |
|---|---|---|
| `microstrip_impedance_sweep.py` | characteristic impedance `Z_PV` and `eps_eff` vs strip width | **validated** — ~50 Ω at w=3.1 mm, textbook trend |
| `microstrip_qtem_modes.ipynb` | the qTEM mode used vs CST *Port1 e1*; full spectrum; S/Z via `rom.plot_s/plot_z` | port mode **matches CST** (β within ~1%) |
| `microstrip.py` | full FOM solve + S-parameters vs CST | see caveat below |
| `cst_compare.py` | helpers to read CST ASCII exports | — |

## What is validated

The **quasi-TEM port mode** is correct: the effective permittivity and
propagation constant match analytic microstrip theory and the CST port
information to ~1% (e.g. at 5 GHz β ≈ 190 vs CST 191, wave impedance ≈ 207 vs
205 Ω), and the power-voltage line impedance `Z_PV` reproduces the ~50 Ω design.

## Caveat: through-line S-parameters

The microstrip **S-parameters do not yet match CST**. The port mode is right,
but a microstrip is an **open** structure: with no absorbing port boundary in
the FOM, the finite air box resonates and the 40 mm line rings like a cavity
(periodic notches in |S21|). A closed rectangular waveguide through the *same*
solver gives a flat |S21| = 1 (see `../rectangular_waveguide/`), confirming the
port machinery is otherwise correct. A clean microstrip S needs an absorbing /
radiation boundary on the air box (or a tuned / de-embedded box) — a solver
addition separate from the port boundary condition.

The CST reference is also **lossy** (FR-4 tan-δ, copper) while cavsim3d here is
**lossless PEC**, so expect small offsets even where they agree.

## CST reference data

The comparison scripts look for the CST exports in
`C:\Users\Soske\Documents\CEM2\cst\microstrip_line\Export` (adjust
`cst_compare.default_cst_dir` for your machine). Missing CST data is handled
gracefully — the cavsim3d curves still plot.
