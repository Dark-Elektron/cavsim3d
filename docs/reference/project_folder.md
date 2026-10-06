# Project folder

The files a project writes, by folder. `<part>` is a part or domain name (`inlet`,
`subdomain1`, ...); a single-part model uses `global` for its reduced model and the mesh
material name for its full-order files.

## Top level

| Path | Content |
|---|---|
| `project.json` | project settings: name, element order, port-mode count, main axis, part name, flags, and the beams and voltage paths (`beam`) |
| `timing.json` | wall time of each stage (FOM, reduction, ROM and joined-model solves) |
| `geometry/history.json` | how the geometry was built, replayed on reopening |
| `geometry/*.step` | STEP export of the model (`cavsim3d.geometry.step`, `assembly.step`, `source_model.step`) |
| `geometry/components/<part>.step` | one STEP file per part of a chain |
| `mesh/mesh.pkl`, `mesh/fes.pkl` | mesh and finite-element space |
| `mesh/mesh_<part>.pkl`, `mesh/fes_<part>.pkl` | per-part mesh of a coupled (netlist) model |

## `fds/`

| Path | Content |
|---|---|
| `config.json` | the request the results belong to (band, samples, order, element kind, port settings, materials) and the beam definition of the stored beam results (`beam`) |
| `solve.log` | log of the last full-order solve |
| `port_modes/port_modes.pkl` | port modes |
| `port_modes/beam_port_fields.pkl` | with a beam: the beam's electrostatic potential on every port face it crosses |
| `imports.json` | imported projects: source path, mode (`reference` / `copy`), fingerprint |
| `sections.json` | parts of a chain solved in this project: the solve settings and geometry each was solved for, and its port data (reused by the next `solve()`, and needed to reduce it again) |
| `checkpoint/<sweep>/sample_*.npz` | finished samples of a running sweep; deleted when the sweep completes |

## Stage folders

Each stage folder holds the same five subfolders, plus the next stage:

| Subfolder | Content |
|---|---|
| `matrices/` | system matrices: `K_<part>.h5`, `M_<part>.h5`, `B_<part>.h5` (full order); `A_r_<part>.h5`, `B_r_<part>.h5`, `W_<part>.h5`, `Q_L_inv_<part>.h5` (reduced); `A.h5`, `B.h5`, `W.h5` (joined) |
| `eigenmodes/` | computed resonances and mode vectors |
| `s/` | S-parameters: `s_<part>.h5` |
| `z/` | Z-parameters: `z_<part>.h5` |
| `snapshots/` | field solutions and their frequencies: `snapshots_<part>.h5`, `snapshots.h5` |

plus `metadata.json` and, where applicable, `reduce.log`, `solve.log` and
`structures.json` (the reduced parts' ports, port-mode fingerprints and training band, used
to join and to import them).

With a beam, a full-order stage folder also holds, per part (`<part>` is `global` for a
single part):

| Path | Content |
|---|---|
| `z_tilde/z_tilde_<part>.h5` | $\tilde{Z} = [[Z, k_Z], [h_Z, z_{oc}]]$ with its labels and the beam definition |
| `s_tilde/s_tilde_<part>.h5` | $\tilde{S} = [[S, k], [h, z_b]]$, the same |
| `snapshots_beam/snapshots_beam_<part>.h5` | the beam's scattered field per sample |
| `matrices/beam_<part>.h5` | the beam data that do not depend on frequency: Gauss points, weights and evaluation matrix of each voltage path; the load of each port face |

The port files (`s/`, `z/`, `snapshots/`) are the same with and without a beam. A model
joined from full-order parts with a beam (`fds/foms/concat/`) holds
`s_tilde/s_tilde.h5` and no matrices.

| Stage | Single part | Several parts |
|---|---|---|
| Full-order model | `fds/fom/` | `fds/foms/` |
| Reduced model | `fds/fom/rom/` | `fds/foms/roms/` |
| Joined model | – | `fds/foms/roms/concat/` |
| Reduced joined model | – | `fds/foms/roms/concat/rom/` |
| Joined full-order models | – | `fds/foms/concat/` |
