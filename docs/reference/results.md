# Results

What a solve returns and how results are addressed: the dictionary returned by `solve()`,
the result objects, the parameter labels and the units.

## The dictionary returned by `solve()`

`proj.fds.solve()` (a single part, or several parts solved in one piece):

| Key | Type | Content |
|---|---|---|
| `frequencies` | array `[n_f]` | frequencies, **Hz** |
| `S` | complex array `[n_f, n_pm, n_pm]` | S-matrices; `S[k, i, j]` is the response at port mode `i` to an excitation at port mode `j` |
| `Z` | complex array `[n_f, n_pm, n_pm]` | Z-matrices, same layout, ohm |
| `S_dict`, `Z_dict` | dict | the same entries keyed by label (below), plus `'frequencies'` |
| `ports` | list | external ports, in matrix order |
| `external_ports`, `internal_ports`, `all_ports` | list | port names by role |
| `domains`, `n_domains`, `is_compound` | | the model's domains |
| `snapshots` | dict | field solutions per domain (used by `reduce()`) |
| `residuals` | dict | iterative-solver convergence per sample |

`n_pm` is the total number of port modes: the sum of `nportmodes` over the external
ports. Port modes are ordered port by port, and by mode within a port.

`proj.fds.solve()` on a model of several parts solved per part returns per-domain results
instead of `S` and `Z` (`Z_per_domain`, `S_per_domain`, `domain_port_map`); for a coupled
(netlist) chain it returns `{'netlist_sections': [...]}`. The joined result comes from
`concat.solve()`.

`rom.solve()`, `roms.solve()` and `concat.solve()` return `frequencies`, `S`, `Z`, `S_dict`,
`Z_dict` with the same layout.

With a beam (`proj.add_beam()`), the dictionary also holds:

| Key | Type | Content |
|---|---|---|
| `S_tilde` | complex array `[n_f, n_pm + n_path, n_pm + n_beam]` | generalised scattering matrix $\tilde{S} = [[S, k], [h, z_b]]$ |
| `Z_tilde` | complex array, same layout | $\tilde{Z} = [[Z, k_Z], [h_Z, z_{oc}]]$ |
| `tilde_labels` | (list, list) | row and column labels of both |
| `S_tilde_per_domain`, `Z_tilde_per_domain` | dict | the same per part, for a model solved part by part |

## Beams

A beam is a line current of 1 A; every beam quantity is per ampere. The rows of
$\tilde{S}$ and $\tilde{Z}$ are the port modes and then the voltage paths, the columns the
port modes and then the beams. Every beam is also a path, and comes first:

| Block | Meaning | Unit |
|---|---|---|
| $k$ (port-mode rows, beam columns) | wave the beam sends into each port mode, every port mode matched | $\sqrt{\Omega}$ |
| $h$ (path rows, port-mode columns) | beam voltage of a unit incoming wave | $\sqrt{\Omega}$ |
| $z_b$ (path rows, beam columns) | beam voltage per beam current, every port mode matched | Ω |
| $k_Z$, $h_Z$, $z_{oc}$ | the same in $\tilde{Z}$, every port mode open (magnetic walls) | Ω |

The longitudinal impedance is $Z_\parallel = -v/i = -z_b$.

Accessors of a full-order result solved with a beam (`proj.fds.fom`, `proj.fds.foms[i]`),
of a model joined from such results (`proj.fds.foms.concatenate()`, for glued parts and
for repeated or imported ones), of a reduced model with the beam (`proj.fds.fom.rom`) and
of reduced parts joined with it (`proj.fds.foms.roms.concat`), after their `solve()`:

| Accessor | Returns |
|---|---|
| `has_beam` | True if the result has beam rows and columns |
| `s_tilde`, `z_tilde` | the arrays above (a joined model derives $\tilde{Z}$ from $\tilde{S}$) |
| `s_tilde_dict`, `z_tilde_dict` | the same keyed by label, plus `'frequencies'` |
| `tilde_labels`, `beam_names` | (rows, columns); label → beam or path name |
| `beam_impedance(beam=None, path=None, ports='matched')` | $Z_\parallel$ per frequency, Ω: `-z_b`; `ports='open'`: `-z_oc`. `beam`, `path`: name or label; defaults: the first beam, read on its own line |
| `beam_field(i, beam=None, total=True)` | the beam's field at sample `i`: $E_s + E^{free}$ as a CoefficientFunction, or (`total=False`) the scattered field $E_s$ as a GridFunction (full-order results only) |
| `plot_s_tilde()`, `plot_z_tilde()`, `plot_beam_impedance()` | plots, arguments as `plot_s()` |

Labels: port modes as below, beams and paths `'b(1)'`, `'b(2)'`, ... in the order of
`proj.beam_paths`. Keys are excitation first: `'b(1)b(1)'` is $z_b$, `'b(1)2(1)'` is $k$ from
beam 1 into port 2 mode 1, `'1(1)b(2)'` is $h$ from port 1 mode 1 onto path 2.

$k$ and $h$ carry the beam's phase $e^{\mp j k_b s}$ along the main axis, with $s$ measured in
the model's frame; for repeated or imported parts, that is the frame of the first part.
$z_b$ does not depend on it.

## Result objects

| Object | Holds | Created by |
|---|---|---|
| `proj.fds.fom` | full-order result of a single part (or of a whole mesh) | `proj.fds.solve()` |
| `proj.fds.foms` | one full-order result per part or domain | `proj.fds.solve()` on several parts |
| `proj.fds.fom.rom` | reduced model | `fom.reduce(tol)` |
| `proj.fds.foms.roms` | one reduced model per part | `foms.reduce(tol)` |
| `proj.fds.foms.roms.concat` | joined model | `roms.concatenate()` |
| `proj.fds.foms.concat` | joined model at the full-order frequencies (glued parts; with a beam, also repeated or imported parts) | `foms.concatenate()` |
| `concat.rom` | reduced joined model | `concat.reduce(tol)` |

Each has `frequencies` (Hz), `S_dict`, `Z_dict`, `plot_s()`, `plot_z()`, `compare_s()`,
`compare_z()`. After reopening a project, each is loaded from disk on first access.

A joined model has fewer ports than its parts: the ports where parts join are gone, and
the others are numbered part by part, in each part's own port order.
`concat.print_port_map()` prints which part (a copy of a repeated part is
`<name>_<copy>`) and which of its own ports each one is; `concat.port_map()` returns the
same as a list of dictionaries.

## Parameter labels

`'<port>(<mode>)<port>(<mode>)'`, **excitation first, response second**, 1-based:

| Label | Meaning | Matrix entry |
|---|---|---|
| `'1(1)1(1)'` | $S_{11}$, mode 1 | `S[:, 0, 0]` |
| `'1(1)2(1)'` | $S_{21}$: excite port 1 mode 1, read port 2 mode 1 | `S[:, n1, 0]` |
| `'1(3)2(1)'` | port 1 mode 3 in, port 2 mode 1 out | `S[:, n1, 2]` |

`n1` is the number of modes on port 1. A port mode alone is written `'<port>(<mode>)'`;
`cavsim3d.analysis.port_mode_labels(model)` lists them in matrix order.

CST Studio Suite exports name entries response first (`S2,1`); `CSTResult` and
`cavsim3d.analysis.network_matrix` handle both conventions.

## Plot types

`plot_s()` and `plot_z()` take `plot_type=`:

| Value | Plots |
|---|---|
| `"db"` | $20 \log_{10} |x|$ |
| `"mag"` | $|x|$ |
| `"phase"` | phase in degrees |
| `"re"`, `"im"` | real, imaginary part |

## Figures of merit

`get_figures_of_merit(mode_index)` on `proj.fds`, `fom`, a reduced model or a joined model
returns a dictionary. Each key carries its unit, as in cavsim2d. Voltages, fields and
losses are for the mode scaled to a stored energy of 1 J.

| Key | Quantity |
|---|---|
| `freq [MHz]` | resonant frequency |
| `Q []` | unloaded Q: walls and lossy materials |
| `Vacc [MV]`, `Eacc [MV/m]` | voltage for a charge at `beta` c along the beam line; over `active_length` |
| `Epk [MV/m]`, `Hpk [A/m]`, `Bpk [mT]` | peak fields on the walls |
| `ff [%]` | field flatness, min/max of the cells' on-axis peaks (with `n_cells` > 1) |
| `Rsh [MOhm]` | shunt impedance, R/Q × Q |
| `R/Q [Ohm]` | $V_{acc}^2/(\omega U)$, linac convention (twice the circuit one) |
| `Epk/Eacc []`, `Bpk/Eacc [mT/MV/m]` | peak-field ratios |
| `G [Ohm]` | geometry factor, $Q_{wall} R_s$ |
| `GR/Q [Ohm^2]` | G × R/Q |
| `U [J]` | stored energy (1) |
| `Ploss [W]` | wall loss, $\frac{R_s}{2}\oint \lvert H \rvert^2 dS$ |
| `k_loss [V/pC]` | loss factor of the mode, $V_{acc}^2/(4U)$ |
| `Vt [MV]`, `Et [MV/m]` | transverse kick at `offset` (Panofsky–Wenzel) |
| `R/Q_t [Ohm]` | $V_t^2/(\omega U)$ |
| `k_kick [V/pC/m]` | kick factor, $k V_t^2/(4U)$ with $k = \omega/(\beta c)$ |
| `Rs [Ohm]` | surface resistance used for the walls |
| `Active Length [mm]`, `N Cells` | the normalisation used |
| `Q_wall []`, `Q_diel []`, `Pdiel [W]` | wall and material Q, material loss (with a lossy material) |
| `U_frac_<material> []`, `Epk_<material> [MV/m]` | a material's share of the electric energy and its peak field (with several materials) |

`get_cell_coupling(first, last)` returns $k_{cc} = 2 (f_{last} - f_{first}) / (f_{last} + f_{first})$
in %.

## Units and conventions

| Quantity | Unit |
|---|---|
| lengths (geometry, `maxh`) | m |
| `fmin`, `fmax` (inputs) | GHz |
| `frequencies` (outputs), `get_resonant_frequencies()`, `get_external_q()`, `get_rq()` | Hz |
| `get_rq()`: `RQ`, `V`, `U` | ohm, V, J (the mode scaled to U = 1 J) |
| beam: current, $z_b$, $z_{oc}$, $Z_\parallel$, $k$, $h$ | 1 A; ohm; $\sqrt{\Omega}$ |
| `get_figures_of_merit()` | the unit in each key |
| `chain_eigenfrequencies()`, `RWGAnalytical` / `CWGAnalytical` inputs and outputs | GHz |
| Z | ohm |
| time convention | $e^{+j\omega t}$ |
| port mode normalisation | $\int |E_t|^2\,dS = 1$ |
