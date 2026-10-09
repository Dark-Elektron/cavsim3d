# Changelog

Notable changes to cavsim3d. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No version has been released yet. The entries compare with the code published on
2026-08-21.

### Added

- **Projects of several parts.** `proj.import_geometry(path, name=...)`,
  `proj.create_primitive(kind, name=...)`, `proj.import_project(path, name=...)` and
  `proj.add(name, part)` add a part; the same name replaces it, and `n=N` repeats it. The
  parts are chained in list order along `proj.main_axis` (Z unless set), and `proj.parts`
  lists them.
- **Glued and coupled parts.** Plain geometry is meshed as one conformal mesh. An imported
  or repeated part is solved once and coupled to its neighbours through the modes of the
  ports that face each other. `asm.set_mesh_strategy('glued' | 'coupled')` overrides the
  choice.
- **Reuse of solved projects.** `proj.import_project(..., mode='reference')` (the default)
  reads another project's results in place, `mode='copy'` copies them in, and
  `proj.localize()` turns references into copies. The source project is never written:
  whatever an imported part lacks is computed into the importing project, and `solve()`
  first prints a plan (reuse, reduce or recompute).
- A chain's parts are reused by the next `solve()` with the same settings and geometry,
  also after reopening (`fds/sections.json`). `proj.fds.foms.reduce()` can be called again
  with another `tol`, and reopening restores `proj.fds.foms`, `.roms` and `.roms.concat`
  with its last sweep.
- An interrupted full-order sweep resumes from the finished samples in `fds/checkpoint/`.
- `solve(rerun=None)`, the default: the stored results are returned for the same request,
  and recomputed when the request changed, with a list of what changed.
- First-kind Nédélec elements, `solve(nedelec='first')`, the default. Projects saved
  without the setting keep `'second'`.
- `solve(impedance_reference='line' | 'wave')` for TEM ports. The TE and TM modes of a
  coaxial port are referred to their own wave impedance.
- **Bodies of revolution**: the cavsim2d models `EllipticalCavity`,
  `EllipticalCavityFlatTop`, `RFGun`, `Pillbox`, `SplineCavity`, `Beampipe`, `BLA`,
  `Bellows` and `Taper` (`cavsim3d.geometry.axisymmetric`), with cavsim2d's names and
  arguments, revolved about Z. Also available as primitives, for example
  `proj.create_primitive('elliptical_cavity', ...)`.
- `set_materials()` on every geometry, not only on CAD imports.
- Eigenfrequencies of the full-order model: `proj.fds.get_resonant_frequencies()`.
- Figures of merit of an eigenmode: `get_figures_of_merit(i)` (R/Q, Eacc, peak surface
  fields, wall Q, G, Rsh, transverse kick, Q_diel), `get_rq(i)` and
  `get_cell_coupling(i_0, i_pi)`.
- Loaded and external Q of the resonances: `rom.get_external_q(fmin, fmax)`.
- `concat.port_map()` and `concat.print_port_map()`: which part (or copy) and which of
  its own ports each external port of a joined model is.
- `cavsim3d.analysis`: `network_matrix`, `port_mode_labels`, `keep_port_modes`,
  `plot_matrix`, `plot_entries` and `band_difference`, to compare S or Z with a reference.
- `from cavsim3d import EMProject`.
- `RectangularWaveguide` accepts `length=` and `CircularWaveguide` accepts `L=`, so the
  two take the same keywords.
- `Assembly.add(..., attach_port=...)` names the port of the neighbouring part to join.
- `CITATION.cff`, issue and pull-request templates, a release workflow and pre-commit
  hooks.
- **Beam excitation.** `proj.add_beam(name, x=, y=)` adds a beam (a line current of 1 A
  at the speed of light along the main axis) and `proj.add_beam_path(name, ...)` a voltage
  path without current; both are saved in `project.json`. A full-order solve then also
  returns the generalised matrices `fom.s_tilde = [[S, k], [h, z_b]]` and
  `fom.z_tilde = [[Z, k_Z], [h_Z, z_oc]]` (labels `b(1)`, ...),
  `fom.beam_impedance()` (Z_par = -z_b), `fom.beam_field(i)` and the plots
  `plot_s_tilde()`, `plot_z_tilde()`, `plot_beam_impedance()`. The scattered field is
  solved with the same factorisation as the port modes; the beam line need not be part of
  the mesh; materials off the beam line add a contrast load. Single parts and glued
  parts (per part or in one piece); `foms.concatenate()` joins the parts' S~ at the faces
  between them. A beam added to a solved project solves only the beam columns. Without
  a beam the port results are unchanged. Files: `z_tilde/`, `s_tilde/`, `snapshots_beam/`,
  `matrices/beam_<part>.h5`, `port_modes/beam_port_fields.pkl`. `beta = 1` and a beam in
  vacuum only.
- **Beams through repeated or imported parts.** Each unique part of a chain is solved
  once with the beam where it runs through it, in the part's own frame (the beam is given
  in the first part's frame; each next part sits with its joined face centred on the face
  it joins). `proj.fds.foms.concatenate()` joins the copies through their S~, each with
  the beam's phase at its position. A part imported from a project solved without a beam
  gets its beam columns computed in the importing project from its stored port
  solutions; that project is never written.
- **Reduced models with the beam.** `fom.reduce(tol)` and `foms.reduce(tol)` reduce the
  beam column with the port columns when the full-order sweep kept its field snapshots:
  `rom.s_tilde`, `rom.beam_impedance()` at any frequency of the band, without the mesh.
  The PEC walls carry the beam's data as a lift; the beam's load and outputs, whose phase
  runs along the structure, are interpolated in frequency at Chebyshev points of the band
  widened by 10 % on each side (docs/theory/beam_reduction.md). `roms.concatenate()`
  joins reduced parts with the beam (repeated, imported or glued) at the frequencies of
  every `concat.solve()`.
- With a beam defined, `generate_mesh()` curves the mesh to order 4 unless `curve_order`
  is given. The solve warns when a beam runs past curved walls meshed to a lower order.
- **Transverse beam impedance.** `proj.add_transverse_beams(d)` adds two beams per
  transverse plane at +-d, and `transverse_impedance(plane)` on every result with a beam
  gives Z_perp in Ohm/m from the double difference of their impedances (Panofsky-Wenzel):
  the dipole part, without the monopole and quadrupole parts or a coupler's kick.
- `get_hom_power(current)` on every result with a beam: the power the beam leaves in each
  port mode and port, from the amplitudes of the beam current's spectral lines at the
  result's frequencies. `cavsim3d.analysis.bunch_train_spectrum()` gives the lines of a
  train of Gaussian bunches. Joined models now keep the reference impedances of their
  port modes with S~.
- The transverse kick per plane: `get_figures_of_merit()` adds `Vt_x`, `Vt_y`,
  `R/Q_t_x`, `R/Q_t_y`, `k_kick_x` and `k_kick_y` (the planes across the beam axis), and
  `get_rq()` returns the voltage with its phase as `V_complex`, so the multipole parts of
  a mixed mode can be separated from lines at opposite offsets.
- `rom.solve(frequencies=...)` and `concat.solve(frequencies=...)` take any array of
  frequencies in GHz in place of `fmin`, `fmax` and `nsamples`, for example points on the
  narrow resonances that `get_external_q()` finds.
- `rom.solve(store_snapshots=False)` and `concat.solve(store_snapshots=False)` keep and
  save S, Z and the beam blocks only, not the reduced solution of every frequency (709 MB
  for a module of two reduced cavities at 6001 frequencies). A field at one frequency is
  then solved again when asked for, and `concat.reduce()` solves the states it needs.

### Changed

- `solve(solver_type='auto')` is the default: a direct factorisation when it fits in 60 %
  of the free memory (estimated from the number of unknowns and matrix entries), the
  iterative solver otherwise. The default used to be `'iterative'`.
- The iterative solver is COCG (conjugate gradients for complex-symmetric systems) with a
  BDDC preconditioner; a solve that stalls is finished by GMRES.
  `iterative_opts={'method': 'gmres'}` uses GMRES throughout. `tol` (default `1e-8`) is
  relative to the right-hand side; it used to be an absolute `1e-6`.
- `rom.solve()`, `roms.solve()` and `concat.solve()` save the sweep's own results (S, Z,
  snapshots, the beam's S~) and `timing.json`, not the whole project: the full-order
  results and the reduced or coupled matrices are left as they are. A 13-frequency
  reduced sweep of the TESLA cavity took 1.1 s, almost all of it rewriting unchanged
  files; it now takes 0.1 s. The joined model of glued parts reduced per domain
  (`proj.fds.foms.roms.concat`) now saves its sweep too.

- `solve()` raises `TypeError` for an option it does not know, and suggests the closest
  one (`n_port_modes=2` → "did you mean 'nportmodes'?"). Unknown options used to be
  ignored. `fmin`, `fmax` and `nsamples` are checked: finite, `0 < fmin <= fmax`, and
  `nsamples` a whole number of at least 1.
- Material properties are checked: real and finite, `eps_r` and `mu_r` above 0, `sigma`
  and `tan_delta` at least 0. A complex or text value used to become 1.
  `set_materials()` of a CAD import rejects unknown keys.
- A boundary is a port when its name starts with `port`, in any case. Any name containing
  "port" (`support_ring`) used to count; such a name now triggers a warning. Port names
  that differ only in case (`Port3`, `port3`) raise `ValueError`.
- Every named material of a mesh is a domain. When a material name contained "cell", the
  other materials used to be dropped.
- `export_touchstone()` renormalises S to 50 Ω by default (`z0=50.0`), so the file's
  `R 50` line is exact. `z0=None` writes S referenced to each port's own impedance, as
  before, and warns. The file name may be a `pathlib.Path`.
- `EMProject(name, overwrite=True)` deletes the folder only if it is a cavsim3d project;
  another folder, or a file, raises `FileExistsError`. A project name must be a single
  folder name: `""`, `"."`, `".."`, a path or an absolute path raise `ValueError`.
- `EMProject.load()` raises `FileNotFoundError` for a missing project, and `ValueError`
  with `overwrite=True`.
- Log output: the `cavsim3d` logger writes to the current `sys.stdout` and no longer
  propagates to the root logger. Colour is used only in a terminal or in Jupyter, and
  `NO_COLOR` and `FORCE_COLOR` are honoured. Importing the package no longer reconfigures
  `sys.stdout`.
- `get_resonant_frequencies()` of a reduced or joined model lists the resonances within
  10 % of its training band's edges; far from the band the projection has spurious
  eigenvalues. The mode indices of `get_eigenmode()`, `get_rq()`,
  `get_figures_of_merit()` and `get_external_q()` count the same list, and so do, on a
  joined model, `chain_eigenfrequencies()`, `chain_axis_profile()`,
  `reconstruct_chain_eigenmode()`, `reconstruct_eigenmode()` and `plot_eigenmode()`.
  These used to count every eigenvalue of the coupled matrix, the spurious ones below
  the band included, so the same index named another mode there. `fmin=` and `fmax=`
  list others (`fmin=0`: all).
- `NetlistSection.project` is always `None`: the scratch project a section is solved in
  is deleted once its results are copied into the project. `section.fom` reads them from
  there.
- The project folder: every stage folder holds only `matrices/`, `eigenmodes/`, `s/`,
  `z/`, `snapshots/` and the next stage's folder; parts are told apart by file name; each
  project has one `mesh/` and one `geometry/` folder.
- Port modes are identical on both faces of a join, and modes with equal cutoffs (TE20
  and TE01 when a = 2b) are numbered alike on every port.
- The notebook banner (logo, version and project name) is smaller, comes before the
  "Creating new project" line, and appears only when a project is created, not when one
  is reopened.
- The documentation notebooks leave their 3D views commented out, which keeps the
  notebooks and the site small. Uncomment them to see the geometry when running a
  notebook.
- Dependencies: `gmsh`, `tqdm`, `termcolor` and the `full` extra were removed; `dev` and
  `docs` extras were added. `requirements.txt` installs `-e .[dev]`. `threadpoolctl` is
  now a dependency.
- The POD factors the snapshot matrix as X = QR in place and passes only R to the SVD:
  about one copy of the snapshots in memory instead of several, which matters for a large
  model with a beam.
- The beam blocks of a reduced model and the joins of reduced sections run on one BLAS
  thread, and the join is computed for many frequencies at once. Measured on a module of
  eight reduced cavities on 16 cores, the join ran 10 times slower on every thread than
  on four, and about 100 times slower with three such processes at once.

### Deprecated

- `proj.n_port_modes`: the solver never read it. Pass `nportmodes` to `solve()`.
  Setting it warns.
- `proj.create_importer()`: use `proj.import_geometry()`.
- `proj.fds.import_model()`: use `proj.import_project()`.

### Removed

- The Dockerfile, which no workflow used.

### Fixed

- Locating points on a line through a mesh (the beam's voltage path, the field on axis
  of the figures of merit, `chain_axis_profile()`, the voltage path of a quasi-TEM port)
  could crash the process on a curved mesh: NGSolve was asked for points just outside
  it. The stretches of the line inside the mesh now come from its crossings with the
  boundary, moved onto the curved faces, and only points inside are located.
- A square port face was taken for a coaxial one; it is detected as rectangular.
- `overwrite=True` could delete the wrong folder: `""` or `"."` deleted the base folder,
  and `".."` its parent.
- A project could not be reopened after `invalidate_results()` or `invalidate_mesh()`.
- Reducing a chain of repeated or imported parts a second time raised `KeyError`.
- A chain, and its joined model's sweep, was not restored on reopening, and every
  `solve()` recomputed its parts.
- Each part of a chain left a scratch folder in the system temp folder.
- `Assembly.connect()` ignored the target part, the ports and the gap, and a connected
  part appeared twice after reopening.
- `from cavsim3d.rom import ...` failed with a circular import in a fresh interpreter.
- Material and boundary names were read as regular expressions: a body named `Body(1)`
  or `window+flange` selected no mesh elements.
- Loading an imported project could prompt to delete the source project's results when
  the source's CAD file had changed.
- The number of propagating modes of a circular port was 0 in the TE11-only band, so the
  warning for a join that carries too few modes could not fire there.
- Numeric port modes failed on a coarse port face (`LinAlgError`, or NaN at random), or
  reported a spurious TEM mode. Port faces of up to 600 free unknowns are now solved
  exactly, and a face that resolves fewer modes than requested raises a `ValueError`
  that says what to change.
- `nsamples=4.0` crashed, and NaN or infinite frequencies were accepted.
- `reduce(max_rank=0)` crashed.
- `export_touchstone()` crashed with a `pathlib.Path`.
- `band_difference()` with default arguments returned `{}`.
- On Windows, pythonocc-core 8 (OpenCASCADE 8), the version conda-forge installs since
  2026-09-23, cannot run next to the OpenCASCADE 7.8 that NGSolve's netgen loads:
  importing ngsolve failed ("WinError 127"). The install instructions and CI pin
  `pythonocc-core=7.9`, and on Windows a mismatched pair now raises an `ImportError` that
  says which version to install.
- On Linux, pythonocc modules imported after netgen ran on netgen's OpenCASCADE instead
  of their own. Reading a STEP file then failed (`KeyError: 'OpenCASCADE Error
  [Standard_NoSuchObject] …'`, or "Wrong number or type of arguments") when the solver
  modules had been imported first, and crashed with pythonocc-core 8. netgen's
  OpenCASCADE is now loaded privately, so the two copies stay apart whatever the import
  order and version.
- `get_external_q()` over a band that crosses a port mode's cutoff listed some loaded
  resonances several times and left others out, and its `mode_index` could name one
  closed-problem mode for several resonances: the loaded problem was first solved with
  the port impedances of the band's middle. Each loaded resonance is now solved with the
  impedances at its own frequency and belongs to a different closed-problem mode, and
  the result no longer depends on the band. `refine` is now the most re-solves
  (default 10); they stop once the frequency settles.
- A reduced model reopened from disk failed in `get_resonant_frequencies()` and its
  other eigen methods, and a reduced model swept outside its band recorded that sweep as
  its training band, which the checks on joined models then used.
- `compute_error()` failed when the model and the reference had frequency grids of
  different lengths, such as a ROM swept more finely than its FOM.
- The example scripts `concatenation_example.py`, `plot_simulation_results.py` and
  `rwg/rwg_analysis.py`, and five example notebooks, failed: variables had been renamed
  to `cavsim3d.rom.…` and `cavsim3d.analytical.…`, and some calls no longer exist. The
  analytical references were also given Hz where they take GHz. `tesla.ipynb` ran only
  where its project had been saved before, and `multi_rwg.ipynb` ended in an unfinished
  line.
- A reopened project solved with port mode counts given per port
  (`nportmodes={'port1': 5, 'port2': 3}`) discarded its sweep and solved it again at every
  `solve()`: the request was compared with the largest count instead of the request
  saved. When a changed request does recompute stored results, the console now lists
  what changed.
- `get_external_q()` on a joined model with TE/TM ports (whose Z0 depends on frequency)
  computed every eigenpair of the loaded problem, of size 2r, for each group of closed
  modes and up to ten times more while refining: on four joined copies of a cavity
  (r = 1093) it had not finished after an hour. Each solve now computes the eigenpairs
  near the group's modes by shift-invert, with one factorisation of size r per shift,
  and every eigenpair only where those do not hold the group's modes (a strongly damped
  mode); a long call reports its progress. On 32 joined copies of an iris cavity
  (r = 993, 256 resonances) the call takes 50 s instead of 284 s, with the same results.
- A project whose mesh could not be curved to the order asked for recorded the order
  asked, so every reopening tried it again, failed and warned again. The order reached
  is recorded too, and a reopened mesh is curved to it directly. The warning names the
  sliver edges (micrometres long) where OCC's projection fails, and with a beam the
  solve no longer advises an order that has already failed.
- A reduction whose SVD failed (LAPACK could not allocate its workspace, and returned NaN
  singular values) gave a reduced model of 0 DOFs, reported it as complete, and saved it
  over the good one; the joins built from it then failed with "Null space empty". The
  reduction now raises `FloatingPointError` and the stored reduced model stays as it was,
  in memory and on disk. A join of a section with 0 DOFs names that section.
- A project saved with empty port modes crashed at the next solve.
- Confirmation prompts failed in scripted runs (nbconvert, papermill); with nobody to
  answer, they count as "no".
- NGSolve printed "used dof inconsistency" at every port-mode solve.
- The Jupyter banner's logo was missing after a regular (non-editable) install.
- The README's license badge said LGPL; the license is MIT.

[Unreleased]: https://github.com/Dark-Elektron/cavsim3d/commits/master
