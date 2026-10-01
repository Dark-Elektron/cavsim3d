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

### Changed

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
  `docs` extras were added. `requirements.txt` installs `-e .[dev]`.

### Deprecated

- `proj.n_port_modes`: the solver never read it. Pass `nportmodes` to `solve()`.
  Setting it warns.
- `proj.create_importer()`: use `proj.import_geometry()`.
- `proj.fds.import_model()`: use `proj.import_project()`.

### Removed

- The Dockerfile, which no workflow used.

### Fixed

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
- A project saved with empty port modes crashed at the next solve.
- Confirmation prompts failed in scripted runs (nbconvert, papermill); with nobody to
  answer, they count as "no".
- NGSolve printed "used dof inconsistency" at every port-mode solve.
- The Jupyter banner's logo was missing after a regular (non-editable) install.
- The README's license badge said LGPL; the license is MIT.

[Unreleased]: https://github.com/Dark-Elektron/cavsim3d/commits/master
