# cavsim3d — working rules

## What an fds does (the mental model — do not violate)
An `fds` (frequency-domain solver) belongs to **exactly one project, which has exactly one `fds`**. Never nest projects/solvers (no `fds/foms/<name>/fds/fom/…`). What it does, in order:
1. Takes a **geometry/mesh**, builds a **finite element space**, and assembles the **system matrices** (K, M, B) → saved under `fds/foms/matrices/` (per domain: `K_<domain>.h5`, …).
2. **Solves** → produces **eigenmodes, S, Z, snapshots** for **every unique domain** of a multi-solid model → `fds/foms/{eigenmodes,s,z,snapshots}/` (per-domain-suffixed files).
3. **Reduce** (`foms.reduce`) → a **ROM**: reduced matrices/vectors under `fds/foms/roms/matrices/`; solving the ROM regenerates `{eigenmodes,s,z,snapshots}/` under `fds/foms/roms/`.
4. **Concatenate** (`roms.concatenate`) → the **concat**: a *single unified* model (its own matrices) under `fds/foms/roms/concat/`; solving it produces `{eigenmodes,s,z,snapshots}/`. Only at this stage is the object self-contained — it needs nothing from other folders.

**Importing another project just replaces step 1–2 (or 3) computation with fetching pre-computed artifacts.** There must be **NO distinction on disk between a computed section and an imported one**: importing copies files folder-to-matching-folder (`already_solved/fds/foms/matrices/*` → `current/fds/foms/matrices/*`, same for s/z/eigenmodes/snapshots/roms, mesh, geometry), **renamed to the section's index in the current project** (`K_global.h5` → `K_legacy.h5`).

## Folder contents (canonical — do not violate)
- `fom`/`foms` may contain ONLY: `matrices/`, `eigenmodes/`, `s/`, `z/`, `snapshots/` + the nested `rom/`/`roms/`.
- `rom`/`roms` contain the same five + optionally `concat/` (in `roms`) or a further `rom/`/`roms/` (further reduction).
- `concat` contains the same five + optionally `rom/`.
- **Never** create per-section subfolders inside `fom(s)`/`rom(s)`/`concat` — sections are distinguished ONLY by the filename suffix convention (`K_<section>.h5`, `s_<section>.h5`, …).
- **One `mesh/` and one `geometry/` folder per project**, at the top level next to `fds/`. Per-section meshes are suffix-renamed files inside it (`mesh_<section>.pkl`, `fes_<section>.pkl`); per-section geometry goes to `geometry/components/<section>.step`. (Netlist sections have independent meshes — they couple through modal ports; a fully-coupled multi-solid keeps its single glued conformal mesh files.)

## Operation philosophy (do not violate)
- `proj.fds` is the engine object (a future time-domain solver would be `proj.tds`). The pipeline is staged and user-controlled, with each stage a real, persisted object:
  **FOM → ROM → Concatenation → (optional) ROM again**
- Fluent access mirrors stage cardinality: `proj.fds.fom.rom` (single solid), `proj.fds.foms.roms.concat` (multi-solid), `concat.rom` (further reduction). `foms.concatenate()` (FOM-level concat) is allowed but must warn.
- **`concat` is not a geometry operation** — geometry composition is the Assembly's job. `concat` is the object *returned by* `concatenate()` on a `foms`/`roms` collection. The user **never imports or constructs `ConcatenatedSystem`** (`from_flat_roms` is internal plumbing used by `roms.concatenate()`).
- `Assembly` is a **passive netlist**: components (geometry | imported model | sub-assembly), connections, and repeat counts (`asm.add(name, comp, n=N)`, default 1). It never drives computation. `proj.create_assembly(main_axis=...)` remains the entry point.
- **Importing an existing project**: `imported = proj.fds.import_model(path)` (named so because `import` is a Python keyword) returns a fail-fast handle that `asm.add()` accepts like a geometry. Netlist assemblies run through the SAME pipeline: `proj.fds.solve()` → `proj.fds.foms.reduce(tol)` → `.concatenate()`. An imported section occupies a normal per-domain slot in the single `fds/foms/…` tree — its pre-computed artifacts are **copied into the slot (renamed to its position)**, never recomputed and never nested as a sub-project.
- There is **no separate chain/driver class** (the old `Chain` was removed). Never reintroduce parallel pipelines; extend the existing stage objects.
- FOM/ROM/concat artifacts are **portable**: saved per project (`fds/fom/rom/`, `fds/foms/roms/`), reloadable without the original solver (load-if-exists / run-if-not).
- Compatibility at joins is a **checked condition**, not ownership: port-mode counts + per-mode fingerprints (type, indices, cutoff kc, polarization) must match; ROM training bands must overlap (disjoint → error, extrapolation → warning).

## Living tutorial (mandatory)
`tutorials/core_workflows.py` is the always-current reference for how the core pieces connect. **Whenever core functionality changes or a new core feature is added, update it in the same change.** Helper functions (plotting utilities etc.) do not require tutorial updates. Keep it runnable end-to-end with small rectangular-waveguide examples.

## Practicalities
- Python: `C:\Users\Soske\anaconda3\envs\cavsim3d\python.exe` (conda env `cavsim3d`). The IDE type-checker may point at a different interpreter — its "cannot find module ngsolve/numpy" diagnostics are false positives.
- Run tests with that interpreter: `python -m pytest tests/ -q`.
- Import order matters in tests/scripts: import `cavsim3d.core.em_project` (or `cavsim3d.solvers.frequency_domain`) before `cavsim3d.rom.*` to avoid a circular import.
- **Interactive test bench** (ngapp): `python -m testbench` (`--dev` for hot reload) after a one-time `pip install -e testbench/`. Template layout under `testbench/` (`src/testbench/{app,appconfig,__main__}.py`, `pyproject.toml`). **Open the FULL URL the server prints** (it carries `websocketPort` + `wsToken`; plain `localhost:8765` stays blank). Writes inspectable projects to `testbench_runs/` (gitignored). Helper app — no living-tutorial updates needed, but keep it working when pipeline APIs change. Browser-verify UI changes with Playwright (installed; headed mode required — headless Chromium has no WebGPU).
