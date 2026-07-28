#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cavsim3d core workflows — LIVING TUTORIAL.

=============================================================================
THIS FILE IS THE ALWAYS-CURRENT REFERENCE FOR HOW THE CORE PIECES CONNECT.
It MUST be updated whenever core functionality changes or a new core feature
is added (solver stages, ROM, concatenation, assembly/netlist, import/reuse).
Helper functions (plotting utilities etc.) do not require updates here.
Last updated: 2026-07-09 (canonical folder layout: fom(s)/rom(s)/concat hold
ONLY matrices/eigenmodes/s/z/snapshots + nested stage folders; ONE mesh/ and
geometry/ per project; imported sections copied + renamed, indistinguishable
from computed ones; quasi-TEM/microstrip inhomogeneous ports — section 4b).
=============================================================================

Operation philosophy
--------------------
``proj.fds`` is the engine (a future time-domain solver would be ``proj.tds``).
Everything is a staged, user-controlled pipeline; each stage is a real,
inspectable, persisted object:

    FOM  ->  ROM  ->  Concatenation   (and optionally: -> ROM again)

``concat`` is NOT a geometry operation — geometry composition is the
:class:`Assembly`'s job.  ``concat`` is the OBJECT RETURNED by calling
``concatenate()`` on a ``foms``/``roms`` collection:

    proj.fds.fom.rom                      # single-solid model
    proj.fds.foms.roms.concat             # multi-solid model (per-solid FOMs)
    proj.fds.foms.roms.concat.rom         # further reduction of the coupled system
    proj.fds.foms.concatenate()           # FOM-level concat: allowed, but WARNS

The :class:`Assembly` is a PASSIVE NETLIST — it never computes.  It holds the
components, how they connect, and repeat counts (``asm.add(name, comp, n=8)``,
default n=1).  Components may be geometry objects, sub-assemblies, or models
IMPORTED from already-run projects:

    imported = proj.fds.import_model("path/to/earlier_project")
    asm.add("hom", imported, after="cavity")

(the method is ``import_model`` because ``import`` is a reserved Python
keyword).  Saved FOM/ROM/concat artifacts are PORTABLE ("IKEA screws"): each
lives in its project folder and is loaded — never recomputed — wherever it is
referenced.  Compatibility at a joint is a CHECKED CONDITION, not ownership:
  * port-mode COUNTS must match at connected interfaces (error),
  * per-mode FINGERPRINTS must correspond — type, modal indices, cutoff kc
    (i.e. cross-section dimensions), polarization (error; polarization matters
    for degenerate and numerically-computed modes),
  * ROM TRAINING BANDS must overlap — disjoint bands error; sweeping outside
    the shared band warns (extrapolation beyond snapshot coverage).

Run:  python tutorials/core_workflows.py   (fast; small rectangular waveguides)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_tutorial_"))
A, B_, L, MAXH = 0.1, 0.05, 0.06667, 0.06
FOM_CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)


def banner(msg):
    print("\n" + "=" * 74 + f"\n {msg}\n" + "=" * 74)


# =========================================================================== #
# 1. SINGLE-SOLID MODEL:   proj.fds.solve() -> fds.fom.reduce() -> rom        #
# =========================================================================== #
banner("1. Single solid:  fds.solve() -> fds.fom.reduce() -> rom.solve()")

proj = EMProject(name="single_rwg", base_dir=str(WORK), overwrite=True)
proj.geometry = RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH)

proj.fds.solve(config=FOM_CFG)              # STAGE 1: full-order model
fom = proj.fds.fom                          # the FOM artifact (persisted)

rom = fom.reduce(tol=1e-9)                  # STAGE 2: reduced-order model
#   -> auto-saved to <project>/fds/fom/rom/, so it can be IMPORTED into any
#      later analysis without recomputing (see section 4).

res = rom.solve(fmin=1.8, fmax=2.4, nsamples=200)    # cheap fine sweep
print(f"   ROM sweep: {res['Z'].shape[0]} frequency points, "
      f"reduced size {rom.reduced_dimensions}")


# =========================================================================== #
# 2. MULTI-SOLID MODEL (one glued mesh):  foms -> roms -> concat [-> rom]     #
# =========================================================================== #
banner("2. Multi-solid: fds.foms.reduce() -> roms.concatenate() [-> .reduce()]")

proj2 = EMProject(name="multi_solid", base_dir=str(WORK), overwrite=True)
asm_geo = proj2.create_assembly(main_axis="Z")       # assembly AS geometry
asm_geo.add("h1", RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH))
asm_geo.add("h2", RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH), after="h1")
asm_geo.generate_mesh(maxh=MAXH)                     # ONE glued mesh

proj2.fds.solve(config=dict(**FOM_CFG, per_domain=True,
                            store_snapshots=True, global_method=None))

roms2 = proj2.fds.foms.reduce(tol=1e-9)     # per-solid ROMs
concat2 = roms2.concatenate()               # STAGE 3: coupled at the junction
res2 = concat2.solve(fmin=1.8, fmax=2.4, nsamples=100)
print(f"   per-solid ROMs coupled: Z shape {res2['Z'].shape}")
# NOTE: proj2.fds.foms.concatenate() (FOM-level, skipping the ROM stage) also
# exists for validation, but it WARNS: it builds dense full-order matrices.
# The logical path is always FOM -> ROM -> Concatenation.


# =========================================================================== #
# 3. REPEAT-N NETLIST — same pipeline, components computed ONCE               #
# =========================================================================== #
banner("3. Netlist repeat-N:  asm.add(geo, n=3) -> the SAME fds pipeline")

proj3 = EMProject(name="chain_module", base_dir=str(WORK), overwrite=True)
asm3 = proj3.create_assembly(main_axis="Z")          # passive netlist
asm3.add("cell", RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH), n=3)
#        ^ 3 consecutive copies, computed ONCE (default n=1).
#          Consecutive instances couple port2 -> port1.

proj3.fds.solve(config=FOM_CFG)             # STAGE 1: FOM per UNIQUE section
#   ONE fds, laid out exactly like a multi-solid project (a section == a
#   domain, distinguished ONLY by the filename suffix).  fom(s)/rom(s)/concat
#   hold ONLY matrices/eigenmodes/s/z/snapshots (+ the nested stage folder):
#      <project>/fds/foms/matrices/K_<section>.h5, M_<section>.h5, B_<section>.h5
#      <project>/fds/foms/{s,z,eigenmodes,snapshots}/<name>_<section>.h5
#      <project>/fds/foms/roms/matrices/A_r_<section>.h5 ...   (ROM stage)
#      <project>/fds/foms/roms/concat/                          (concat stage)
#   ONE mesh/ and ONE geometry/ folder per project, at the top level:
#      <project>/mesh/mesh_<section>.pkl, fes_<section>.pkl
#      <project>/geometry/components/<section>.step
#   No per-section folders, and NEVER a nested sub-project (one fds/project).

roms3 = proj3.fds.foms.reduce(tol=1e-9)     # STAGE 2: ROM per unique component
concat3 = roms3.concatenate()               # STAGE 3: netlist expanded + coupled
print(f"   {len(concat3.structures)} coupled instances, "
      f"{concat3.n_external_ports} external ports")

res3 = concat3.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=200))
print(f"   |S21| at mid-band ~ {abs(res3['S'][100, 1, 0]):.3f} (matched guide -> ~1)")

rom_of_concat = concat3.reduce(tol=1e-10)   # STAGE 4 (optional): concat.rom
print(f"   further-reduced coupled system: {type(rom_of_concat).__name__}")


# =========================================================================== #
# 4. IMPORT AN ALREADY-RUN PROJECT and mix it with new geometry               #
# =========================================================================== #
banner("4. Cross-project reuse:  proj.fds.import_model(path) -> asm.add(...)")

proj4 = EMProject(name="mixed_module", base_dir=str(WORK), overwrite=True)
asm4 = proj4.create_assembly(main_axis="Z")

# 'single_rwg' from section 1 is a campaign you ran earlier — import it:
legacy = proj4.fds.import_model(str(WORK / "single_rwg"))
legacy2 = proj4.fds.import_model(str(WORK / "single_rwg"))
print(f"   {legacy}")                        # fail-fast handle: ports, band

asm4.add("fresh", RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH))
asm4.add("legacy", legacy, after="fresh")    # imported model in the netlist
asm4.add("legacy2", legacy2, after="legacy")    # imported model in the netlist

proj4.fds.solve(config=FOM_CFG)              # runs 'fresh'; 'legacy' is loaded
concat4 = proj4.fds.foms.reduce(tol=1e-9).concatenate()
res4 = concat4.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=100))
print(f"   mixed netlist coupled: {len(concat4.structures)} sections, "
      f"|S21| ~ {abs(res4['S'][50, 1, 0]):.3f}")
# Importing copies folder-to-matching-folder, renamed to the section's index —
# on disk there is NO distinction between 'legacy' (imported) and 'fresh'
# (computed):  K_legacy.h5 next to K_fresh.h5, mesh/mesh_legacy.pkl next to
# mesh/mesh_fresh.pkl, geometry/components/legacy.step next to fresh.step.
# The module is SELF-CONTAINED — it keeps working even if the source project
# is later moved or deleted.  Nothing is ever recomputed for an import.
#   * A missing source at solve time -> FileNotFoundError (nothing to copy).


# =========================================================================== #
# 4b. QUASI-TEM PORTS (inhomogeneous / microstrip cross-sections)             #
# =========================================================================== #
banner("4b. Quasi-TEM ports:  inhomogeneous microstrip cross-section")

# A microstrip port cross-section is inhomogeneous (dielectric substrate + air)
# and quasi-TEM — no analytic mode.  cavsim3d solves it with a mixed
# HCurl(Et) x H1(Ez) eigenproblem that yields the propagation constant beta
# directly, orders modes like CST (descending real(beta): fundamental first),
# and renormalises S to the power-voltage line impedance Z_PV.
#
# The port faces of ONE physical port are split by material and share a
# 'port<N>' prefix (e.g. 'port1_substrate' + 'port1_air'); they auto-group into
# the logical port 'port1'.  Such inhomogeneous ports auto-enable qTEM; the
# PEC conductor outlines on the port plane (e.g. 'microstrip_edges|ground_edges')
# drive the port solver's dirichlet_bbnd (declared by the geometry, or via the
# 'qtem_conductor_bbnd' solve option).
from cavsim3d.geometry import MicrostripLine   # noqa: E402

ms_proj = EMProject(name="microstrip", base_dir=str(WORK), overwrite=True)
ms_proj.geometry = MicrostripLine(maxh=2.0e-3)          # coarse: fast tutorial
ms_proj.fds.solve(fmin=1.0, fmax=6.0, nsamples=4,
                  config=dict(order=2, nportmodes=1,
                              qtem_ports=['port1', 'port2'],
                              solver_type='direct', store_snapshots=False))
_ps = ms_proj.fds.port_solver
_ee = _ps.port_eps_eff['port1'][0].real
_z = _ps.port_line_impedance['port1'][0].real
print(f"   port1 fundamental quasi-TEM: eps_eff~{_ee:.2f}, Z_PV~{_z:.1f} ohm "
      f"(1 mode selected == CST ordering)")
# The same fds.fom.reduce() / foms.concatenate() pipeline applies unchanged.


# =========================================================================== #
# 5. THE JOIN GUARDS (what stops you from building nonsense)                  #
# =========================================================================== #
banner("5. Join guards: port modes, mode fingerprints, training bands")

print("""
   * Port-mode COUNT mismatch at a connected interface  -> ValueError
     ("The number of port modes must match at connected interfaces...")
   * Mode FINGERPRINT mismatch (type / indices / cutoff kc / polarization)
     -> ValueError.  Matching cross-section dimensions give matching kc;
     polarization matters for degenerate (e.g. TE11) and NUMERIC modes:
     solve both sections with the same polarization_angle convention.
   * ROM training bands: sections reduced over DISJOINT frequency bands
     cannot be coupled (ValueError).  Narrow overlap or sweeping outside the
     shared band -> UserWarning (extrapolation beyond snapshot coverage:
     results may be inaccurate or wrong).
""")

print(f"All tutorial artifacts under: {WORK}")
banner("DONE")
