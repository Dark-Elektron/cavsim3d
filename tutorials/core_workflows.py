#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cavsim3d core workflows — LIVING TUTORIAL.

=============================================================================
THIS FILE IS THE ALWAYS-CURRENT REFERENCE FOR HOW THE CORE PIECES CONNECT.
It MUST be updated whenever core functionality changes or a new core feature
is added (solver stages, ROM, concatenation, assembly/netlist, import/reuse).
Helper functions (plotting utilities etc.) do not require updates here.
Last updated: 2026-10-07 (reduced models with the beam: fom.reduce() /
foms.reduce() carry the beam, roms.concatenate() joins reduced parts with it
at any frequency -- 6e; beam excitation: proj.add_beam(), fom.s_tilde, a beam
added to a solved project, the parts' S~ joined by foms.concatenate() -- also
coupled parts, repeated or imported (6d); generate_mesh() curves to order 4
when a beam is defined -- section 6).  What changed: CHANGELOG.md.
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

A project holds a LIST OF PARTS, added one by one:

    proj.import_geometry("cell.step", name="cell", unit="mm")   # CAD file
    proj.create_primitive("rwg", name="taper", a=..., L=...)     # primitive
    proj.create_primitive("elliptical_cavity", name="cav", ...)  # cavsim2d model
    proj.import_project("path/to/earlier_project", name="hom")   # solved project
    proj.add("module", some_assembly)                            # anything else

The same name REPLACES a part (re-running a cell does not double the model);
``n=8`` repeats a part.  With one part, the project's geometry is that part;
with more, the parts are chained in list order along ``proj.main_axis`` (Z
unless set -- meshing prints the chain) by an :class:`Assembly`, which stays a
PASSIVE NETLIST: it never computes.  (The method is ``import_project``, not
``import``: that is a reserved Python keyword.)

Parts are either GLUED into one conformal mesh (plain geometry, each once) or
COUPLED through their port modes (a part is imported or repeated) -- the mesh
summary says which; ``asm.set_mesh_strategy('glued'|'coupled')`` overrides.
Coupled parts join through the ports that FACE each other along the axis.

An imported project is REFERENCED by default (its saved results are read where
they are) or COPIED (``mode='copy'``, the project then stands alone;
``proj.localize()`` converts references later).  The source is NEVER written:
``solve()`` prints a plan -- reuse / reduce / recompute -- and anything an
imported part lacks (a reduced model, a wider band, more port modes) is
computed INTO THIS project.  Compatibility at a joint is a CHECKED CONDITION,
not ownership:
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

from cavsim3d.core.em_project import EMProject   # also: from cavsim3d import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide

WORK = Path(tempfile.mkdtemp(prefix="cavsim3d_tutorial_"))
A, B_, L, MAXH = 0.1, 0.05, 0.06667, 0.06
FOM_CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)
#   A name solve() does not know raises TypeError with the closest option
#   (n_port_modes=2 -> "did you mean 'nportmodes'?"): a typo never runs with
#   the defaults.  A config written for the full-order solve may be reused for
#   rom.solve() / concat.solve(): its full-order options are ignored there.
#   nedelec='first' (the default solve setting) builds every H(curl) space from
#   first-kind Nedelec elements: the same curls as 'second' with ~1/3 fewer
#   unknowns at order 2 (the count CST uses); saved with the project, a change
#   recomputes.  With 'second' the direct solver stores and factorises the
#   symmetric system matrix as symmetric (same speed, half the memory); with
#   'first' the full matrix factorises faster.
#   solver_type='auto' (the default) factorises A(w) when the factorisation
#   fits in free memory -- one factorisation per sample serves every port mode
#   and beam -- and else solves iteratively: COCG with a BDDC preconditioner,
#   finished by GMRES if COCG stalls (iterative_opts={'method': 'gmres'} to
#   use GMRES throughout; 'tol' is relative to the right-hand side).


def banner(msg):
    print("\n" + "=" * 74 + f"\n {msg}\n" + "=" * 74)


# =========================================================================== #
# 1. SINGLE-SOLID MODEL:   proj.fds.solve() -> fds.fom.reduce() -> rom        #
# =========================================================================== #
banner("1. Single solid:  fds.solve() -> fds.fom.reduce() -> rom.solve()")

proj = EMProject(name="single_rwg", base_dir=str(WORK), overwrite=True)
proj.create_primitive("rwg", name="guide", a=A, L=L, b=B_, maxh=MAXH)
#   one part: the project's geometry IS that part (proj.parts -> {'guide': ...})

proj.fds.solve(config=FOM_CFG)              # STAGE 1: full-order model
fom = proj.fds.fom                          # the FOM artifact (persisted)

rom = fom.reduce(tol=1e-9)                  # STAGE 2: reduced-order model
#   -> auto-saved to <project>/fds/fom/rom/, so it can be IMPORTED into any
#      later analysis without recomputing (see section 4).

res = rom.solve(fmin=1.8, fmax=2.4, nsamples=200)    # cheap fine sweep
print(f"   ROM sweep: {res['Z'].shape[0]} frequency points, "
      f"reduced size {rom.reduced_dimensions}")
# Every stage reuses stored results for the SAME request and recomputes when
# the request changed (sweep, order, nportmodes, port settings, materials,
# geometry) -- solve(rerun=None), the default.  rerun=True forces a
# recompute; rerun=False keeps whatever is stored.
# An INTERRUPTED full-order sweep resumes: every finished sample is written to
# fds/checkpoint/ as it completes, and calling solve() again with the same
# request computes only the missing samples (also after reopening the project,
# and for every part of a netlist -- see section 3).

# Resonances of the FOM operator (K, M).  The port faces are natural
# (magnetic-wall) boundaries, so a guide of length L resonates at
# f = c/2 * sqrt((m/a)^2 + (n/b)^2 + (p/L)^2) with TE p >= 0 -- compare with
# RWGAnalytical(...).all_eigenfrequencies() (boundary_type='PMC', the default).
f_res = proj.fds.fom.get_resonant_frequencies(n_modes=3)
print(f"   FOM resonances [GHz]: {np.round(f_res / 1e9, 4)}")


# --------------------------------------------------------------------------- #
# 1b. PER-PORT MODE COUNTS:  port_map() then nportmodes as int | list | dict   #
# --------------------------------------------------------------------------- #
banner("1b. Per-port mode counts:  fds.port_map() -> nportmodes=[...]")

# port_map() needs only the mesh -- no solve -- so the port order and each
# port's geometry are known BEFORE choosing how many modes each one carries.
proj.fds.print_port_map()

# A TEM (coaxial) port usually needs one mode while a TE/TM waveguide port may
# need several. All three spellings are accepted:
#     nportmodes=1                     same count everywhere
#     nportmodes=[2, 1]                positional, in port_map() order
#     nportmodes={'port1': 2, 'default': 1}
# A list whose length does not match the port count, or a dict naming a port
# that does not exist, raises immediately rather than silently solving the
# wrong problem.
_names = [r["port"] for r in proj.fds.port_map()]
_spec = [2] + [1] * (len(_names) - 1)       # 2 modes on the first port only

proj_pm = EMProject(name="per_port_modes", base_dir=str(WORK), overwrite=True)
proj_pm.geometry = RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH)
proj_pm.fds.solve(config=dict(FOM_CFG, nportmodes=_spec))
_counts = {p: len(m) for p, m in proj_pm.fds.port_solver.port_modes.items()}
print(f"   modes per port: {_counts}")
print(f"   Z is square in the TOTAL port-modes: "
      f"{proj_pm.fds.fom._Z_matrix.shape[-1]} = {sum(_spec)}")


# --------------------------------------------------------------------------- #
# 1c. MATERIALS AND LOSSES:  geo.set_materials({...}) -> same pipeline         #
# --------------------------------------------------------------------------- #
banner("1c. Lossy filling:  geo.set_materials({'*': {eps_r, tan_delta, sigma}})")

# Any geometry takes material properties per mesh material ('*' = all).  The
# solver uses  eps = eps0*eps_r*(1 - j tan_delta) - j sigma/omega, i.e.
#     A(w) = K + j w C - w^2 (M - j D),   C = int sigma,  D = int eps0 eps_r tan_delta
# so a lossy model is complex; FOM, ROM and concatenation all carry C and D.
# (Port modes can be computed numerically for arbitrary cross-sections with
# solve(mode_source='numeric').)
proj_l = EMProject(name="lossy_rwg", base_dir=str(WORK), overwrite=True)
geo_l = RectangularWaveguide(a=A, L=L, b=B_, maxh=MAXH)
geo_l.set_materials({'*': {'eps_r': 1.5, 'tan_delta': 0.01}})
proj_l.geometry = geo_l
proj_l.fds.solve(config=dict(FOM_CFG, fmin=1.4, fmax=1.8))
rom_l = proj_l.fds.fom.reduce(tol=1e-9)
res_l = rom_l.solve(fmin=1.4, fmax=1.8, nsamples=50)
_S = res_l['S']
print(f"   |S21| ~ {abs(_S[25, 1, 0]):.3f}, power |S11|^2+|S21|^2 = "
      f"{abs(_S[25, 0, 0])**2 + abs(_S[25, 1, 0])**2:.3f} (< 1: the filling absorbs)")


# --------------------------------------------------------------------------- #
# 1d. BODIES OF REVOLUTION:  the cavsim2d models, revolved about Z             #
# --------------------------------------------------------------------------- #
banner("1d. Bodies of revolution:  create_primitive('elliptical_cavity', ...)")

# The cavsim2d models keep their names AND constructor arguments (dimensions in
# mm by default, as in cavsim2d -- unit='m' etc. to change it; maxh in metres):
# EllipticalCavity, EllipticalCavityFlatTop, RFGun, Pillbox, SplineCavity,
# Beampipe, BLA, Bellows, Taper (cavsim3d.geometry.axisymmetric).  Each builds
# its meridian as a Profile and revolves it about Z: every beam aperture is a
# port (port1 at low z), the rest is the PEC wall 'default' -- so the part
# enters the SAME pipeline as any other.  Arguments may come as one dict,
# config={...}.  The part is built WITHOUT a mesh: generate_mesh() makes it
# (else the first solve, with the part's own maxh).  kind = class name or
# snake case; a new shape needs only a profile() method.
TESLA = [42, 42, 12, 19, 35, 57.7, 103.353]          # A, B, a, b, Ri, L, Req [mm]
proj_ax = EMProject(name="tesla_cell", base_dir=str(WORK), overwrite=True)
cell = proj_ax.create_primitive("elliptical_cavity", name="cell",
                                config=dict(n_cells=1, mid_cell=TESLA, beampipe="both"))
proj_ax.generate_mesh(maxh=0.04)
print(f"   {type(cell).__name__}: ports {sorted(cell.ports)}, {cell.mesh.ne} elements")
res_ax = proj_ax.fds.solve(config=dict(fmin=1.2, fmax=1.4, nsamples=3, nportmodes=1,
                                       order=2, solver_type='direct'))
print(f"   |S21| at 1.2/1.3/1.4 GHz [dB]: "
      f"{np.round(20 * np.log10(abs(res_ax['S'][:, 1, 0])), 1)} "
      f"(TE11 cutoff of the 35 mm pipe is 2.51 GHz: evanescent ports)")
# Its (K, M) fundamental, proj_ax.fds.fom.get_resonant_frequencies(1), is
# 1.2873 GHz -- cavsim2d's 2D solve of the same meridian gives 1.28739 GHz.

# Figures of merit of an eigenmode: get_figures_of_merit(i) takes the mode index
# of the spectrum listed last (full-order, reduced or joined model) and returns
# cavsim2d's keys and units: R/Q = V^2/(w U) along the beam axis, Eacc over the
# cavity's active length (2 L n_cells), peak surface fields, the wall Q and
# G = Q_wall Rs (copper walls unless conductivity= / surface_resistance=),
# Rsh, the transverse kick (Panofsky-Wenzel), and with lossy materials Q_diel.
# Absolute values are for a stored energy of 1 J.  get_rq(i) gives R/Q alone,
# get_cell_coupling(i_0, i_pi) the kcc of a multi-cell passband.
f_ax = proj_ax.fds.get_resonant_frequencies(n_modes=1)
fm_ax = proj_ax.fds.get_figures_of_merit(0)
print(f"   TM010 {f_ax[0] / 1e9:.4f} GHz: R/Q = {fm_ax['R/Q [Ohm]']:.1f} Ohm, "
      f"G = {fm_ax['G [Ohm]']:.1f} Ohm, Epk/Eacc = {fm_ax['Epk/Eacc []']:.2f}, "
      f"Bpk/Eacc = {fm_ax['Bpk/Eacc [mT/MV/m]']:.2f} mT/(MV/m)")
# cavsim2d on the same meridian: 117.7 Ohm, 269.6 Ohm, 1.76, 4.09.  This mesh
# (maxh 0.04, order 2) is coarse; peak fields converge last (order 3, finer maxh).


# --------------------------------------------------------------------------- #
# 1e. LOADED RESONANCES:  rom.get_external_q() -> Q_L and Qext per port        #
# --------------------------------------------------------------------------- #
banner("1e. Loaded Q:  fds.fom.reduce(tol).get_external_q(fmin, fmax)")

# The (K, M) resonances have magnetic walls at the ports: no power leaves.
# get_external_q() terminates every port mode in its reference impedance (the
# load the S-parameters assume) and solves the reduced model's loaded
# eigenproblem: frequency, loaded Q and each port's external Q.  Pole residues
# of the closed problem are NOT used -- a feed line or a strongly coupled
# neighbour at the port changes Qext, here the guide stubs in front of the
# irises.  Any geometry is a BaseGeometry with a build(): a guide, two irises.
from cavsim3d.geometry.base import BaseGeometry
from netgen.occ import Box, Pnt, Z as OCC_Z


class IrisCavity(BaseGeometry):
    """A 100 x 50 mm guide with two irises (20 mm windows) 100 mm apart."""

    def build(self):
        a, b, lin, lc, t, w = 0.10, 0.05, 0.06, 0.10, 0.003, 0.02
        self.geo = Box(Pnt(0, 0, 0), Pnt(a, b, 2 * lin + lc + 2 * t))
        for z0 in (lin, lin + t + lc):
            self.geo -= (Box(Pnt(0, 0, z0), Pnt(a, b, z0 + t))
                         - Box(Pnt((a - w) / 2, 0, z0), Pnt((a + w) / 2, b, z0 + t)))
        for f in self.geo.faces:
            f.name = "default"
        self.geo.faces.Min(OCC_Z).name, self.geo.faces.Max(OCC_Z).name = "port1", "port2"
        self.geo.mat("vacuum")
        self.bc = "default"


proj_q = EMProject(name="iris_cavity", base_dir=str(WORK), overwrite=True)
proj_q.geometry = IrisCavity()
proj_q.generate_mesh(maxh=0.02)
proj_q.fds.solve(config=dict(fmin=1.9, fmax=2.3, nsamples=7, nportmodes=1, order=1))
rom_q = proj_q.fds.fom.reduce(tol=1e-9)
q = rom_q.get_external_q(fmin=1.9, fmax=2.3)      # also strongly damped stub modes
k = int(np.argmax(q["Q_L"]))                      # the cavity mode
f0, q_l = q["frequencies"][k], q["Q_L"][k]
print(f"   cavity mode {f0 / 1e9:.4f} GHz: Q_L = {q_l:.0f}, "
      f"Qext port1/port2 = {q['Qext']['port1'][k]:.0f}/{q['Qext']['port2'][k]:.0f}")
# Q_L is the 3-dB width of |S21| (1/Q_L = 1/Qext1 + 1/Qext2):
rom_q.solve(fmin=f0 * (1 - 3 / q_l) / 1e9, fmax=f0 * (1 + 3 / q_l) / 1e9, nsamples=3001)
s21 = np.abs(np.asarray(rom_q.S_dict["1(1)2(1)"]))
band = rom_q.frequencies[s21 >= s21.max() / np.sqrt(2)]
print(f"   |S21| 3-dB width: Q_L = {f0 / (band[-1] - band[0]):.0f}")
# The unloaded Q (copper walls) of the same mode, from its closed-problem index:
fm_q = rom_q.get_figures_of_merit(int(q["mode_index"][k]))
print(f"   unloaded Q = {fm_q['Q []']:.0f}, G = {fm_q['G [Ohm]']:.0f} Ohm")
# A ROM is trusted near its training band only (far from it the projection has
# spurious eigenvalues): get_resonant_frequencies() lists the modes within 10 %
# of the band's edges, 1.71-2.53 GHz here, and the mode indices of
# get_eigenmode / get_rq / get_figures_of_merit count that list.  fmin=0 (and
# fmax=) list others.
f_in = rom_q.get_resonant_frequencies()
f_all = rom_q.get_resonant_frequencies(fmin=0)
print(f"   ROM resonances: {len(f_in)} near the training band, {len(f_all)} with fmin=0")
# get_external_q(), get_rq() and get_figures_of_merit() work the same way on a
# joined model (roms.concatenate()): its coupled eigenproblem, with the parts
# of a coupled chain laid end to end along the beam axis.


# =========================================================================== #
# 2. MULTI-SOLID MODEL (one glued mesh):  foms -> roms -> concat [-> rom]     #
# =========================================================================== #
banner("2. Multi-solid: fds.foms.reduce() -> roms.concatenate() [-> .reduce()]")

proj2 = EMProject(name="multi_solid", base_dir=str(WORK), overwrite=True)
proj2.create_primitive("rwg", name="h1", a=A, L=L, b=B_, maxh=MAXH)
proj2.create_primitive("rwg", name="h2", a=A, L=L, b=B_, maxh=MAXH)
#   a second part: the geometry becomes a chain h1 -> h2 along +Z
proj2.generate_mesh(maxh=MAXH)
#   prints "Main axis: Z (default)", the parts in order, and
#   "Mesh strategy: glued" -- plain parts, each used once: ONE glued mesh

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
banner("3. Netlist repeat-N:  create_primitive(..., n=3) -> the SAME fds pipeline")

proj3 = EMProject(name="chain_module", base_dir=str(WORK), overwrite=True)
proj3.create_primitive("rwg", name="cell", n=3, a=A, L=L, b=B_, maxh=MAXH)
#        ^ 3 consecutive copies, computed ONCE (default n=1).  A repeated part
#          is COUPLED: consecutive copies join through the ports that face
#          each other along the main axis.

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

#   Each section is solved in a throwaway scratch project, staged in, and the
#   scratch deleted at once.  fds/sections.json records the settings and the
#   geometry each section was solved for, and its port data.  So:
proj3.fds.solve(config=FOM_CFG)             # same request: the plan says 'reuse'

roms3 = proj3.fds.foms.reduce(tol=1e-6)     # STAGE 2: ROM per unique component
roms3 = proj3.fds.foms.reduce(tol=1e-9)     # ... again, tighter: reduced from the
#                                             staged full-order files (a section
#                                             reduced with the same tol is reused)
concat3 = roms3.concatenate()               # STAGE 3: netlist expanded + coupled
print(f"   {len(concat3.structures)} coupled instances, "
      f"{concat3.n_external_ports} external ports")
# The joined ports are gone; the rest are renumbered part by part.  Which part
# (copy) and which of its own ports each external port is:
concat3.print_port_map()

res3 = concat3.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=200))
print(f"   |S21| at mid-band ~ {abs(res3['S'][100, 1, 0]):.3f} (matched guide -> ~1)")

# A new session restores every stage from the project, the joined model's sweep
# included -- nothing is solved again:
proj3_later = EMProject(name="chain_module", base_dir=str(WORK))
restored = proj3_later.fds.foms.roms.concat
print(f"   reopened: {proj3_later.fds.foms}, joined model with its "
      f"{len(restored.frequencies)}-point sweep")

rom_of_concat = concat3.reduce(tol=1e-10)   # STAGE 4 (optional): concat.rom
print(f"   further-reduced coupled system: {type(rom_of_concat).__name__}")

# --------------------------------------------------------------------------- #
# Chain eigenmodes, and why identical cells need a balanced basis              #
# --------------------------------------------------------------------------- #
# chain_eigenfrequencies() returns (indices, GHz) of the modes that
# get_resonant_frequencies() lists.  Every eigen method of the joined model
# counts that list: reconstruct_chain_eigenmode(), chain_axis_profile(),
# plot_eigenmode(), get_eigenmode(), get_rq(), get_figures_of_merit() and the
# mode_index of get_external_q() -- so a mode found here can be drawn directly
# on a compound mesh of the replicated section.
idx3, f3 = concat3.chain_eigenfrequencies(fmin_ghz=1.8, fmax_ghz=2.4)
print(f"   {len(f3)} chain modes in band; first at {f3[0]:.4f} GHz")

# N identical cells put each mode into a near-exact degenerate group of N. Any
# orthonormal basis of that group is a valid set of eigenvectors, and eigh
# returns an arbitrary one -- typically localised on a single cell, so the
# exported field shows one cell lit and the rest dark. reconstruct_chain_
# eigenmode() therefore rotates inside the group to the most evenly spread
# combination. Pass balance_degenerate=False for the raw eigh basis.
cf3, cmesh3, label3 = concat3.reconstruct_chain_eigenmode(int(idx3[0]),
                                                          component="abs")
print(f"   reconstructed '{label3}' on {cmesh3.ne} elements")


# =========================================================================== #
# 4. IMPORT AN ALREADY-RUN PROJECT and mix it with new geometry               #
# =========================================================================== #
banner("4. Cross-project reuse:  proj.import_project(path) -> the SAME pipeline")

proj4 = EMProject(name="mixed_module", base_dir=str(WORK), overwrite=True)
proj4.create_primitive("rwg", name="fresh", a=A, L=L, b=B_, maxh=MAXH)

# 'single_rwg' from section 1 is a campaign you ran earlier -- import it twice:
legacy = proj4.import_project(WORK / "single_rwg", name="legacy", n=2)
print(f"   {legacy}")               # handle: mode, what it holds, ports, band

proj4.fds.solve(config=FOM_CFG)
#   prints the solve plan: 'fresh' -> compute (full-order solve),
#   'legacy' -> reuse (its reduced model fits the request)
concat4 = proj4.fds.foms.reduce(tol=1e-9).concatenate()
res4 = concat4.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=100))
print(f"   mixed netlist coupled: {len(concat4.structures)} sections, "
      f"|S21| ~ {abs(res4['S'][50, 1, 0]):.3f}")
# mode='reference' (default): nothing of 'legacy' is copied -- its reduced
# model is read from single_rwg/ (recorded with a fingerprint in
# fds/imports.json; reopening this project warns if single_rwg changed or
# moved).  mode='copy' copies it in, folder-to-matching-folder and renamed to
# the part's name (K_legacy.h5 next to K_fresh.h5 ...), so the project stands
# alone; a copy is a snapshot and keeps working if the source is deleted.
n_local = proj4.localize()          # turn the references into copies now
print(f"   localize(): {n_local} referenced part(s) copied into the project")


# --------------------------------------------------------------------------- #
# 4a. WHEN AN IMPORTED PART DOES NOT FIT: solve() computes it HERE             #
# --------------------------------------------------------------------------- #
banner("4a. Solve plan: an imported part that does not fit is recomputed here")

proj4a = EMProject(name="wider_band", base_dir=str(WORK), overwrite=True)
proj4a.import_project(WORK / "single_rwg", name="guide", n=2)
# single_rwg was trained on 1.8-2.4 GHz; this project asks for up to 2.6 GHz,
# so the plan says 'recompute' and the full-order solve runs in THIS project
# (from single_rwg's geometry).  single_rwg itself is never touched.  In a
# script (nobody to answer), a full-order recompute of an imported part needs
# rerun=True; in a notebook the plan is printed and run.
proj4a.fds.solve(config=dict(FOM_CFG, fmax=2.6), rerun=True)
concat4a = proj4a.fds.foms.reduce(tol=1e-9).concatenate()
print(f"   trained band now {concat4a.structures[0].training_band}")


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
     degenerate modes (e.g. the two TE11 polarisations) must be numbered
     alike on both sides -- parts solved with cavsim3d are.
   * A join that carries FEWER modes than propagate below the band's top
     -> UserWarning naming the port and the count needed: the modes left out
     see a magnetic wall and are reflected at the join.
   * flip=True turns a GLUED part end-for-end; a coupled part must be solved
     in the orientation it is used (NotImplementedError).
   * ROM training bands: sections reduced over DISJOINT frequency bands
     cannot be coupled (ValueError).  Narrow overlap or sweeping outside the
     shared band -> UserWarning (extrapolation beyond snapshot coverage:
     results may be inaccurate or wrong).
   * Importing a MULTI-SOLID project as one netlist section -> ValueError
     (a section is one domain; add its solids individually instead).
   * Port modes are built in a frame that does not depend on which way a
     port face points, so odd modes (TE01, TE20, ...) have the SAME sign on
     both faces of a join and couple correctly mode-by-mode.  Modes with
     equal cutoffs (TE01/TE20 when a = 2b) are ordered by type and indices,
     so 'mode 2' is the same mode on every port.
""")


# =========================================================================== #
# 6. BEAM EXCITATION:  proj.add_beam() -> fom.s_tilde, fom.beam_impedance()   #
# =========================================================================== #
banner("6. Beam: proj.add_beam() -> fom.s_tilde, fom.beam_impedance()")

# A beam is a line current of 1 A along proj.main_axis at the speed of light
# (beta = 1), at the transverse position x=, y= (metres; default: the axis).
# It is project-level input (saved in project.json, the same name replaces).
# The same solve() adds one column per beam and one row per voltage path --
# every beam is a path; add_beam_path() adds paths without current -- to the
# port results, as the generalised matrices
#     fom.s_tilde = [[S, k], [h, z_b]]      fom.z_tilde = [[Z, k_Z], [h_Z, z_oc]]
# k: the waves the beam sends into the port modes; h: the beam voltage of an
# incoming wave; z_b / z_oc: the beam impedance with every port mode matched /
# open.  fom.beam_impedance() = Z_par = -z_b (ports='open': -z_oc).  Labels:
# port modes '1(1)', ..., beams and paths 'b(1)', ...; keys excitation first
# ('b(1)2(1)' = k into port 2 mode 1, 'b(1)b(1)' = z_b).
# The solver carries the scattered field E_s = E - E_free (the beam's own field
# E_free is known in closed form): one factorisation per sample serves ports
# and beams, the beam line need not be part of the mesh, and without a beam
# every port result is bit-identical.  Curved walls: with a beam defined,
# generate_mesh() curves the mesh to order 4 unless curve_order= is given (the
# beam impedance is sensitive to the wall's facets; a solve on a mesh curved
# to a lower order warns).
# Materials off the beam line (dielectric, lossy) are fine; the beam itself
# must run in vacuum and enter and leave through port faces across the axis.
from netgen.occ import Glue


class SteppedGuide(BaseGeometry):
    """A 60 x 40 mm guide stepping down to 60 x 25 mm.  Two solids, 'wide' and
    'narrow', meet 60 mm after the step (internal port 'port3')."""

    def build(self):
        a, b, b2 = 0.06, 0.04, 0.025
        wide = Box(Pnt(0, 0, 0), Pnt(a, b, 0.05)) + Box(Pnt(0, 0, 0.05), Pnt(a, b2, 0.11))
        narrow = Box(Pnt(0, 0, 0.11), Pnt(a, b2, 0.16))
        wide.mat("wide")
        narrow.mat("narrow")
        self.geo = Glue([wide, narrow])
        for f in self.geo.faces:
            lo, hi = f.bounding_box
            across = hi.z - lo.z < 1e-6           # a face across the axis
            f.name = ({0.0: "port1", 0.16: "port2", 0.11: "port3"}.get(round(lo.z, 6), "default")
                      if across else "default")
        self.bc = "default"


BEAM_CFG = dict(fmin=3.0, fmax=4.0, nsamples=3, nportmodes=3, order=2, solver_type="direct")

# 6a. One piece: the beam on the narrow guide's axis, through the step
proj6 = EMProject(name="beam_step", base_dir=str(WORK), overwrite=True)
proj6.geometry = SteppedGuide()
proj6.generate_mesh(maxh=0.012)
proj6.add_beam("beam", x=0.03, y=0.0125)
proj6.fds.solve(config=dict(BEAM_CFG, per_domain=False))      # the whole mesh: fds.fom
fom6 = proj6.fds.fom
rows, cols = fom6.tilde_labels
print(f"   S~ rows {rows}\n      columns {cols}")
print(f"   Z_par = {np.round(fom6.beam_impedance(), 3)} Ohm "
      f"(open ports: {np.round(fom6.beam_impedance(ports='open'), 3)})")
#   fom6.s_tilde_dict['b(1)1(1)'] (k into port 1 mode 1), fom6.plot_beam_impedance(),
#   fom6.beam_field(i) (E_s + E_free at sample i), fom6.z_tilde: all saved in
#   fds/fom/{s_tilde,z_tilde,snapshots_beam}/ and matrices/beam_global.h5.

# 6b. A beam added to a SOLVED project: only the beam columns are solved (with
# the stored port solutions), the port results stay bit for bit.  The same
# after reopening; remove_beam() drops the beam results again.
proj1b = EMProject(name="single_rwg", base_dir=str(WORK))       # solved in section 1
S_before = proj1b.fds.fom._S_matrix.copy()
proj1b.add_beam("beam", x=A / 2, y=B_ / 2)
proj1b.fds.solve(config=FOM_CFG)
print(f"   beam added to single_rwg: S unchanged "
      f"{np.array_equal(proj1b.fds.fom._S_matrix, S_before)}, S~ "
      f"{proj1b.fds.fom.s_tilde.shape[1:]}")
#   In a uniform guide the beam's field is the guide's own: Z_par, k and h are 0.
#   This mesh (maxh 60 mm, made for the port modes) is far too coarse for the
#   beam's field, which is strong near the walls: a beam needs a finer mesh
#   (maxh 10 mm, order 3 here gives |Z_par| < 1e-3 Ohm; tests/test_beam.py).

# 6c. FOM + concatenation: the parts solved one by one (their faces to each
# other are ports), each with its S~; foms.concatenate() joins them at the cut
# (CSC-BEAM): the waves leaving one face enter the other, the beam current is
# the same in both parts, their beam voltages add.  No full-order matrices are
# built; the join holds the full-order frequencies (other frequencies: join
# reduced models, 6e).
proj6p = EMProject(name="beam_step_parts", base_dir=str(WORK), overwrite=True)
proj6p.geometry = SteppedGuide()
proj6p.generate_mesh(maxh=0.012)
proj6p.add_beam("beam", x=0.03, y=0.0125)
proj6p.fds.solve(config=dict(BEAM_CFG, per_domain=True, global_method=None))
joined6 = proj6p.fds.foms.concatenate()
rel = np.abs(joined6.beam_impedance() - fom6.beam_impedance()) / np.abs(fom6.beam_impedance())
print(f"   parts joined vs one piece: Z_par differs by {rel.max():.1%} "
      f"(the modes carried at the cut, and the two meshes' solutions)")

# 6d. COUPLED parts (a part repeated with n=..., or an imported project): each
# unique part is solved once, with the beams where they run through it, in its
# own frame.  The beams are given in the first part's frame; every next part
# sits with its joined face centred on the face it joins.  fds.foms.concatenate()
# joins the copies through their S~, each with the phase of its position along
# the axis (exp(-j k z) on its beam columns, exp(+j k z) on its path rows).  An
# imported part solved without a beam gets its beam columns computed HERE from
# its stored port solutions (its project is read, never written); its samples
# must be the requested ones, or it is solved again here (rerun=True).


class Cell(BaseGeometry):
    """A 70 x 50 x 40 mm box between two 60 x 40 mm pipes of 80 mm, along z."""

    def build(self):
        def pipe(z0):
            return Box(Pnt(0.005, 0.005, z0), Pnt(0.065, 0.045, z0 + 0.08))
        self.geo = pipe(0.0) + Box(Pnt(0, 0, 0.08), Pnt(0.07, 0.05, 0.12)) + pipe(0.12)
        self.geo.mat("vacuum")
        for f in self.geo.faces:
            lo, hi = f.bounding_box
            across = hi.z - lo.z < 1e-6
            f.name = ({0.0: "port1", 0.2: "port2"}.get(round(lo.z, 6), "default")
                      if across else "default")
        self.bc = "default"


CELL_CFG = dict(fmin=2.6, fmax=3.4, nsamples=3, nportmodes=1, order=3, solver_type="direct")
proj6c = EMProject(name="beam_cells", base_dir=str(WORK), overwrite=True)
proj6c.add("cell", Cell(), n=2)                    # repeated: coupled through port modes
proj6c.generate_mesh(maxh=0.014)
proj6c.add_beam("beam", x=0.035, y=0.03)           # 5 mm off the pipe's centre
proj6c.fds.solve(config=CELL_CFG)                  # the cell once, with the beam
chain6 = proj6c.fds.foms.concatenate()             # the copies joined through S~
proj6w = EMProject(name="beam_cells_whole", base_dir=str(WORK), overwrite=True)
proj6w.add("c1", Cell())
proj6w.add("c2", Cell())                           # each once: glued, one mesh
proj6w.generate_mesh(maxh=0.014)
proj6w.add_beam("beam", x=0.035, y=0.03)
proj6w.fds.solve(config=dict(CELL_CFG, per_domain=False))   # in one piece
z_whole = proj6w.fds.fom.beam_impedance()
rel = np.abs(chain6.beam_impedance() - z_whole) / np.abs(z_whole)
print(f"   two copies joined vs one piece: Z_par differs by {rel.max():.1%}")
#   proj.import_project(path, name="cell", n=2) instead of proj.add(...): the
#   solve plan says "beam columns computed here from its port solutions".

# 6e. FOM -> ROM -> Concatenation WITH the beam.  A sweep that keeps its field
# snapshots (store_snapshots=True, the default) is reduced with the beam: one
# basis for the port and the beam columns (each family scaled by its largest
# singular value; only the beam field's free part -- its wall values are the
# lift of -E_free, added back).  The beam's phase runs along the structure, so
# its load and outputs are stored at Chebyshev points of the band and
# interpolated: the reduced model gives S~ at any frequency of its band (+10 %
# on each side; further raises) without the mesh.  roms.concatenate() joins the
# reduced parts' S~ at every concat.solve(), each copy with the phase of its
# position.  Sample finer than v_b / (2 L) for a structure of length L.
# A single part: proj.fds.fom.reduce(tol) -> rom.solve(...) -> rom.s_tilde,
# rom.beam_impedance().  docs/theory/beam_reduction.md (section 10).
proj6c.fds.solve(config=dict(CELL_CFG, nsamples=9))      # snapshots for the reduction
fom_join6 = proj6c.fds.foms.concatenate()                 # at the 9 full-order samples
concat6 = proj6c.fds.foms.reduce(tol=1e-8).concatenate()
concat6.solve(fmin=2.6, fmax=3.4, nsamples=9)
rel = (np.abs(concat6.beam_impedance() - fom_join6.beam_impedance())
       / np.abs(fom_join6.beam_impedance()))
print(f"   reduced copies joined vs full-order join: Z_par differs by {rel.max():.1e}")
concat6.solve(fmin=2.6, fmax=3.4, nsamples=401)           # any frequencies of the band
print(f"   reduced join, 401 frequencies: S~ {concat6.s_tilde.shape}")

print(f"All tutorial artifacts under: {WORK}")
banner("DONE")
