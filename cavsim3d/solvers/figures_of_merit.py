"""Figures of merit of a cavity eigenmode.

The quantities, their keys and their units follow cavsim2d's ``evaluate_qois``:
R/Q in the linac convention ``V^2 / (w U)``, peak surface fields, the wall Q
and geometry factor for a surface resistance, the dielectric Q from the
material loss terms, the transverse kick by Panofsky-Wenzel and the field
flatness.

Every function here works on a list of :class:`ModePiece`, the mode on one
mesh placed along the beam axis, so a joined model whose sections have
separate meshes is evaluated the same way as a single mesh.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cavsim3d.core.constants import SIGMA_COPPER, c0, mu0


@dataclass
class ModePiece:
    """The electric field of an eigenmode on one mesh.

    ``shift`` [m] moves the mesh along the beam axis into the model's frame.
    ``walls`` is the region pattern of its conducting boundaries, ``mu_r`` the
    relative permeability (a CoefficientFunction or a number) and ``eps_r``
    the relative permittivity of each mesh material.
    """
    mesh: Any
    E: Any
    shift: float = 0.0
    walls: str = 'default'
    mu_r: Any = 1.0
    eps_r: Dict[str, float] = field(default_factory=dict)


def surface_resistance(w: float, conductivity: float = SIGMA_COPPER,
                       rs: Optional[float] = None) -> float:
    """Surface resistance [Ohm] at angular frequency *w*.

    ``sqrt(mu0 w / (2 conductivity))`` for a normal conductor; *rs* [Ohm]
    replaces it (a superconductor, or a measured value).
    """
    if rs is not None:
        return float(rs)
    return float(np.sqrt(mu0 * w / (2 * conductivity)))


def _trapezoid(y, x):
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    return trapezoid(y, x)


def _coordinates(mesh) -> np.ndarray:
    return np.asarray(mesh.ngmesh.Coordinates())


def beam_line(pieces: Sequence[ModePiece], a: int, span, n_points: int) -> np.ndarray:
    """Sample positions along axis *a*; by default the extent of all pieces."""
    if span is None:
        lo = min(_coordinates(p.mesh)[:, a].min() + p.shift for p in pieces)
        hi = max(_coordinates(p.mesh)[:, a].max() + p.shift for p in pieces)
        pad = 1e-6 * (hi - lo)
        span = (lo + pad, hi - pad)
    return np.linspace(float(span[0]), float(span[1]), int(n_points))


def inside_samples(mesh, a: int, point, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(mask, s_eval)``: which positions *s* along the line parallel to axis
    *a* through *point* lie in the mesh, and where to evaluate them -- a
    position on the boundary is moved just inside.

    From the line's crossings with the mesh boundary, so no point outside the
    mesh is located (NGSolve's point search can crash there on a curved mesh).
    """
    from cavsim3d.utils.mesh_geometry import line_intervals
    d = 1e-9 * float(np.ptp(_coordinates(mesh)[:, a]))
    mask = np.zeros(len(s), dtype=bool)
    s_eval = np.array(s, dtype=float)
    for s0, s1, _, _ in line_intervals(mesh, point, a):
        dd = min(d, 0.25 * (s1 - s0))
        hit = (s >= s0 - dd) & (s <= s1 + dd) & ~mask
        s_eval[hit] = np.clip(s[hit], s0 + dd, s1 - dd)
        mask |= hit
    return mask, s_eval


def field_on_line(pieces: Sequence[ModePiece], a: int, offset, s: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """``(E_a(s), inside)`` along the line parallel to axis *a* through the
    transverse point *offset*.  Points outside every mesh (in a conductor)
    give 0 and ``inside = False``."""
    others = [i for i in range(3) if i != a]
    e = np.zeros(len(s), dtype=complex)
    inside = np.zeros(len(s), dtype=bool)
    point = [0.0, 0.0, 0.0]
    point[others[0]], point[others[1]] = float(offset[0]), float(offset[1])
    for p in pieces:
        mask, s_eval = inside_samples(p.mesh, a, point, s - p.shift)
        hit = mask & ~inside
        n = int(hit.sum())
        if not n:
            continue
        xyz = [None, None, None]
        xyz[a] = s_eval[hit]
        xyz[others[0]] = np.full(n, point[others[0]])
        xyz[others[1]] = np.full(n, point[others[1]])
        mips = p.mesh(*xyz)
        found = mips['nr'] >= 0
        idx = np.nonzero(hit)[0][found]
        if len(idx):
            vals = np.asarray(p.E(mips[found])).reshape(len(idx), 3)
            e[idx] = vals[:, a]
            inside[idx] = True
    return e, inside


def voltage(e_s: np.ndarray, s: np.ndarray, w: float, beta: float = 1.0) -> complex:
    """``int E_s exp(j w s / (beta c0)) ds``: the voltage a charge at beta*c gains."""
    return complex(_trapezoid(e_s * np.exp(1j * w * s / (beta * c0)), s))


def transverse_voltage(pieces, a, offset, s, w, beta=1.0, step=None,
                       inside_axis=None) -> Tuple[float, float]:
    """``(|V_t|, step)``: the transverse kick voltage at *offset* [V].

    Panofsky-Wenzel, ``V_t = (beta c0 / w) |grad_t V_z|``, with the gradient
    of the longitudinal voltage by central differences over *step* [m] in both
    transverse directions.  A dipole mode's ``V_z`` is linear in the offset,
    so the difference is exact for it; a monopole's is even and drops out.

    Without *step*, it starts at 2% of the smallest transverse extent and
    halves until the four shifted lines stay inside the mesh wherever the beam
    line is (a narrow aperture).
    """
    auto = step is None
    if auto:
        others = [i for i in range(3) if i != a]
        widths = [np.ptp(_coordinates(p.mesh)[:, i]) for p in pieces for i in others]
        step = 0.02 * min(widths)
    for _ in range(8):
        v, clipped = {}, False
        for d in (0, 1):
            for sign in (1, -1):
                off = [float(offset[0]), float(offset[1])]
                off[d] += sign * step
                e, inside = field_on_line(pieces, a, off, s)
                if auto and inside_axis is not None and np.sum(inside_axis & ~inside) > 2:
                    clipped = True
                v[d, sign] = voltage(e, s, w, beta)
        if not (auto and clipped):
            break
        step *= 0.5
    grad = [(v[d, 1] - v[d, -1]) / (2 * step) for d in (0, 1)]
    return float(np.sqrt(abs(grad[0]) ** 2 + abs(grad[1]) ** 2) * beta * c0 / w), step


def _lattice(et, n: int):
    """Integration rule whose points are a uniform lattice on the reference
    element, vertices and edges included (weights unused)."""
    from ngsolve import IntegrationRule, QUAD, TET, TRIG
    if et == TRIG:
        pts = [(i / n, j / n, 0.0) for i in range(n + 1) for j in range(n + 1 - i)]
    elif et == QUAD:
        pts = [(i / n, j / n, 0.0) for i in range(n + 1) for j in range(n + 1)]
    elif et == TET:
        pts = [(i / n, j / n, k / n) for i in range(n + 1)
               for j in range(n + 1 - i) for k in range(n + 1 - i - j)]
    else:
        raise ValueError(f"no sampling lattice for {et}")
    return IntegrationRule(pts, [0.0] * len(pts))


def _indicator(region):
    """1 on *region* (a Region of the mesh), 0 elsewhere."""
    from ngsolve import CoefficientFunction
    mask = region.Mask()
    return CoefficientFunction([1.0 if mask[i] else 0.0 for i in range(len(mask))])


def _field_order(E) -> int:
    space = getattr(E, 'space', None)
    return int(getattr(space, 'globalorder', 2) or 2)


def surface_peaks(piece: ModePiece, w: float) -> Tuple[float, float]:
    """``(max |E|, max |H|)`` on the walls of *piece*.

    The wall trace of the discrete field jumps from one element to the next,
    and its largest value over many elements overshoots the peak (by ~20 %
    for the TESLA cell at maxh 0.04, order 2).  The magnitudes are therefore
    averaged into continuous fields on the walls first (NGSolve's local
    projection, ``Set``), then sampled on a lattice over every wall element,
    edges and vertices included.  The magnitude, not the vector, is averaged:
    at an edge between two walls the vectors point along different normals.
    """
    from ngsolve import (BND, H1, QUAD, TRIG, BoundaryFromVolumeCF, GridFunction, Norm,
                         curl)
    mesh = piece.mesh
    walls = mesh.Boundaries(piece.walls)
    if not any(walls.Mask()):
        return 0.0, 0.0
    p = _field_order(piece.E)
    e = GridFunction(H1(mesh, order=p, definedon=walls))
    e.Set(BoundaryFromVolumeCF(Norm(piece.E)), definedon=walls)
    h = GridFunction(H1(mesh, order=max(p - 1, 1), definedon=walls))
    h.Set(BoundaryFromVolumeCF(Norm(curl(piece.E)) / piece.mu_r), definedon=walls)
    n = max(2, 2 * p)
    pts = mesh.MapToAllElements({TRIG: _lattice(TRIG, n), QUAD: _lattice(QUAD, n)}, BND)
    pts = pts[np.asarray(_indicator(walls)(pts)).ravel() > 0.5]
    if len(pts) == 0:
        return 0.0, 0.0
    return (float(np.max(np.asarray(e(pts)))),
            float(np.max(np.asarray(h(pts)))) / (w * mu0))


def wall_h2(piece: ModePiece, w: float) -> float:
    """``int |H|^2 dS`` over the walls of *piece*.

    On a wall with zero tangential E the discrete ``n . curl E`` vanishes
    exactly, so ``|H|^2`` is its tangential part, and the wall trace is taken
    from the volume element next to the wall.
    """
    from ngsolve import BoundaryFromVolumeCF, InnerProduct, Integrate, curl
    mesh = piece.mesh
    walls = mesh.Boundaries(piece.walls)
    if not any(walls.Mask()):
        return 0.0
    h = BoundaryFromVolumeCF(curl(piece.E) / piece.mu_r)
    val = Integrate(InnerProduct(h, h), mesh, definedon=walls,
                    order=2 * _field_order(piece.E) + 2)
    return abs(complex(val).real) / (w * mu0) ** 2


def material_shares(pieces: Sequence[ModePiece]) -> Dict[str, Tuple[float, float]]:
    """``{material: (share of the electric energy, max |E| inside)}``.

    As on the walls, ``|E|`` is averaged into a continuous field before its
    peak is taken, within each material only: the normal field jumps at a
    material interface.
    """
    from ngsolve import (VOL, TET, H1, BitArray, GridFunction, InnerProduct, Integrate,
                         Norm, Region)
    energy: Dict[str, float] = {}
    peak: Dict[str, float] = {}
    for p in pieces:
        mesh = p.mesh
        names = list(mesh.GetMaterials())
        order = _field_order(p.E)
        el_mat = np.array([el.index for el in mesh.Elements(VOL)])
        per_el = np.asarray(Integrate(InnerProduct(p.E, p.E), mesh, VOL,
                                      element_wise=True, order=2 * order)).real
        pts = mesh.MapToAllElements(_lattice(TET, max(2, order)), VOL)
        pt_mat = el_mat[pts['nr']]
        for idx, name in enumerate(names):
            sel = el_mat == idx
            if not sel.any():
                continue
            energy[name] = energy.get(name, 0.0) + p.eps_r.get(name, 1.0) * per_el[sel].sum()
            mask = BitArray(len(names))
            mask.Clear()
            mask.Set(idx)
            region = Region(mesh, VOL, mask)
            e = GridFunction(H1(mesh, order=order, definedon=region))
            e.Set(Norm(p.E), definedon=region)
            hits = pts[pt_mat == idx]
            if len(hits):
                peak[name] = max(peak.get(name, 0.0), float(np.max(np.asarray(e(hits)))))
    total = sum(energy.values())
    return {name: (energy[name] / total if total > 0 else np.nan, peak.get(name, 0.0))
            for name in energy}


def field_flatness(e_abs: np.ndarray, s: np.ndarray, n_cells: int,
                   cell_length: float) -> float:
    """``min / max`` [%] of the *n_cells* highest peaks of ``|E_s|``.

    Peaks closer than half a cell are one peak: the discrete longitudinal
    field jumps between elements and would otherwise count twice.
    """
    from scipy.signal import find_peaks
    ds = (s[-1] - s[0]) / max(len(s) - 1, 1)
    distance = max(1, int(0.5 * cell_length / ds)) if ds > 0 else 1
    peaks, _ = find_peaks(e_abs, distance=distance)
    if len(peaks) < n_cells:
        return np.nan
    top = np.sort(e_abs[peaks])[-n_cells:]
    return float(100 * top.min() / top.max())


def figures_of_merit(pieces: List[ModePiece], freq: float, U: float, P_diel: float,
                     axis: str = 'Z', offset=(0.0, 0.0), span=None,
                     n_points: int = 2001, beta: float = 1.0,
                     active_length: Optional[float] = None, n_cells: int = 1,
                     conductivity: float = SIGMA_COPPER,
                     surface_resistance_ohm: Optional[float] = None,
                     kick_step: Optional[float] = None) -> Dict[str, float]:
    """The figures of merit of one eigenmode, scaled to a stored energy of 1 J.

    *U* and *P_diel* are the stored energy and the material loss of the
    field in *pieces* at its own amplitude.
    """
    w = 2 * np.pi * freq
    a = 'XYZ'.index(axis.upper())
    s = beam_line(pieces, a, span, n_points)
    e_s, inside = field_on_line(pieces, a, offset, s)
    V = abs(voltage(e_s, s, w, beta))
    Vt, _step = transverse_voltage(pieces, a, offset, s, w, beta, kick_step, inside)
    peaks = [surface_peaks(p, w) for p in pieces]
    Epk = max(e for e, _ in peaks)
    Hpk = max(h for _, h in peaks)
    Rs = surface_resistance(w, conductivity, surface_resistance_ohm)
    P_wall = 0.5 * Rs * sum(wall_h2(p, w) for p in pieces)

    # to U = 1 J: fields scale by 1/sqrt(U), powers by 1/U
    root = np.sqrt(U)
    V, Vt, Epk, Hpk = V / root, Vt / root, Epk / root, Hpk / root
    P_wall, P_diel = P_wall / U, P_diel / U
    length = float(active_length) if active_length else float(s[-1] - s[0])
    Eacc, Et = V / length, Vt / length
    k = w / (beta * c0)

    with np.errstate(divide='ignore', invalid='ignore'):
        Q_wall = w / P_wall if P_wall > 0 else np.inf
        Q_diel = w / P_diel if P_diel > 0 else np.inf
        Q = w / (P_wall + P_diel) if P_wall + P_diel > 0 else np.inf
        RQ = V ** 2 / w
        G = Q_wall * Rs
        epk_eacc = Epk / Eacc if Eacc > 0 else np.inf
        bpk_eacc = mu0 * Hpk * 1e9 / Eacc if Eacc > 0 else np.inf

    qois = {
        "freq [MHz]": freq * 1e-6,
        "Q []": Q,
        "Vacc [MV]": V * 1e-6,
        "Eacc [MV/m]": Eacc * 1e-6,
        "Epk [MV/m]": Epk * 1e-6,
        "Hpk [A/m]": Hpk,
        "Bpk [mT]": mu0 * Hpk * 1e3,
    }
    if n_cells > 1:
        qois["ff [%]"] = field_flatness(np.abs(e_s), s, n_cells, length / n_cells)
    qois.update({
        "Rsh [MOhm]": RQ * Q * 1e-6,
        "R/Q [Ohm]": RQ,
        "Epk/Eacc []": epk_eacc,
        "Bpk/Eacc [mT/MV/m]": bpk_eacc,
        "G [Ohm]": G,
        "GR/Q [Ohm^2]": G * RQ,
        "U [J]": 1.0,
        "Ploss [W]": P_wall,
        "k_loss [V/pC]": V ** 2 / 4 * 1e-12,
        "Vt [MV]": Vt * 1e-6,
        "Et [MV/m]": Et * 1e-6,
        "R/Q_t [Ohm]": Vt ** 2 / w,
        "k_kick [V/pC/m]": k * Vt ** 2 / 4 * 1e-12,
        "Rs [Ohm]": Rs,
        "Active Length [mm]": length * 1e3,
        "N Cells": int(n_cells),
    })
    if P_diel > 0:
        qois["Q_wall []"] = Q_wall
        qois["Q_diel []"] = Q_diel
        qois["Pdiel [W]"] = P_diel

    materials = {name for p in pieces for name in p.mesh.GetMaterials()}
    if len(materials) > 1:
        for name, (share, peak) in sorted(material_shares(pieces).items()):
            qois[f"U_frac_{name} []"] = share
            qois[f"Epk_{name} [MV/m]"] = peak / root * 1e-6
    return qois
