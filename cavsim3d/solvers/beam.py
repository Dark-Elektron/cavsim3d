"""Beam excitation: a beam travelling parallel to the main axis (docs/theory/beam.md §9).

Scattered-field formulation (§9.6), beta = 1 in vacuum.  The beam's own field
E_free is known in closed form and never put on the mesh; the finite elements
carry the scattered field E_s = E - E_free, driven by

* the Dirichlet lift g(w): the interpolant of -E_free on the PEC walls;
* a load on every port face: n x H_s = -n_a v_b eps_b E_reg,t on a face the
  beam crosses (E_reg: the beam's field in the pipe, minus E_free), and
  n x H_s = -n x H_free on a face it does not cross;
* a volume load wherever the material is not vacuum (the contrast load);

with the system matrix A(w) of the port excitations, so one factorisation or
preconditioner per sample serves the ports and the beams.  The beam current is
i = 1 A: every output is per ampere.

The beam voltage v = int E_a exp(+j k_b s) ds is read along a *path* (a line
parallel to the main axis, s its coordinate along the axis) at Gauss points on
the pieces of the line inside each element, so the line need not be made of
mesh edges.  Sources (beams) and paths are separate: every beam is a path, and
extra paths (no current) read voltages away from the beam.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from ngsolve import (BND, VOL, BilinearForm, CoefficientFunction, Cross,
                     GridFunction, H1, InnerProduct, Integrate, LinearForm, TaskManager,
                     cos, ds, dx, exp, grad, log, sin, specialcf, x, y, z)

from cavsim3d.core.constants import c0, eps0, mu0
from cavsim3d.utils.mesh_geometry import line_crossings, line_intervals, surface_triangles
from cavsim3d.utils.names import region_pattern

AXES = ('X', 'Y', 'Z')
_COORDS = (x, y, z)

#: version of the stored beam data; a change makes old beam results stale
BEAM_DATA_VERSION = 1

#: curve order of a mesh generated while a beam is defined: the beam impedance
#: is sensitive to how closely the mesh follows curved walls
BEAM_CURVE_ORDER = 4


def axis_index(axis: str) -> int:
    axis = str(axis).upper()
    if axis not in AXES:
        raise ValueError(f"the beam axis must be 'X', 'Y' or 'Z', got {axis!r}")
    return AXES.index(axis)


def transverse_names(axis: str) -> Tuple[str, str]:
    """The two coordinates across the main axis, e.g. ('x', 'y') for 'Z'."""
    a = axis_index(axis)
    return tuple(n for i, n in enumerate('xyz') if i != a)


# =============================================================================
# Definition
# =============================================================================

@dataclass(frozen=True)
class BeamLine:
    """A beam (``current=True``: a source, which is also a voltage path) or a
    voltage path without current, on a line parallel to the main axis.

    ``point`` holds the line's transverse position in metres; its main-axis
    coordinate is 0.
    """
    name: str
    point: Tuple[float, float, float]
    beta: float = 1.0
    current: bool = True

    def to_dict(self) -> Dict:
        return {"name": self.name, "point": [float(v) for v in self.point],
                "beta": float(self.beta), "current": bool(self.current)}

    @classmethod
    def from_dict(cls, d: Dict) -> "BeamLine":
        return cls(name=str(d["name"]), point=tuple(float(v) for v in d["point"]),
                   beta=float(d.get("beta", 1.0)), current=bool(d.get("current", True)))


class BeamSetup:
    """The beams and voltage paths of a project, along its main axis.

    Row and column labels of the generalised matrices: the paths are
    ``b(1)``, ``b(2)``, ... in this order -- every beam first (its own line is
    a path), then the extra paths -- and beam ``i`` is column ``b(i)``.
    """

    def __init__(self, axis: str = 'Z', lines: Sequence[BeamLine] = ()):
        self.axis = AXES[axis_index(axis)]
        self._lines: List[BeamLine] = []
        for line in lines:
            self.add(line)

    # -- editing --------------------------------------------------------------
    def add(self, line: BeamLine) -> None:
        """Add a line; one of the same name is replaced in place."""
        if line.beta != 1.0:
            raise NotImplementedError(
                f"Beam '{line.name}': only beta = 1 is implemented (got beta={line.beta}). "
                "A slower beam needs the Bessel free field (docs/theory/beam.md §9.4).")
        names = [l.name for l in self._lines]
        if line.name in names:
            self._lines[names.index(line.name)] = line
        else:
            self._lines.append(line)

    def remove(self, name: str, current: Optional[bool] = None) -> None:
        keep = [l for l in self._lines
                if not (l.name == name and (current is None or l.current == current))]
        if len(keep) == len(self._lines):
            raise KeyError(f"no beam or beam path named {name!r}")
        self._lines = keep

    # -- views ----------------------------------------------------------------
    @property
    def sources(self) -> List[BeamLine]:
        return [l for l in self._lines if l.current]

    @property
    def paths(self) -> List[BeamLine]:
        """Every voltage path: the beams first, then the paths without current."""
        return self.sources + [l for l in self._lines if not l.current]

    def __bool__(self) -> bool:
        return bool(self.sources)

    @property
    def path_labels(self) -> List[str]:
        return [f"b({i + 1})" for i in range(len(self.paths))]

    @property
    def source_labels(self) -> List[str]:
        return [f"b({i + 1})" for i in range(len(self.sources))]

    def to_dict(self) -> Dict:
        return {"axis": self.axis, "lines": [l.to_dict() for l in self._lines]}

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "BeamSetup":
        if not d:
            return cls()
        return cls(axis=d.get("axis", 'Z'),
                   lines=[BeamLine.from_dict(l) for l in d.get("lines", [])])

    def fingerprint(self) -> str:
        """Identity of the beam definition (what the beam results depend on)."""
        payload = dict(self.to_dict(), version=BEAM_DATA_VERSION)
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def shifted(self, shift) -> "BeamSetup":
        """The same lines seen from a frame whose origin sits at ``shift``
        (a part placed there): the transverse positions minus the shift's,
        to 1 pm, so equal placements give equal fingerprints."""
        a = axis_index(self.axis)
        lines = [BeamLine(l.name, tuple(0.0 if i == a else
                                        round(l.point[i] - float(shift[i]), 12) + 0.0
                                        for i in range(3)), l.beta, l.current)
                 for l in self._lines]
        return BeamSetup(self.axis, lines)

    def same_lines(self, other: "BeamSetup", tol: float = 1e-9) -> bool:
        """True if ``other`` has the same lines (names, kinds, beta) at the
        same positions, to ``tol`` metres."""
        if self.axis != other.axis or len(self._lines) != len(other._lines):
            return False
        return all(l.name == m.name and l.current == m.current and l.beta == m.beta
                   and np.allclose(l.point, m.point, rtol=0, atol=tol)
                   for l, m in zip(self._lines, other._lines))

    def __repr__(self) -> str:
        parts = []
        for l, lab in zip(self.paths, self.path_labels):
            t = transverse_names(self.axis)
            a = axis_index(self.axis)
            pos = ", ".join(f"{n}={l.point[i]:g}" for n, i in zip(t, [i for i in range(3) if i != a]))
            parts.append(f"{lab} {l.name!r} ({'beam' if l.current else 'path'}, {pos})")
        return f"BeamSetup(axis={self.axis}: " + "; ".join(parts) + ")"


# =============================================================================
# Closed-form fields (beta = 1, vacuum, current 1 A)
# =============================================================================

def _geometry_cfs(point, a: int):
    """(rho_vec, rho2, s): the transverse offset from the line, its square, and
    the coordinate along the axis, as CoefficientFunctions."""
    comps = [(_COORDS[i] - point[i]) if i != a else CoefficientFunction(0.0)
             for i in range(3)]
    rho_vec = CoefficientFunction(tuple(comps))
    rho2 = sum(comps[i] * comps[i] for i in range(3) if i != a)
    return rho_vec, rho2, _COORDS[a]


def free_field_profile(point, a: int):
    """E_free / exp(-j k s): i/(2 pi v_b eps0) rho_vec / rho^2 for i = 1 A."""
    rho_vec, rho2, _ = _geometry_cfs(point, a)
    return (1.0 / (2 * np.pi * c0 * eps0)) * rho_vec / rho2


def free_h_profile(point, a: int):
    """H_free / exp(-j k s) = i/(2 pi) (a_hat x rho_vec)/rho^2 for i = 1 A."""
    rho_vec, rho2, _ = _geometry_cfs(point, a)
    ahat = CoefficientFunction(tuple(1.0 if i == a else 0.0 for i in range(3)))
    return (1.0 / (2 * np.pi)) * Cross(ahat, rho_vec) / rho2


# =============================================================================
# Beam line inside a mesh: pieces per element and Gauss points
# =============================================================================

def _locate(mesh, point, a: int, s: np.ndarray, vorb=VOL) -> np.ndarray:
    coords = [np.full_like(s, point[i]) if i != a else s for i in range(3)]
    return mesh(coords[0], coords[1], coords[2], vorb)['nr']


def _element_changes(mesh, point, a: int, s: np.ndarray, tol: float) -> np.ndarray:
    """Where the line changes element between the samples ``s`` (all inside
    the mesh), by bisection to ``tol``; several changes between two samples
    are all found."""
    nr = _locate(mesh, point, a, s)
    change = np.nonzero(nr[1:] != nr[:-1])[0]
    pend_lo, pend_hi = s[change].copy(), s[change + 1].copy()
    e_lo, e_hi = nr[change].copy(), nr[change + 1].copy()
    found = []
    for _ in range(20):                  # one round per further change in an interval
        if not len(pend_lo):
            break
        s_lo, s_hi = pend_lo.copy(), pend_hi.copy()
        for _ in range(80):
            if np.all(s_hi - s_lo <= tol):
                break
            mid = 0.5 * (s_lo + s_hi)
            same = _locate(mesh, point, a, mid) == e_lo
            s_lo = np.where(same, mid, s_lo)
            s_hi = np.where(same, s_hi, mid)
        found.append(0.5 * (s_lo + s_hi))
        e_after = _locate(mesh, point, a, s_hi)
        more = e_after != e_hi
        pend_lo, pend_hi = s_hi[more], pend_hi[more]
        e_lo, e_hi = e_after[more], e_hi[more]
    return np.concatenate(found) if found else np.zeros(0)


def _perpendicular_crossings(mesh, point, a: int) -> List[float]:
    """Axis coordinates where the line crosses a face lying across the axis
    (exact on the straight triangles: such a face stays plane when curved)."""
    tris, _ = surface_triangles(mesh)
    s, idx = line_crossings(tris, point, a)
    if not len(s):
        return []
    t = tris[idx]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    across = np.abs(n[:, a]) >= (1 - 1e-9) * np.linalg.norm(n, axis=1)
    return sorted(set(np.round(s[across], 15).tolist()))


def line_pieces(mesh, point, a: int, domains: Optional[Sequence[int]] = None,
                snap: Sequence[float] = (), n_samples: Optional[int] = None,
                tol: float = 1e-11, snap_tol: float = 1e-4):
    """Break points of the line through ``point`` along axis ``a`` where it
    changes element, and per piece whether it lies in the region (the mesh,
    or its ``domains``, 1-based).

    The stretches in the region come from the line's crossings with the
    region's boundary (:func:`line_intervals`), so no point outside the mesh
    is located (NGSolve's point search can crash there on a curved mesh).
    Inside them, element changes are found by point location -- any element
    type, curved or not -- and refined by bisection to ``tol`` (relative to
    the model's length).  Point location on curved elements accepts points
    slightly outside an element, so a break within ``snap_tol`` (relative) of
    a plane in ``snap`` or of a face across the axis is moved onto it.
    Returns ``(breaks, inside, ends)``: ``ends`` the boundary names where
    each stretch begins and ends.
    """
    coords = np.asarray(mesh.ngmesh.Coordinates())
    lo, hi = float(coords[:, a].min()), float(coords[:, a].max())
    span = hi - lo
    if n_samples is None:
        # about 20 samples per typical element length
        box = np.maximum(np.ptp(coords, axis=0), 1e-12 * max(span, 1e-12))
        h = (float(np.prod(box)) / max(mesh.ne, 1)) ** (1.0 / 3.0)
        n_samples = int(min(2_000_001, max(20_001, 20 * span / max(h, 1e-300))))
    stretches = line_intervals(mesh, point, a, domains)
    if not stretches:
        return np.array([lo, hi]), np.array([False]), []
    planes = np.array(list(snap) + _perpendicular_crossings(mesh, point, a))
    index = np.asarray(mesh.ngmesh.Elements3D().NumPy()['index'])
    in_region = None
    if domains is not None:
        in_region = np.zeros(int(index.max()) + 1, dtype=bool)
        in_region[[d for d in domains if 0 < d < len(in_region)]] = True
    breaks: List[float] = []
    inside: List[bool] = []
    for s0, s1, _, _ in stretches:
        # samples strictly inside the stretch
        d = min(1e-9 * span, 0.25 * (s1 - s0))
        n = max(3, int(n_samples * (s1 - s0) / span) + 1)
        found = _element_changes(mesh, point, a, np.linspace(s0 + d, s1 - d, n), tol * span)
        for plane in planes:
            found[np.abs(found - plane) <= snap_tol * span] = plane
        b = np.unique(np.concatenate([[s0], found[(found > s0) & (found < s1)], [s1]]))
        nr = _locate(mesh, point, a, 0.5 * (b[:-1] + b[1:]))
        ok = nr >= 0
        if in_region is not None:
            ok &= in_region[index[np.maximum(nr, 0)]]
        if breaks and b[0] > breaks[-1]:
            inside.append(False)              # a stretch outside the region
        elif breaks:
            b = b[1:]
        breaks.extend(b.tolist())
        inside.extend(ok.tolist())
    ends = [(f_in, f_out) for _, _, f_in, f_out in stretches]
    return np.array(breaks), np.array(inside, dtype=bool), ends


def gauss_points(breaks: np.ndarray, inside: np.ndarray, n_gauss: int):
    """Gauss points and weights on every piece of the line inside the mesh."""
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)
    a_, b_ = breaks[:-1][inside, None], breaks[1:][inside, None]
    s = (0.5 * (b_ - a_) * xi + 0.5 * (b_ + a_)).ravel()
    w = (0.5 * (b_ - a_) * wi).ravel()
    return s, w


def inside_intervals(breaks: np.ndarray, inside: np.ndarray) -> List[Tuple[float, float]]:
    """The connected stretches of the line inside the mesh (or domain)."""
    out = []
    start = None
    for k, ok in enumerate(inside):
        if ok and start is None:
            start = breaks[k]
        if not ok and start is not None:
            out.append((float(start), float(breaks[k])))
            start = None
    if start is not None:
        out.append((float(start), float(breaks[-1])))
    return out


def evaluation_matrix(fes, points: np.ndarray, a: int) -> sp.csc_matrix:
    """Sparse (ndof x n_points): column k evaluates the axis component at point k."""
    v = fes.TestFunction()
    ndof = fes.ndof
    rows, cols, vals = [], [], []
    for k, p in enumerate(points):
        lf = LinearForm(fes)
        lf += v[a](float(p[0]), float(p[1]), float(p[2]))
        lf.Assemble()
        col = lf.vec.FV().NumPy()
        nz = np.flatnonzero(col)
        rows.append(nz)
        cols.append(np.full(len(nz), k))
        vals.append(np.real(col[nz]))
    if not rows:
        return sp.csc_matrix((ndof, 0))
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(ndof, len(points)))


# =============================================================================
# The beam data of one system (the whole mesh, or one domain of it)
# =============================================================================

@dataclass
class PathData:
    """Quadrature of one voltage path in one system."""
    s: np.ndarray                  # Gauss points along the axis
    w: np.ndarray                  # weights
    P: sp.csc_matrix               # (ndof x n_points): E_a at the Gauss points
    k_over_omega: float            # k_b / omega = 1 / v_b
    intervals: List[Tuple[float, float]] = field(default_factory=list)

    def functional(self, omega: float) -> np.ndarray:
        """c(w)/i per DOF: v = c^T x with c = sum_k w_k exp(j k_b s_k) p_k."""
        return self.P @ (self.w * np.exp(1j * omega * self.k_over_omega * self.s))


@dataclass
class FaceData:
    """A port face of the system and what the beams do there."""
    port: str
    region: str                    # NGSolve region pattern of its faces
    columns: List[int]             # its columns in the system's B
    normal: np.ndarray             # outward unit normal (of the system)
    s_face: float                  # its main-axis coordinate (perpendicular faces)
    perpendicular: bool
    crossed: List[bool]            # per source: does the beam cross this face?
    sign: float = 1.0              # +1 if NGSolve's normal on the face points out of the system
    f_unit: List[Optional[np.ndarray]] = field(default_factory=list)   # per source
    b_reg: List[Optional[np.ndarray]] = field(default_factory=list)    # per source, B^T e_reg
    phi: List[Optional[object]] = field(default_factory=list)          # per source, Phi_reg


class BeamSystem:
    """Frequency-independent beam data of one system and its per-sample loads.

    ``fes`` is the system's H(curl) space (Dirichlet set = the PEC walls),
    ``B`` its port matrix, ``port_columns`` ``{port: [columns of B]}``,
    ``materials`` maps every mesh material to (eps_r, mu_r, sigma, tan_delta)
    and ``region_materials`` lists the system's materials (None: the whole
    mesh).
    """

    def __init__(self, fds, key: str, fes, B: np.ndarray, port_columns: Dict[str, List[int]],
                 setup: BeamSetup, materials: Dict[str, Tuple[float, float, float, float]],
                 region_materials: Optional[List[str]] = None):
        self.key = key
        self.fes = fes
        self.mesh = fes.mesh
        self.setup = setup
        self.a = axis_index(setup.axis)
        self.B = B
        self.complex = bool(fes.is_complex)
        self.walls = fds.bc
        self.region_materials = region_materials
        self.n_gauss = int(fds.order) + 2
        # the beam data of a reduced model (affine_data), set by a sweep that
        # keeps its field snapshots
        self.affine: Optional[Dict] = None

        mats = list(self.mesh.GetMaterials())
        if region_materials is None:
            self.system_mats = set(mats)
        else:
            self.system_mats = set(region_materials)
        # the system's domains (1-based)
        self.domains = [i + 1 for i, m in enumerate(mats) if m in self.system_mats]

        # the contrast load: materials of the system that are not vacuum
        self.contrast = {m: materials[m] for m in self.system_mats
                         if tuple(materials[m]) != (1.0, 1.0, 0.0, 0.0)}

        self.faces: List[FaceData] = [
            self._face_data(fds, port, cols) for port, cols in port_columns.items()]
        planes = [f.s_face for f in self.faces if f.perpendicular]

        self.paths: List[PathData] = []
        for line in setup.paths:
            breaks, inside, _ = line_pieces(self.mesh, line.point, self.a, self.domains,
                                            snap=planes)
            s, w = gauss_points(breaks, inside, self.n_gauss)
            pts = np.zeros((len(s), 3))
            for i in range(3):
                pts[:, i] = s if i == self.a else line.point[i]
            self.paths.append(PathData(s=s, w=w, P=evaluation_matrix(fes, pts, self.a),
                                       k_over_omega=1.0 / (line.beta * c0),
                                       intervals=inside_intervals(breaks, inside)))
            if line.current and len(s):
                self._check_line_medium(line, pts, materials)
        self._check_beams_leave_through_ports()

    # -- set-up ---------------------------------------------------------------
    def _element_material(self, nr: np.ndarray) -> List[str]:
        mats = list(self.mesh.GetMaterials())
        index = np.asarray(self.mesh.ngmesh.Elements3D().NumPy()['index']) - 1
        return [mats[index[n]] for n in nr]

    def _check_line_medium(self, line: BeamLine, pts: np.ndarray, materials) -> None:
        nr = self.mesh(pts[:, 0], pts[:, 1], pts[:, 2])['nr']
        for m in set(self._element_material(nr)):
            if tuple(materials[m]) != (1.0, 1.0, 0.0, 0.0):
                raise NotImplementedError(
                    f"Beam '{line.name}' runs through material '{m}' (eps_r, mu_r, sigma, "
                    f"tan_delta = {tuple(materials[m])}): only a beam in vacuum is "
                    "implemented (beta = 1 with the speed of light of the medium).")

    def _face_data(self, fds, port: str, cols: List[int]) -> FaceData:
        region = fds._region(port)
        mesh = self.mesh
        bnd = mesh.Boundaries(region)
        area = Integrate(CoefficientFunction(1.0), mesh, BND, definedon=bnd)
        nvec = np.array([Integrate(specialcf.normal(3)[i], mesh, BND, definedon=bnd)
                         for i in range(3)]) / area
        centre = np.array([Integrate(_COORDS[i], mesh, BND, definedon=bnd)
                           for i in range(3)]) / area
        unit = nvec / max(np.linalg.norm(nvec), 1e-300)
        perpendicular = abs(abs(unit[self.a]) - 1.0) < 1e-6
        sign = self._outward_sign(region)
        normal = unit * sign
        crossed = []
        for line in self.setup.sources:
            crossed.append(perpendicular and self._line_hits_face(line, region))
        face = FaceData(port=port, region=region, columns=list(cols), normal=normal,
                        s_face=float(centre[self.a]), perpendicular=perpendicular,
                        crossed=crossed, sign=sign)
        v = self.fes.TestFunction()
        for line, hit in zip(self.setup.sources, crossed):
            if not hit:
                face.f_unit.append(None)
                face.b_reg.append(None)
                face.phi.append(None)
                continue
            self._check_face_medium(fds, port, line)
            phi = _potential_reg(mesh, region, self.walls, line.point, self.a,
                                 int(fds.order) + 1)
            # f_p at unit phase: n_a v_b eps_b int grad(Phi_reg) . v dS
            lf = LinearForm(self.fes)
            lf += (float(normal[self.a]) * c0 * eps0) * InnerProduct(
                grad(phi), v.Trace()) * ds(region)
            with TaskManager():
                lf.Assemble()
            face.f_unit.append(lf.vec.FV().NumPy().copy())
            # B^T e_reg (unit phase): E_reg = -grad Phi_reg on the face
            g = GridFunction(self.fes)
            g.Set(-grad(phi), BND, definedon=bnd)
            face.b_reg.append(self.B.T @ g.vec.FV().NumPy())
            face.phi.append(phi)
        return face

    def _outward_sign(self, region: str) -> float:
        """+1 if NGSolve's normal on the faces of ``region`` points out of this
        system, else -1.

        Netgen orients a face from its ``domin`` to its ``domout`` domain (0:
        outside), and the boundary normal follows; the system is on the
        ``domin`` side when that domain is one of its materials.  (Locating a
        point just outside a curved mesh instead can crash NGSolve.)
        """
        names = set(region.replace('\\', '').split('|'))
        mats = list(self.mesh.GetMaterials())
        for fd in self.mesh.ngmesh.FaceDescriptors():
            if fd.bcname not in names:
                continue
            m_in = mats[fd.domin - 1] if fd.domin > 0 else None
            m_out = mats[fd.domout - 1] if fd.domout > 0 else None
            if m_in in self.system_mats:
                return 1.0
            if m_out in self.system_mats:
                return -1.0
        return 1.0

    def _line_hits_face(self, line: BeamLine, region: str) -> bool:
        """Does the beam's line cross the faces of ``region``?  (On the
        straight triangles: a face across the axis stays plane when curved.)"""
        names = set(region.replace('\\', '').split('|')) | {region}
        tris, _ = surface_triangles(self.mesh, names=names)
        s, _ = line_crossings(tris, line.point, self.a)
        return len(s) > 0

    def _check_face_medium(self, fds, port: str, line: BeamLine) -> None:
        eps = getattr(fds.port_solver, 'port_media_eps', {}) or {}
        mu = getattr(fds.port_solver, 'port_media_mu', {}) or {}
        if abs(eps.get(port, 1.0) - 1.0) > 1e-12 or abs(mu.get(port, 1.0) - 1.0) > 1e-12:
            raise NotImplementedError(
                f"Beam '{line.name}' crosses port '{port}', which is filled with a medium "
                f"(eps_r={eps.get(port, 1.0)}, mu_r={mu.get(port, 1.0)}): only vacuum is "
                "implemented.")

    def _check_beams_leave_through_ports(self) -> None:
        tol = 1e-6 * max(1.0, float(np.ptp(np.asarray(self.mesh.ngmesh.Coordinates())[:, self.a])))
        for j, (line, path) in enumerate(zip(self.setup.sources, self.paths)):
            ends = [s for iv in path.intervals for s in iv]
            faces = [f.s_face for f in self.faces if f.crossed[j]]
            for s_end in ends:
                if not any(abs(s_end - sf) <= tol for sf in faces):
                    raise ValueError(
                        f"Beam '{line.name}' leaves the {'model' if self.key == 'global' else 'domain ' + repr(self.key)} "
                        f"at {self.setup.axis.lower()} = {s_end:.6g} m through a wall: a beam must "
                        "enter and leave through port faces perpendicular to the main axis.")

    # -- per sample -----------------------------------------------------------
    @property
    def n_sources(self) -> int:
        return len(self.setup.sources)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def _phase_parts(self, k: float):
        """exp(-j k s) as a complex CF, or (cos, -sin) as two real CFs."""
        s = _COORDS[self.a]
        if self.complex:
            return exp(-1j * k * s)
        return cos(k * s), -sin(k * s)

    def lift_and_load(self, j: int, omega: float, k: Optional[float] = None,
                      crossed: bool = True):
        """For beam ``j`` at ``omega``: ``[(g, f)]`` -- one pair for a complex
        system, (Re, Im) for a real one.  ``g`` is the wall lift (an NGSolve
        vector), ``f`` the load from the port faces and the contrast regions
        (a numpy vector, before subtracting A g).

        ``k``: the wavenumber of the beam's phase exp(-j k s) (default
        omega / v_b); ``omega`` then sets only the factors omega, omega^2 of
        the loads.  ``crossed=False`` leaves out the load of the faces the
        beam crosses (the separable part, docs/theory/beam_reduction.md §10.4).
        """
        line = self.setup.sources[j]
        if k is None:
            k = omega / (line.beta * c0)
        prof = free_field_profile(line.point, self.a)
        hprof = free_h_profile(line.point, self.a)
        n = specialcf.normal(3)
        v = self.fes.TestFunction()
        walls = self.mesh.Boundaries(self.walls) if self.walls else None
        out = []
        # exp(-j k s): one complex CF, or its real and imaginary parts
        phase_list = [self._phase_parts(k)] if self.complex else list(self._phase_parts(k))
        for part, ph in enumerate(phase_list):
            g = GridFunction(self.fes)
            if walls is not None:
                g.Set(-prof * ph, BND, definedon=walls)
            f = np.zeros(self.fes.ndof, dtype=complex if self.complex else float)
            lf = None
            for face in self.faces:
                if face.f_unit[j] is not None:
                    if not crossed:
                        continue
                    fac = 1j * omega * np.exp(-1j * k * face.s_face)
                    if self.complex:
                        f += fac * face.f_unit[j]
                    else:
                        f += (fac.real if part == 0 else fac.imag) * face.f_unit[j]
                else:
                    # n x H_s = -n x H_free (n outward): load jw int (-n x H_free) . v dS
                    if lf is None:
                        lf = LinearForm(self.fes)
                    n_out = face.sign * n
                    if self.complex:
                        lf += (1j * omega) * InnerProduct(-Cross(n_out, hprof) * ph,
                                                          v.Trace()) * ds(face.region)
                    else:
                        # Re/Im of jw exp(-jks) = w (sin ks + j cos ks); ph is cos or -sin
                        other = phase_list[1] if part == 0 else phase_list[0]
                        coef = -other if part == 0 else other
                        lf += omega * InnerProduct(-Cross(n_out, hprof) * coef,
                                                   v.Trace()) * ds(face.region)
            for m, (eps_r, mu_r, sigma, tand) in self.contrast.items():
                if lf is None:
                    lf = LinearForm(self.fes)
                curl_e = (-1j * omega * mu0) * hprof        # curl E_free / exp(-jks)
                if self.complex:
                    vol = (1 / mu0 - 1 / (mu0 * mu_r)) * InnerProduct(curl_e * ph, _curl(v)) \
                        + (omega ** 2 * eps0 * (eps_r - 1) - 1j * omega ** 2 * eps0 * eps_r * tand
                           - 1j * omega * sigma) * InnerProduct(prof * ph, v)
                    lf += vol * dx(region_pattern([m]))
                else:
                    if sigma or tand:
                        raise RuntimeError("a lossy material makes the system complex")
                    # curl E_free = -j w mu0 h exp(-jks): Re/Im with ph = cos / -sin
                    other = phase_list[1] if part == 0 else phase_list[0]
                    curl_part = (omega * mu0) * hprof * (other if part == 0 else -other)
                    vol = (1 / mu0 - 1 / (mu0 * mu_r)) * InnerProduct(curl_part, _curl(v)) \
                        + omega ** 2 * eps0 * (eps_r - 1) * InnerProduct(prof * ph, v)
                    lf += vol * dx(region_pattern([m]))
            if lf is not None:
                with TaskManager():
                    lf.Assemble()
                f = f + lf.vec.FV().NumPy()
            out.append((g, f))
        return out

    def port_voltages(self, j: int, omega: float, e_s: np.ndarray) -> np.ndarray:
        """k_Z column of beam ``j``: B^T (e_s - sum_p exp(-j k s_p) e_reg,p) on the
        faces it crosses, B^T (e_s + E_free) on the others (i = 1 A)."""
        line = self.setup.sources[j]
        k = omega / (line.beta * c0)
        kz = self.B.T @ e_s
        other = [f for f in self.faces if f.b_reg[j] is None]
        for face in self.faces:
            if face.b_reg[j] is not None:
                kz = kz - np.exp(-1j * k * face.s_face) * face.b_reg[j]
        if other:
            prof = free_field_profile(line.point, self.a)
            region = "|".join(f.region for f in other)
            gfr = GridFunction(self.fes)
            if self.complex:
                gfr.Set(prof * exp(-1j * k * _COORDS[self.a]), BND,
                        definedon=self.mesh.Boundaries(region))
                kz = kz + self.B.T @ gfr.vec.FV().NumPy()
            else:
                c_, s_ = self._phase_parts(k)
                gfr.Set(prof * c_, BND, definedon=self.mesh.Boundaries(region))
                re = self.B.T @ gfr.vec.FV().NumPy()
                gfr.Set(prof * s_, BND, definedon=self.mesh.Boundaries(region))
                kz = kz + re + 1j * (self.B.T @ gfr.vec.FV().NumPy())
        return kz

    def voltages(self, omega: float, fields: np.ndarray) -> np.ndarray:
        """c_l(w)^T x for every path l and every column of ``fields``."""
        fields = np.asarray(fields)
        return np.array([p.functional(omega) @ fields if p.P.shape[1] else
                         np.zeros(fields.shape[1:], dtype=complex)
                         for p in self.paths])

    # -- frequency-separable form (reduced models) ----------------------------
    def affine_data(self, frequencies) -> Dict:
        """The beam data a reduced model needs, in a form evaluated without the
        mesh at any frequency of the band (docs/theory/beam_reduction.md §10.4).

        The wall lift g(w), the loads of the contrast regions and of the faces
        the beam does not cross, and their share of k_Z and z_oc carry the
        beam's phase over a region: they are evaluated at the Chebyshev points
        w_l of the band of ``frequencies`` [Hz] (widened by
        :data:`PHASE_BAND_MARGIN` on each side), each with the phase of the
        centre z_c taken out.  The loads are split into their factors of w and
        w^2.  Returns ``band`` [rad/s], ``zc``, ``nodes`` and per beam
        ``G`` (ndof x m, sparse), ``F1``, ``F2`` (sparse or None), ``Q``
        (port modes x m) and ``Pg`` (per path: Gauss points x m); ``free``
        marks the free unknowns.
        """
        w_lo, w_hi = phase_band(frequencies)
        s_all = np.asarray(self.mesh.ngmesh.Coordinates())[:, self.a]
        zc = 0.5 * (float(s_all.min()) + float(s_all.max()))
        length = float(s_all.max() - s_all.min())
        v_min = min(line.beta * c0 for line in self.setup.sources)
        m = chebyshev_count(0.25 * (w_hi - w_lo) * length / v_min)
        nodes = chebyshev_nodes(w_lo, w_hi, m)
        free = np.array([bool(v) for v in self.fes.FreeDofs()], dtype=bool)
        sources = []
        for j, line in enumerate(self.setup.sources):
            k_over_w = 1.0 / (line.beta * c0)
            G, F1, F2, Q = [], [], [], []
            Pg = [[] for _ in self.paths]
            has_load = False
            for wl in nodes:
                kl = wl * k_over_w
                shift = np.exp(1j * kl * zc)
                g, f1 = _combine(self.lift_and_load(j, 1.0, k=kl, crossed=False))
                if np.any(f1):
                    # loads w a1 + w^2 a2: from w = 1 and w = 2 at the same phase
                    _g, f2 = _combine(self.lift_and_load(j, 2.0, k=kl, crossed=False))
                    a2 = 0.5 * (f2 - 2.0 * f1)
                    a1 = f1 - a2
                    has_load = True
                else:
                    a1 = a2 = np.zeros_like(g)
                q = self.port_voltages(j, wl, g)
                for face in self.faces:
                    if face.b_reg[j] is not None:
                        q = q + np.exp(-1j * kl * face.s_face) * face.b_reg[j]
                G.append(shift * g)
                F1.append(shift * a1)
                F2.append(shift * a2)
                Q.append(shift * q)
                for i, p in enumerate(self.paths):
                    Pg[i].append(shift * (p.P.T @ g) if p.P.shape[1] else
                                 np.zeros(0, dtype=complex))
            sources.append({
                'G': _sparse_columns(G),
                'F1': _sparse_columns(F1) if has_load else None,
                'F2': _sparse_columns(F2) if has_load else None,
                'Q': np.array(Q).T,
                'Pg': [np.array(c).T if c and len(c[0]) else np.zeros((0, m), dtype=complex)
                       for c in Pg],
            })
        return {'band': [float(w_lo), float(w_hi)], 'zc': zc, 'length': length,
                'nodes': nodes, 'free': free, 'sources': sources}

    # -- fingerprints ---------------------------------------------------------
    def summary(self) -> Dict:
        """What the beam data were built from (stored with the results)."""
        return {
            "key": self.key,
            "axis": self.setup.axis,
            "n_gauss": self.n_gauss,
            "faces": [{"port": f.port, "crossed": list(map(bool, f.crossed)),
                       "s_face": f.s_face, "normal_axis": float(f.normal[self.a]),
                       "perpendicular": bool(f.perpendicular)} for f in self.faces],
            "paths": [{"intervals": p.intervals, "n_points": int(len(p.s))}
                      for p in self.paths],
            "contrast_materials": sorted(self.contrast),
        }


def _curl(v):
    from ngsolve import curl
    return curl(v)


def _potential_reg(mesh, region: str, walls: str, point, a: int, order: int):
    """Phi_reg on a port face (unit current, unit phase, beta = 1, vacuum):
    harmonic on the face, equal to -Phi_free = ln(rho)/(2 pi c0 eps0) on its rim."""
    fes_p = H1(mesh, order=order, dirichlet=walls, definedon=mesh.Boundaries(region))
    u_p, v_p = fes_p.TnT()
    am = BilinearForm(InnerProduct(grad(u_p).Trace(), grad(v_p).Trace()) * ds(region))
    with TaskManager():
        am.Assemble()
    _, rho2, _ = _geometry_cfs(point, a)
    phi = GridFunction(fes_p)
    phi.Set((1.0 / (4 * np.pi * c0 * eps0)) * log(rho2 + 1e-300), BND,
            definedon=mesh.Boundaries(region))
    with TaskManager():
        phi.vec.data -= am.mat.Inverse(fes_p.FreeDofs(), inverse='sparsecholesky') \
            * (am.mat * phi.vec)
    return phi


# =============================================================================
# Interpolation of the beam's phase in frequency (docs/theory/beam_reduction.md)
# =============================================================================

#: bound on the relative error of the phase interpolation (§10.4)
PHASE_INTERP_TOL = 1e-13

#: the interpolation band reaches this fraction of the solved band's width
#: beyond each of its ends, so a reduced model can be evaluated a little
#: outside its snapshots' band
PHASE_BAND_MARGIN = 0.1


def phase_band(frequencies) -> Tuple[float, float]:
    """The interpolation band [rad/s] of a sweep over ``frequencies`` [Hz]."""
    f = np.asarray(frequencies, dtype=float)
    lo, hi = float(f.min()), float(f.max())
    span = max(hi - lo, 1e-3 * hi)
    return 2 * np.pi * max(lo - PHASE_BAND_MARGIN * span, 0.0), \
        2 * np.pi * (hi + PHASE_BAND_MARGIN * span)


def chebyshev_count(c: float, tol: float = PHASE_INTERP_TOL) -> int:
    """Fewest Chebyshev points m (> c) that interpolate exp(-j c x) on [-1, 1]
    to ``tol``: the bound 4 sum_{n >= m} |J_n(c)| (§10.4)."""
    from scipy.special import jv
    m = max(3, int(np.ceil(c)) + 1)
    while 4.0 * np.sum(np.abs(jv(np.arange(m, m + 60), c))) > tol:
        m += 1
    return m


def chebyshev_nodes(w_lo: float, w_hi: float, m: int) -> np.ndarray:
    """The m Chebyshev points (second kind) of [w_lo, w_hi], ascending."""
    return 0.5 * (w_lo + w_hi) - 0.5 * (w_hi - w_lo) * np.cos(np.pi * np.arange(m) / (m - 1))


def lagrange_values(omega: float, nodes: np.ndarray) -> np.ndarray:
    """l_k(omega) of the Chebyshev points ``nodes`` (barycentric formula)."""
    m = len(nodes)
    wts = (-1.0) ** np.arange(m)
    wts[0] *= 0.5
    wts[-1] *= 0.5
    d = omega - nodes
    hit = np.flatnonzero(np.abs(d) <= 1e-14 * max(abs(omega), 1.0))
    if len(hit):
        out = np.zeros(m)
        out[hit[0]] = 1.0
        return out
    t = wts / d
    return t / t.sum()


def _combine(parts) -> Tuple[np.ndarray, np.ndarray]:
    """(g, f) of :meth:`BeamSystem.lift_and_load` as complex numpy vectors: one
    pair, or the (Re, Im) pairs of a real system."""
    g = np.asarray(parts[0][0].vec.FV().NumPy(), dtype=complex).copy()
    f = np.asarray(parts[0][1], dtype=complex).copy()
    if len(parts) > 1:
        g = g + 1j * np.asarray(parts[1][0].vec.FV().NumPy())
        f = f + 1j * np.asarray(parts[1][1])
    return g, f


def _sparse_columns(cols, rel_tol: float = 1e-15) -> sp.csc_matrix:
    """The vectors ``cols`` as the columns of a sparse matrix (entries below
    ``rel_tol`` times the largest dropped: a lift lives on the walls only)."""
    X = np.array(cols).T
    scale = np.abs(X).max() if X.size else 0.0
    X[np.abs(X) <= rel_tol * scale] = 0.0
    return sp.csc_matrix(X)


# =============================================================================
# Generalised matrices
# =============================================================================

def z_tilde(Z: np.ndarray, kZ: np.ndarray, hZ: np.ndarray, zoc: np.ndarray) -> np.ndarray:
    """[[Z, k_Z], [h_Z, z_oc]] per frequency."""
    n_f, P = Z.shape[0], Z.shape[1]
    L, S = zoc.shape[1], zoc.shape[2]
    out = np.zeros((n_f, P + L, P + S), dtype=complex)
    out[:, :P, :P] = Z
    out[:, :P, P:] = kZ
    out[:, P:, :P] = hZ
    out[:, P:, P:] = zoc
    return out


def s_tilde(Z: np.ndarray, kZ: np.ndarray, hZ: np.ndarray, zoc: np.ndarray,
            Zref: np.ndarray) -> np.ndarray:
    """[[S, k], [h, z_b]] per frequency (docs/theory/beam.md §9.8).

    ``Zref`` (n_f, P, P) is the diagonal reference impedance of the port modes:
    S = Zref^1/2 (Z + Zref)^-1 (Z - Zref) Zref^-1/2,  k = Zref^1/2 (Z + Zref)^-1 k_Z,
    h = 2 h_Z (Z + Zref)^-1 Zref^1/2,  z_b = z_oc - h_Z (Z + Zref)^-1 k_Z.
    """
    n_f, P = Z.shape[0], Z.shape[1]
    L, S = zoc.shape[1], zoc.shape[2]
    out = np.zeros((n_f, P + L, P + S), dtype=complex)
    for i in range(n_f):
        z0 = np.diag(Zref[i]).astype(complex)
        sq = np.diag(np.sqrt(z0))
        isq = np.diag(1.0 / np.sqrt(z0))
        Zd = Z[i] + np.diag(z0)
        try:
            inv = np.linalg.inv(Zd)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(Zd)
        out[i, :P, :P] = isq @ (Z[i] - np.diag(z0)) @ inv @ sq
        out[i, :P, P:] = sq @ inv @ kZ[i]
        out[i, P:, :P] = 2 * hZ[i] @ inv @ sq
        out[i, P:, P:] = zoc[i] - hZ[i] @ inv @ kZ[i]
    return out


def matrix_labels(port_labels: List[str], setup: BeamSetup) -> Tuple[List[str], List[str]]:
    """(rows, columns) of the generalised matrices: the port modes, then the
    paths (rows) / beams (columns)."""
    return list(port_labels) + setup.path_labels, list(port_labels) + setup.source_labels


def labelled_dict(matrix: np.ndarray, rows: List[str], cols: List[str],
                  frequencies: np.ndarray) -> Dict[str, np.ndarray]:
    """Excitation-first keys like the S_dict: '<column><row>'."""
    d = {'frequencies': frequencies}
    for r, rl in enumerate(rows):
        for c, cl in enumerate(cols):
            d[f"{cl}{rl}"] = matrix[:, r, c]
    return d


# =============================================================================
# Stored results
# =============================================================================

def save_tilde(path, tilde: Dict, which: str) -> None:
    """Write one generalised matrix (``which``: 'Z' or 'S') with its labels."""
    import h5py
    from cavsim3d.core.persistence import H5Serializer
    data = tilde.get(f'{which}_tilde')
    if data is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=H5Serializer.to_complex_h5(np.asarray(data)))
        f.create_dataset("frequencies", data=np.asarray(tilde['frequencies']))
        f.attrs["rows"] = json.dumps(list(tilde['rows']))
        f.attrs["cols"] = json.dumps(list(tilde['cols']))
        f.attrs["names"] = json.dumps(tilde.get('names', {}))
        f.attrs["setup"] = json.dumps(tilde.get('setup', {}))
        f.attrs["fingerprint"] = str(tilde.get('fingerprint', ''))
        f.attrs["summary"] = json.dumps(tilde.get('summary') or {}, default=float)
        f.attrs["version"] = BEAM_DATA_VERSION
        # what joining this part to others needs: the (port, mode) of every
        # port row, their reference impedances, the port positions and the
        # identity of each mode
        if tilde.get('port_modes') is not None:
            f.attrs["port_modes"] = json.dumps([[str(p), int(m)] for p, m in tilde['port_modes']])
        if tilde.get('ports') is not None:
            f.attrs["ports"] = json.dumps(tilde['ports'], default=float)
        if tilde.get('fingerprints') is not None:
            f.attrs["mode_fingerprints"] = json.dumps(tilde['fingerprints'], default=float)
        if which == 'S' and tilde.get('zref') is not None:
            f.create_dataset("zref", data=H5Serializer.to_complex_h5(np.asarray(tilde['zref'])))


def load_tilde(path) -> Optional[Dict]:
    """Read a generalised matrix written by :func:`save_tilde` (None if absent)."""
    import h5py
    from cavsim3d.core.persistence import H5Serializer
    if path is None or not path.exists():
        return None
    with h5py.File(path, "r") as f:
        out = {
            'data': H5Serializer.load_dataset(f["data"]),
            'frequencies': f["frequencies"][()],
            'rows': json.loads(f.attrs["rows"]),
            'cols': json.loads(f.attrs["cols"]),
            'names': json.loads(f.attrs.get("names", "{}")),
            'setup': json.loads(f.attrs.get("setup", "{}")),
            'fingerprint': str(f.attrs.get("fingerprint", "")),
            'summary': json.loads(f.attrs.get("summary", "{}")),
            'port_modes': ([(p, int(m)) for p, m in json.loads(f.attrs["port_modes"])]
                           if "port_modes" in f.attrs else None),
            'ports': json.loads(f.attrs["ports"]) if "ports" in f.attrs else None,
            'fingerprints': (json.loads(f.attrs["mode_fingerprints"])
                             if "mode_fingerprints" in f.attrs else None),
            'zref': H5Serializer.load_dataset(f["zref"]) if "zref" in f else None,
        }
    return out


def save_beam_data(path, system: "BeamSystem") -> None:
    """Frequency-independent beam data of one system (``matrices/beam_<section>.h5``):
    the Gauss points, weights and evaluation matrices of the paths, and per port
    face and beam the unit-phase load f_p and B^T e_reg."""
    import h5py
    from cavsim3d.core.persistence import H5Serializer
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["setup"] = json.dumps(system.setup.to_dict())
        f.attrs["fingerprint"] = system.setup.fingerprint()
        f.attrs["summary"] = json.dumps(system.summary(), default=float)
        f.attrs["version"] = BEAM_DATA_VERSION
        for i, p in enumerate(system.paths):
            g = f.create_group(f"path_{i + 1}")
            g.create_dataset("s", data=p.s)
            g.create_dataset("w", data=p.w)
            H5Serializer.save_sparse_csr(g, "P", p.P.tocsr())
        f.attrs["faces"] = json.dumps([face.port for face in system.faces])
        for face in system.faces:
            g = f.create_group(f"face_{face.port}")
            g.attrs["s_face"] = face.s_face
            g.create_dataset("normal", data=face.normal)
            g.create_dataset("crossed", data=np.asarray(face.crossed, dtype=bool))
            for j, (fu, br) in enumerate(zip(face.f_unit, face.b_reg)):
                if fu is not None:
                    g.create_dataset(f"f_unit_{j + 1}", data=np.real(fu))
                    g.create_dataset(f"b_reg_{j + 1}", data=np.real(br))
        aff = system.affine
        if aff is not None:
            g = f.create_group("affine")
            g.create_dataset("band", data=np.asarray(aff['band']))
            g.attrs["zc"] = float(aff['zc'])
            g.attrs["length"] = float(aff['length'])
            g.create_dataset("nodes", data=np.asarray(aff['nodes']))
            g.create_dataset("free", data=np.asarray(aff['free'], dtype=bool))
            for j, src in enumerate(aff['sources']):
                gs = g.create_group(f"source_{j + 1}")
                for name in ("G", "F1", "F2"):
                    if src[name] is not None:
                        H5Serializer.save_sparse_csr(gs, name, src[name].tocsr())
                H5Serializer.save_dataset(gs, "Q", np.asarray(src['Q']))
                for i, pg in enumerate(src['Pg']):
                    H5Serializer.save_dataset(gs, f"Pg_{i + 1}", np.asarray(pg))


def beam_data_of(system: "BeamSystem") -> Dict:
    """The data of a live system in the form of :func:`load_beam_data`."""
    return {
        'setup': system.setup.to_dict(), 'fingerprint': system.setup.fingerprint(),
        'version': BEAM_DATA_VERSION,
        'paths': [{'s': p.s, 'w': p.w, 'P': p.P} for p in system.paths],
        'faces': [{'port': face.port, 's_face': face.s_face, 'crossed': list(face.crossed),
                   'f_unit': [None if v is None else np.real(v) for v in face.f_unit],
                   'b_reg': [None if v is None else np.real(v) for v in face.b_reg]}
                  for face in system.faces],
        'affine': system.affine,
    }


def load_beam_data(path) -> Optional[Dict]:
    """Read :func:`save_beam_data` (None if absent): ``setup``, ``fingerprint``,
    ``paths`` (s, w, P), ``faces`` (s_face, crossed, f_unit, b_reg per beam)
    and ``affine`` (None if the sweep kept no field snapshots)."""
    import h5py
    from cavsim3d.core.persistence import H5Serializer
    if path is None or not path.exists():
        return None
    with h5py.File(path, "r") as f:
        setup = json.loads(f.attrs.get("setup", "{}"))
        n_src = len(BeamSetup.from_dict(setup).sources)
        out = {'setup': setup, 'fingerprint': str(f.attrs.get("fingerprint", "")),
               'version': int(f.attrs.get("version", 0)), 'paths': [], 'faces': [],
               'affine': None}
        i = 1
        while f"path_{i}" in f:
            g = f[f"path_{i}"]
            out['paths'].append({'s': g["s"][()], 'w': g["w"][()],
                                 'P': H5Serializer.load_sparse_csr(g["P"]).tocsc()})
            i += 1
        ports = (json.loads(f.attrs["faces"]) if "faces" in f.attrs else
                 [k[len("face_"):] for k in f.keys() if k.startswith("face_")])
        for port in ports:
            g = f[f"face_{port}"]
            out['faces'].append({
                'port': port, 's_face': float(g.attrs["s_face"]),
                'crossed': [bool(v) for v in g["crossed"][()]],
                'f_unit': [g[f"f_unit_{j + 1}"][()] if f"f_unit_{j + 1}" in g else None
                           for j in range(n_src)],
                'b_reg': [g[f"b_reg_{j + 1}"][()] if f"b_reg_{j + 1}" in g else None
                          for j in range(n_src)]})
        if "affine" in f:
            g = f["affine"]
            aff = {'band': list(g["band"][()]), 'zc': float(g.attrs["zc"]),
                   'length': float(g.attrs.get("length", 0.0)),
                   'nodes': g["nodes"][()], 'free': g["free"][()].astype(bool),
                   'sources': []}
            for j in range(n_src):
                gs = g[f"source_{j + 1}"]
                src = {name: (H5Serializer.load_sparse_csr(gs[name]).tocsc()
                              if name in gs else None) for name in ("G", "F1", "F2")}
                src['Q'] = H5Serializer.load_dataset(gs["Q"])
                src['Pg'] = [H5Serializer.load_dataset(gs[f"Pg_{i + 1}"])
                             for i in range(len(out['paths']))]
                aff['sources'].append(src)
            out['affine'] = aff
    return out


def port_fields(system: "BeamSystem") -> Dict:
    """Electrostatic port solutions Phi_reg of one system: {port: {beam: data}}."""
    out: Dict[str, Dict[str, Dict]] = {}
    for face in system.faces:
        for line, phi in zip(system.setup.sources, face.phi):
            if phi is not None:
                out.setdefault(face.port, {})[line.name] = {
                    "order": int(phi.space.globalorder),
                    "vec": np.asarray(phi.vec.FV().NumPy()).copy()}
    return out


# =============================================================================
# Accessors of a result with a beam
# =============================================================================

class _MatrixView:
    """Minimal PlotMixin host for a labelled dict (plots of S~ and Z~)."""

    def __init__(self, frequencies, s_dict=None, z_dict=None):
        self.frequencies = frequencies
        self.S_dict = s_dict
        self.Z_dict = z_dict

    def _get_freq_ghz(self):
        return self.frequencies

    @staticmethod
    def _ensure_ax(ax=None, figsize=(10, 6)):
        from cavsim3d.utils.plot_mixin import PlotMixin
        return PlotMixin._ensure_ax(ax, figsize)

    @staticmethod
    def _merge_style(defaults, user_kwargs):
        from cavsim3d.utils.plot_mixin import PlotMixin
        return PlotMixin._merge_style(defaults, user_kwargs)

    @staticmethod
    def _apply_data(ax, freq_ghz, data, plot_type, label, style_kwargs):
        from cavsim3d.utils.plot_mixin import PlotMixin
        return PlotMixin._apply_data(ax, freq_ghz, data, plot_type, label, style_kwargs)

    _DEFAULT_PLOT_STYLE: Dict = {}


class BeamResultMixin:
    """Generalised matrices of a result solved with a beam.

    The host sets ``self._beam`` to a dict with ``Z_tilde`` and/or
    ``S_tilde`` (n_freqs, rows, cols), ``rows``, ``cols``, ``frequencies``
    and ``names`` (label -> beam/path name).
    """

    _beam: Optional[Dict] = None

    @property
    def has_beam(self) -> bool:
        """True if this result carries beam rows and columns."""
        return bool(getattr(self, '_beam', None))

    def _beam_data(self) -> Dict:
        b = getattr(self, '_beam', None)
        if not b:
            raise RuntimeError("No beam results here: define a beam with "
                               "proj.add_beam(...) and solve.")
        return b

    def _tilde(self, which: str) -> np.ndarray:
        m = self._beam_data().get(f'{which}_tilde')
        if m is None:
            raise RuntimeError(f"{which}~ is not available for this result.")
        return m

    @property
    def s_tilde(self) -> np.ndarray:
        """S~ = [[S, k], [h, z_b]], shape (n_freqs, port modes + paths,
        port modes + beams): waves and beam voltages out per wave and beam
        current in (docs/theory/beam.md §9.8)."""
        return self._tilde('S')

    @property
    def z_tilde(self) -> np.ndarray:
        """Z~ = [[Z, k_Z], [h_Z, z_oc]]: port and beam voltages per port and beam
        current, every port mode open (docs/theory/beam.md §9.7)."""
        return self._tilde('Z')

    @property
    def tilde_labels(self) -> Tuple[List[str], List[str]]:
        """(rows, columns) of S~ and Z~: '1(1)', ... then 'b(1)', ..."""
        b = self._beam_data()
        return list(b['rows']), list(b['cols'])

    @property
    def beam_names(self) -> Dict[str, str]:
        """Beam/path label -> name, e.g. {'b(1)': 'beam'}."""
        return dict(self._beam_data().get('names', {}))

    def _labelled(self, which: str) -> Dict[str, np.ndarray]:
        b = self._beam_data()
        return labelled_dict(self._tilde(which), b['rows'], b['cols'], b['frequencies'])

    @property
    def s_tilde_dict(self) -> Dict[str, np.ndarray]:
        """S~ keyed like S_dict, excitation first: 'b(1)b(1)' is z_b,
        'b(1)2(1)' is k (beam -> port 2 mode 1), '1(1)b(1)' is h."""
        return self._labelled('S')

    @property
    def z_tilde_dict(self) -> Dict[str, np.ndarray]:
        """Z~ keyed like Z_dict ('b(1)b(1)' is z_oc)."""
        return self._labelled('Z')

    def _beam_label(self, name, sources: bool) -> str:
        b = self._beam_data()
        labels = ([c for c in b['cols'] if c.startswith('b(')] if sources
                  else [r for r in b['rows'] if r.startswith('b(')])
        if name is None:
            return labels[0]
        if name in labels:
            return name
        for lab in labels:
            if b.get('names', {}).get(lab) == name:
                return lab
        kind = "beam" if sources else "path"
        known = [f"{lab} ({b.get('names', {}).get(lab, '?')})" for lab in labels]
        raise KeyError(f"no {kind} {name!r}; known: {', '.join(known)}")

    def beam_impedance(self, beam=None, path=None, ports: str = 'matched') -> np.ndarray:
        """Longitudinal impedance Z_par = -v/i [Ohm] per frequency.

        ``beam``: name or label of the beam (default: the first); ``path``:
        the voltage path (default: the beam's own line); ``ports``:
        ``'matched'`` (z_b: every port mode terminated in its reference
        impedance) or ``'open'`` (z_oc: the port modes see magnetic walls).
        """
        if ports not in ('matched', 'open'):
            raise ValueError(f"ports must be 'matched' or 'open', got {ports!r}")
        col = self._beam_label(beam, sources=True)
        row = col if path is None else self._beam_label(path, sources=False)
        b = self._beam_data()
        m = self._tilde('S' if ports == 'matched' else 'Z')
        return -m[:, b['rows'].index(row), b['cols'].index(col)]

    def plot_s_tilde(self, params=None, plot_type: str = 'db', **kwargs):
        """Plot entries of S~ (keys of :attr:`s_tilde_dict`); arguments as plot_s."""
        from cavsim3d.utils.plot_mixin import PlotMixin
        view = _MatrixView(self._beam_data()['frequencies'], s_dict=self.s_tilde_dict)
        return PlotMixin.plot_s(view, params=params, plot_type=plot_type, **kwargs)

    def plot_z_tilde(self, params=None, plot_type: str = 'db', **kwargs):
        """Plot entries of Z~ (keys of :attr:`z_tilde_dict`); arguments as plot_z."""
        from cavsim3d.utils.plot_mixin import PlotMixin
        view = _MatrixView(self._beam_data()['frequencies'], z_dict=self.z_tilde_dict)
        return PlotMixin.plot_z(view, params=params, plot_type=plot_type, **kwargs)

    def plot_beam_impedance(self, beam=None, path=None, ports: str = 'matched', ax=None,
                            label: Optional[str] = None, **kwargs):
        """Re and Im of :meth:`beam_impedance` against frequency; returns (fig, ax)."""
        import matplotlib.pyplot as plt
        zpar = self.beam_impedance(beam, path, ports)
        f_ghz = np.asarray(self._beam_data()['frequencies']) / 1e9
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.5))
        else:
            fig = ax.get_figure()
        base = f"{label} " if label else ""
        line, = ax.plot(f_ghz, zpar.real, label=base + r"Re $Z_\parallel$", **kwargs)
        style = dict(kwargs)
        style.setdefault('color', line.get_color())
        style['linestyle'] = '--'
        ax.plot(f_ghz, zpar.imag, label=base + r"Im $Z_\parallel$", **style)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(r'$Z_\parallel$ ($\Omega$)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig, ax


# =============================================================================
# Joining segments through their generalised scattering matrices (§9.9)
# =============================================================================

def join_s_tilde(blocks: List[np.ndarray], modes: List[List[int]],
                 pairs: List[Tuple[int, int]], external: List[int],
                 col_phase: Optional[List[np.ndarray]] = None,
                 row_phase: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """S~ of segments joined at their cuts (docs/theory/beam.md §9.9).

    ``blocks[d]``: S~ of segment d, (n_f, P_d + L, P_d + S); ``modes[d]``:
    the global index of each of its P_d port modes; ``pairs``: the global
    port-mode indices joined at a cut (the wave leaving one face enters the
    other); ``external``: the global indices kept, in their order.

    A segment solved with the beams' global phase (one coordinate for the
    whole model) sees the same beam current as the others, and the beam
    voltages of the segments add: d = (1, 1, ...) in §9.9.  A segment solved
    in its own frame, shifted by z_d along the axis, gets the phase of its
    position: ``col_phase[d]`` (n_f, S) multiplies its beam columns
    (exp(-j k_b z_d)) and ``row_phase[d]`` (n_f, L) its path rows
    (exp(+j k_b z_d)).
    Returns S~ of the joined model, (n_f, len(external) + L, len(external) + S).
    """
    D = len(blocks)
    n_f = blocks[0].shape[0]
    P = [len(m) for m in modes]
    L = blocks[0].shape[1] - P[0]
    S = blocks[0].shape[2] - P[0]
    N = sum(P)
    pos = {}
    for d, m in enumerate(modes):
        off = sum(P[:d])
        for i, g in enumerate(m):
            pos[g] = off + i
    internal = [g for pair in pairs for g in pair]
    i_int = [pos[g] for g in internal]
    rows_rest = [pos[g] for g in external] + list(range(N, N + D * L))
    cols_rest = [pos[g] for g in external] + list(range(N, N + D * S))
    F = np.zeros((len(internal), len(internal)))
    where = {g: i for i, g in enumerate(internal)}
    for a, b in pairs:
        F[where[a], where[b]] = F[where[b], where[a]] = 1.0
    E = len(external)
    T_cols = np.zeros((E + D * S, E + S))
    T_cols[:E, :E] = np.eye(E)
    T_rows = np.zeros((E + L, E + D * L))
    T_rows[:E, :E] = np.eye(E)
    for d in range(D):
        T_cols[E + d * S:E + (d + 1) * S, E:] = np.eye(S)
        T_rows[E:, E + d * L:E + (d + 1) * L] = np.eye(L)

    out = np.zeros((n_f, E + L, E + S), dtype=complex)
    for k in range(n_f):
        SR = np.zeros((N + D * L, N + D * S), dtype=complex)
        for d in range(D):
            blk = np.array(blocks[d][k], dtype=complex)
            if col_phase is not None and col_phase[d] is not None:
                blk[:, P[d]:] *= col_phase[d][k][None, :]
            if row_phase is not None and row_phase[d] is not None:
                blk[P[d]:, :] *= row_phase[d][k][:, None]
            p = [pos[g] for g in modes[d]]
            SR[np.ix_(p, p)] = blk[:P[d], :P[d]]
            SR[np.ix_(p, range(N + d * S, N + (d + 1) * S))] = blk[:P[d], P[d]:]
            SR[np.ix_(range(N + d * L, N + (d + 1) * L), p)] = blk[P[d]:, :P[d]]
            SR[N + d * L:N + (d + 1) * L, N + d * S:N + (d + 1) * S] = blk[P[d]:, P[d]:]
        G11 = SR[np.ix_(i_int, i_int)]
        G12 = SR[np.ix_(i_int, cols_rest)]
        G21 = SR[np.ix_(rows_rest, i_int)]
        G22 = SR[np.ix_(rows_rest, cols_rest)]
        J = G22 + G21 @ np.linalg.solve(F - G11, G12) if len(i_int) else G22
        out[k] = T_rows @ J @ T_cols
    return out


def z_tilde_from_s_tilde(St: np.ndarray, Zref: np.ndarray) -> np.ndarray:
    """Z~ from S~ (inverse of :func:`s_tilde`): with A = Z + Zref,
    Z from S as for the ports, k_Z = A Zref^-1/2 k, h_Z = h Zref^-1/2 A / 2,
    z_oc = z_b + h Zref^-1/2 A Zref^-1/2 k / 2."""
    from cavsim3d.solvers.base import ParameterConverter
    n_f = St.shape[0]
    P = Zref.shape[1]
    out = np.zeros_like(St, dtype=complex)
    for i in range(n_f):
        z0 = np.diag(Zref[i]).astype(complex)
        isq = np.diag(1.0 / np.sqrt(z0))
        Z = ParameterConverter.s_to_z(St[i, :P, :P], np.diag(z0))
        A = Z + np.diag(z0)
        k, h, zb = St[i, :P, P:], St[i, P:, :P], St[i, P:, P:]
        out[i, :P, :P] = Z
        out[i, :P, P:] = A @ isq @ k
        out[i, P:, :P] = 0.5 * h @ isq @ A
        out[i, P:, P:] = zb + 0.5 * h @ isq @ A @ isq @ k
    return out
