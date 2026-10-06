"""Geometric queries on a mesh's surface triangles, without NGSolve's point search.

NGSolve's point location (``mesh(x, y, z)``) can crash for a point just outside
a curved mesh; these helpers work on the straight surface triangles instead
(quadrilaterals are split in two), and :func:`line_intervals` moves a line's
crossings onto the curved faces with the element maps, so only points known
to be inside the mesh are ever located.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _surface_selection(mesh, keep: Sequence[int]):
    """Triangles (quadrilaterals split) of the surface elements whose face
    descriptor (1-based) is in ``keep``: vertex coordinates ``(n, 3, 3)``,
    face descriptor, surface element number and whether it came from a
    quadrilateral."""
    ng = mesh.ngmesh
    els = ng.Elements2D().NumPy()
    sel = np.isin(els['index'], list(keep))
    if not np.any(sel):
        return (np.zeros((0, 3, 3)), np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0, dtype=bool))
    coords = np.asarray(ng.Coordinates())
    nodes = els['nodes'][sel].astype(np.int64) - 1          # 1-based point numbers
    npts = els['np'][sel]
    elem = np.nonzero(sel)[0]
    quad = (npts == 4) | (npts == 8)
    tris = [nodes[:, [0, 1, 2]]]
    index = [els['index'][sel]]
    elems = [elem]
    if np.any(quad):
        tris.append(nodes[quad][:, [0, 2, 3]])
        index.append(els['index'][sel][quad])
        elems.append(elem[quad])
    tri_nodes = np.concatenate(tris)
    is_quad = np.concatenate([quad] + ([quad[quad]] if np.any(quad) else []))
    return coords[tri_nodes], np.concatenate(index), np.concatenate(elems), is_quad


def surface_triangles(mesh, names: Optional[Iterable[str]] = None,
                      outer_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex coordinates ``(n, 3, 3)`` of the surface triangles, and the face
    descriptor (1-based) of each.

    ``names``: only faces with these boundary names; ``outer_only``: only faces
    on the outside of the mesh (not those between two domains).
    """
    names = set(names) if names is not None else None
    keep = []
    for i, fd in enumerate(mesh.ngmesh.FaceDescriptors(), start=1):
        if names is not None and fd.bcname not in names:
            continue
        if outer_only and fd.domin > 0 and fd.domout > 0:
            continue
        keep.append(i)
    tris, index, _, _ = _surface_selection(mesh, keep)
    return tris, index


def _barycentric_2d(tri2: np.ndarray, q: np.ndarray):
    """Barycentric coordinates ``(l1, l2)`` (of vertices 1 and 2) of the 2D point
    ``q`` in every triangle ``tri2`` ``(n, 3, 2)``, and twice the signed area."""
    a, b, c = tri2[:, 0], tri2[:, 1], tri2[:, 2]
    v0, v1, v2 = b - a, c - a, q[None, :] - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        l1 = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / det
        l2 = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / det
    return l1, l2, det


def face_contains_point(tris: np.ndarray, point, t1, t2, tol: float = 1e-9) -> bool:
    """True if ``point`` (on the plane of the face spanned by ``t1``, ``t2``)
    lies on one of the face's triangles ``tris`` ``(n, 3, 3)``."""
    if not len(tris):
        return False
    basis = np.stack([np.asarray(t1, float), np.asarray(t2, float)], axis=1)   # (3, 2)
    p0 = np.asarray(point, float)
    tri2 = (tris - p0) @ basis
    l1, l2, det = _barycentric_2d(tri2, np.zeros(2))
    ok = np.abs(det) > 1e-30 * (np.abs(tri2).max() ** 2 + 1e-300)
    inside = ok & (l1 >= -tol) & (l2 >= -tol) & (l1 + l2 <= 1 + tol)
    return bool(np.any(inside))


def _crossings(tris: np.ndarray, point, axis: int, tol: float):
    """Crossings of the axis-parallel line with ``tris``: axis coordinates,
    triangle indices and the barycentric coordinates ``(l1, l2)``."""
    if not len(tris):
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0)
    tr = [i for i in range(3) if i != axis]
    q = np.asarray(point, float)[tr]
    tri2 = tris[:, :, tr]
    l1, l2, det = _barycentric_2d(tri2, q)
    scale = np.abs(tri2 - q).max() ** 2 + 1e-300
    ok = np.abs(det) > 1e-12 * scale
    inside = ok & (l1 >= -tol) & (l2 >= -tol) & (l1 + l2 <= 1 + tol)
    idx = np.nonzero(inside)[0]
    l1, l2 = l1[idx], l2[idx]
    s = ((1 - l1 - l2) * tris[idx, 0, axis] + l1 * tris[idx, 1, axis]
         + l2 * tris[idx, 2, axis])
    return s, idx, l1, l2


def line_crossings(tris: np.ndarray, point, axis: int, tol: float = 1e-9):
    """Where the line through ``point`` parallel to coordinate ``axis`` crosses
    the triangles ``tris`` ``(n, 3, 3)``: the axis coordinates of the
    crossings and the index of each crossed triangle.  Triangles parallel to
    the line are skipped."""
    s, idx, _, _ = _crossings(tris, point, axis, tol)
    return s, idx


def _curved_crossing(mesh, elem: int, quad: bool, q: np.ndarray, tr: List[int],
                     axis: int, start, length: float) -> Optional[float]:
    """Axis coordinate where the line through the transverse point ``q``
    crosses the curved surface element ``elem`` (Newton on its map), or None
    if it does not cross it."""
    from ngsolve import BND, ElementId
    trafo = mesh.GetTrafo(ElementId(BND, int(elem)))
    xi = np.array(start, dtype=float)
    for _ in range(30):
        mip = trafo(float(xi[0]), float(xi[1]))
        p = np.array(mip.point)
        r = p[tr] - q
        if np.max(np.abs(r)) <= 1e-14 * length:
            break
        J = np.array(mip.jacobi)[tr, :]
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return None
        xi = xi + step
        if np.max(np.abs(xi)) > 4.0:            # far outside the element
            return None
    else:
        return None
    t = 1e-8
    if quad:
        inside = bool(np.all(xi >= -t) and np.all(xi <= 1 + t))
    else:
        inside = bool(np.all(xi >= -t) and xi.sum() <= 1 + t)
    return float(p[axis]) if inside else None


def line_intervals(mesh, point, axis: int, domains: Optional[Iterable[int]] = None,
                   tol: float = 1e-9) -> List[Tuple[float, float, str, str]]:
    """The stretches of the line through ``point`` parallel to coordinate
    ``axis`` that lie inside the mesh, or inside its ``domains`` (1-based
    domain numbers): ``[(s_in, s_out, face_in, face_out), ...]``, the axis
    coordinates where the line enters and leaves and the boundary names there.

    Found from the line's crossings with the boundary of the region (the
    straight triangles, the crossing moved onto the curved face when the mesh
    is curved), without locating any point.  Netgen orients a face's
    triangles from its ``domin`` to its ``domout`` domain, which tells whether
    the line enters or leaves the region there.
    """
    ng = mesh.ngmesh
    fds = list(ng.FaceDescriptors())
    ndom = max([max(fd.domin, fd.domout) for fd in fds] + [0])
    in_region = np.zeros(ndom + 1, dtype=bool)
    if domains is None:
        in_region[1:] = True
    else:
        for d in domains:
            if 0 < int(d) <= ndom:
                in_region[int(d)] = True
    keep, out_sign, names = [], {}, {}
    for i, fd in enumerate(fds, start=1):
        a_in, a_out = bool(in_region[fd.domin]), bool(in_region[fd.domout])
        if a_in != a_out:
            keep.append(i)
            out_sign[i] = 1.0 if a_in else -1.0     # +1: the normal points out of the region
            names[i] = fd.bcname
    tris, index, elems, is_quad = _surface_selection(mesh, keep)
    s, idx, l1, l2 = _crossings(tris, point, axis, tol)
    if not len(s):
        return []
    coords = np.asarray(ng.Coordinates())
    length = float(np.ptp(coords, axis=0).max()) or 1.0
    t = tris[idx]
    normal = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    kind = np.sign(normal[:, axis] * np.array([out_sign[f] for f in index[idx]]))  # +1 leaves

    # one crossing per cluster (a line through a shared edge or vertex hits
    # several triangles); a cluster whose kinds cancel only touches the boundary
    order = np.argsort(s)
    clusters: List[List[int]] = []
    for k in order:
        if clusters and s[k] - s[clusters[-1][0]] <= 1e-9 * length:
            clusters[-1].append(k)
        else:
            clusters.append([k])

    curved = mesh.GetCurveOrder() > 1
    tr = [i for i in range(3) if i != axis]
    q = np.asarray(point, float)[tr]
    events = []
    for cl in clusters:
        net = float(np.sum(kind[cl]))
        if net == 0:
            continue
        k0 = cl[0]
        s_k = float(s[k0])
        if curved:
            s_k = _refine_on_curved(mesh, tris, elems, is_quad, idx, l1, l2, cl, q, tr, axis,
                                    length, s_k)
        events.append((s_k, 1 if net > 0 else -1, names[int(index[idx[k0]])]))

    out = []
    start = None
    for s_k, k, name in sorted(events, key=lambda e: e[0]):
        if k < 0 and start is None:
            start = (s_k, name)
        elif k > 0 and start is not None:
            if s_k > start[0]:
                out.append((start[0], s_k, start[1], name))
            start = None
    return out


def _refine_on_curved(mesh, tris, elems, is_quad, idx, l1, l2, cluster, q, tr, axis,
                      length, s_straight):
    """The crossing of a cluster moved onto the curved surface: tried on the
    crossed elements first, then on the elements around them."""
    tried = set()
    for k in cluster:
        e = int(elems[idx[k]])
        if e in tried:
            continue
        tried.add(e)
        if is_quad[idx[k]]:
            start = (0.5, 0.5)
        else:
            start = (1.0 - l1[k] - l2[k], l1[k])     # reference (1,0), (0,1), (0,0)
        hit = _curved_crossing(mesh, e, bool(is_quad[idx[k]]), q, tr, axis, start, length)
        if hit is not None:
            return hit
    # the neighbours (sharing a vertex) of the crossed triangles
    verts = tris[idx[cluster]].reshape(-1, 3)
    near = np.zeros(len(tris), dtype=bool)
    for v in verts:
        near |= np.any(np.all(np.isclose(tris, v[None, None, :], rtol=0,
                                         atol=1e-12 * length), axis=2), axis=1)
    for j in np.nonzero(near)[0]:
        e = int(elems[j])
        if e in tried:
            continue
        tried.add(e)
        start = (0.5, 0.5) if is_quad[j] else (1.0 / 3.0, 1.0 / 3.0)
        hit = _curved_crossing(mesh, e, bool(is_quad[j]), q, tr, axis, start, length)
        if hit is not None:
            return hit
    return s_straight
