"""Tangent-line and corner-rounding solvers for meridian contours.

Ported from cavsim2d: the wall tangent of an elliptical half-cell, and the
inscribed-arc corners that the bellows and the taper round their walls with.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def wall_tangent(A, B, a, b, Ri, L, Req):
    """Tangent points ``[x1, y1, x2, y2]`` of a half-cell wall, iris plane at ``z = 0``.

    The straight wall joins the iris ellipse (semi-axes ``a``, ``b``, centred at
    ``(0, Ri + b)``) to the equator ellipse (semi-axes ``A``, ``B``, centred at
    ``(L, Req - B)``); it is their internal common tangent on the ``-pi/2``
    side. ``(x1, y1)`` is the tangent point on the iris ellipse and
    ``(x2, y2)`` the one on the equator ellipse. Returns ``None`` when the two
    ellipses overlap, so no tangent line joins them.

    Writing the line as ``n . X = c`` with unit normal ``n = (cos t, sin t)``
    leaves one scalar equation in ``t``::

        g(t) = n . (C_eq - C_iris) - |(A n_z, B n_r)| - |(a n_z, b n_r)| = 0

    whose root on the ``-pi/2`` side of the maximum of ``g`` is bracketed and
    solved with Brent's method.
    """
    h, k, p, q = 0.0, Ri + b, L, Req - B
    d_z, d_r = p - h, q - k

    def g(t):
        nz, nr = np.cos(t), np.sin(t)
        return nz * d_z + nr * d_r - np.hypot(A * nz, B * nr) - np.hypot(a * nz, b * nr)

    lo = -np.pi / 2
    peak = minimize_scalar(lambda t: -g(t), bounds=(lo, np.pi / 2), method='bounded',
                           options={'xatol': 1e-14})
    t_max = float(peak.x)
    if not g(t_max) > 0:
        return None
    t = brentq(g, lo, t_max, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)

    nz, nr = np.cos(t), np.sin(t)
    s_iris = np.hypot(a * nz, b * nr)
    s_eq = np.hypot(A * nz, B * nr)
    return np.array([h + a * a * nz / s_iris, k + b * b * nr / s_iris,
                     p - A * A * nz / s_eq, q - B * B * nr / s_eq])


def wall_angle(A, B, a, b, Ri, L, Req):
    """Wall inclination ``alpha`` of a half-cell, in degrees (90 = vertical).

    Returns ``None`` for a degenerate half-cell (no tangent line).
    """
    pts = wall_tangent(A, B, a, b, Ri, L, Req)
    if pts is None:
        return None
    x1, y1, x2, y2 = pts
    return float(180 - np.degrees(np.arctan2(y2 - y1, x2 - x1)))


#: Lengths below this (metres) count as zero when rounding a wall.
CORNER_TOL = 1e-12


def inscribed_corner(vertex, prev_pt, next_pt, radius, tol=CORNER_TOL):
    """The arc of radius *radius* that rounds the corner at *vertex*.

    Returns ``(t_in, t_out, centre)`` (the two tangent points and the arc
    centre) or ``None`` when the corner is straight, the radius is zero, or an
    adjacent edge has no length. For a wedge of angle ``phi`` the tangent
    points sit ``radius / tan(phi / 2)`` back from the vertex along each edge
    and the centre is on the bisector at ``radius / sin(phi / 2)``.
    """
    v = np.asarray(vertex, dtype=float)
    u_in = v - np.asarray(prev_pt, dtype=float)
    u_out = np.asarray(next_pt, dtype=float) - v
    n_in, n_out = np.linalg.norm(u_in), np.linalg.norm(u_out)
    if radius <= 0 or n_in < tol or n_out < tol:
        return None
    u_in, u_out = u_in / n_in, u_out / n_out
    phi = np.arccos(float(np.clip(np.dot(-u_in, u_out), -1.0, 1.0)))
    if phi > np.pi - 1e-9:
        return None
    bisector = u_out - u_in
    norm = np.linalg.norm(bisector)
    if norm < tol:
        return None
    d = radius / np.tan(phi / 2.0)
    centre = v + (radius / np.sin(phi / 2.0)) * (bisector / norm)
    return v - d * u_in, v + d * u_out, centre


def corner_offset(vertex, prev_pt, next_pt, radius):
    """How far back from *vertex* a corner of *radius* reaches, along each edge."""
    got = inscribed_corner(vertex, prev_pt, next_pt, radius)
    if got is None:
        return 0.0
    return float(np.linalg.norm(got[0] - np.asarray(vertex, dtype=float)))


def emit_rounded_wall(prof, vertices, boundary, tol=CORNER_TOL):
    """Draw *vertices* onto *prof*, rounding each one by its own radius.

    ``vertices`` is ``[(z, r, radius), ...]`` in metres, walked in order; the
    first and last are endpoints and their radius is ignored. A corner with
    zero radius stays sharp, and a flat that the rounding consumes entirely is
    skipped rather than emitted as a zero-length edge. The current point of
    *prof* must already be ``vertices[0]``.
    """
    pts = [(float(v[0]), float(v[1])) for v in vertices]
    cursor = np.asarray(pts[0], dtype=float)
    for i in range(1, len(vertices) - 1):
        corner = inscribed_corner(pts[i], pts[i - 1], pts[i + 1], float(vertices[i][2]), tol)
        if corner is None:
            here = np.asarray(pts[i], dtype=float)
            if np.linalg.norm(here - cursor) > tol:
                prof.line_to(pts[i][0], pts[i][1], boundary)
                cursor = here
            continue
        t_in, t_out, centre = corner
        if np.linalg.norm(t_in - cursor) > tol:
            prof.line_to(t_in[0], t_in[1], boundary)
        prof.circle_arc_to(t_out[0], t_out[1], centre, boundary)
        cursor = t_out
    end = np.asarray(pts[-1], dtype=float)
    if np.linalg.norm(end - cursor) > tol:
        prof.line_to(end[0], end[1], boundary)
    return prof
