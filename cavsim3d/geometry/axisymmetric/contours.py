"""Meridian contours of elliptical cavities, built on :class:`Profile`.

Every cell of an elliptical cavity is a *forward half* (iris arc -> tangent
line -> equator arc), an optional flat top, and a *backward half* (equator arc
-> tangent line -> iris arc). One builder therefore covers single-cell and
multicell cavities, asymmetric end cells, one-sided beam pipes and the
flat-top parameterisation. Ported from cavsim2d. Coordinates are in metres.
"""
import numpy as np

from .profile import Profile
from .tangency import wall_tangent


class DegenerateGeometry(ValueError):
    """The parameter set admits no tangent line from the iris to the equator ellipse."""


def tangent_offsets(cell, Req):
    """Tangent points of one half-cell ``(A, B, a, b, Ri, L, ...)``, as offsets
    from its iris plane: ``(dx1, y1, dx2, y2)``."""
    A, B, a, b, Ri, L = cell[:6]
    pts = wall_tangent(A, B, a, b, Ri, L, Req)
    if pts is None:
        raise DegenerateGeometry(
            f'half-cell (A, B, a, b, Ri, L, Req) = '
            f'{tuple(round(float(v) * 1e3, 6) for v in (A, B, a, b, Ri, L, Req))} mm '
            'is degenerate: the iris and equator ellipses overlap, so no straight '
            'wall joins them.')
    return pts


def _beampipe_lengths(beampipe, L_bp):
    bp = str(beampipe).lower()
    if bp not in ('none', 'left', 'right', 'both'):
        raise ValueError(f"beampipe must be 'none', 'left', 'right' or 'both', got {beampipe!r}.")
    return (L_bp if bp in ('both', 'left') else 0.0,
            L_bp if bp in ('both', 'right') else 0.0)


def half_cell_sequence(mid, end_l, end_r, n_cells):
    """The ``2 * n_cells`` half-cells of a cavity, left to right.

    ``[end_l, mid, mid, ..., mid, end_r]``: cell *k* is the pair
    ``(half_cells[2k], half_cells[2k+1])``, its forward and backward half.
    """
    if n_cells == 1:
        halves = [end_l, end_r]
    else:
        halves = [end_l] + [mid] * (2 * n_cells - 2) + [end_r]
    return [list(h) for h in halves]


def elliptical_profile_from_half_cells(half_cells, beampipe, L_bp, flats=None,
                                       name='elliptical'):
    """Contour of an elliptical cavity whose half-cells may all differ.

    ``half_cells`` is a ``(2 * n_cells, >=7)`` sequence in **metres**, ordered by
    :func:`half_cell_sequence`, each ``(A, B, a, b, Ri, L, Req)``. ``flats`` is an
    optional per-cell flat-top length. The cavity is centred on ``z = 0``.

    Raises :class:`DegenerateGeometry` if a half-cell has no tangent solution.
    """
    halves = [list(h) for h in half_cells]
    n_cells = len(halves) // 2
    if len(halves) != 2 * n_cells or n_cells < 1:
        raise ValueError('half_cells must hold an even number of half-cells')
    flats = [0.0] * n_cells if flats is None else list(flats)

    L_bp_l, L_bp_r = _beampipe_lengths(beampipe, L_bp)
    shift = (L_bp_l + L_bp_r + sum(h[5] for h in halves) + sum(flats)) / 2.0

    p = Profile(name)
    z = -shift
    p.start(z, 0.0)
    p.line_to(z, halves[0][4], 'PMC')                   # left aperture
    if L_bp_l > 0:
        z += L_bp_l
        p.line_to(z, halves[0][4], 'PEC')               # left beam pipe

    for k in range(n_cells):
        left, right = halves[2 * k], halves[2 * k + 1]

        # forward half: iris plane at z -> equator
        A, B, a, b, Ri, L, Req = left[:7]
        dx1, y1, dx2, y2 = tangent_offsets(left, Req)
        p.ellipse_arc_to(z + dx1, y1, center=(z, Ri + b), semi_z=a, semi_r=b, boundary='PEC')
        p.line_to(z + dx2, y2, 'PEC')
        z_eq = z + L
        p.ellipse_arc_to(z_eq, Req, center=(z_eq, Req - B), semi_z=A, semi_r=B, boundary='PEC')

        if flats[k] > 0:                                # flat top across the equator
            z_eq += flats[k]
            p.line_to(z_eq, Req, 'PEC')

        # backward half: equator -> next iris plane
        A, B, a, b, Ri, L, Req = right[:7]
        dx1, y1, dx2, y2 = tangent_offsets(right, Req)
        z_next = z_eq + L
        p.ellipse_arc_to(z_next - dx2, y2, center=(z_eq, Req - B), semi_z=A, semi_r=B,
                         boundary='PEC')
        p.line_to(z_next - dx1, y1, 'PEC')
        p.ellipse_arc_to(z_next, Ri, center=(z_next, Ri + b), semi_z=a, semi_r=b, boundary='PEC')
        z = z_next

    if L_bp_r > 0:
        z += L_bp_r
        p.line_to(z, halves[-1][4], 'PEC')              # right beam pipe
    p.line_to(z, 0.0, 'PMC')                            # right aperture
    p.close('AXI')
    return p


def flat_lengths(mid, end_l, end_r, n_cells):
    """Flat-top length of each cell (the 8th parameter ``l``), left to right."""
    l_m, l_el, l_er = mid[7], end_l[7], end_r[7]
    if n_cells == 1:
        # cavsim2d's convention for a single cell: l_el + l_er - l_m.
        return [l_el + l_er - l_m]
    return [l_el] + [l_m] * (n_cells - 2) + [l_er]


def elliptical_profile(mid, end_l, end_r, n_cells, beampipe, flattop=False,
                       beampipe_length=None, name='elliptical'):
    """Meridian :class:`Profile` of an elliptical (or flat-top) cavity.

    ``mid`` / ``end_l`` / ``end_r`` are per-cell parameters in **metres**,
    ``(A, B, a, b, Ri, L, Req)`` plus a trailing flat-top length ``l`` when
    ``flattop`` is set. ``Req`` is taken from ``mid`` for every cell.
    ``beampipe_length`` defaults to ``2 * L_m``.
    """
    halves = half_cell_sequence(mid, end_l, end_r, n_cells)
    for h in halves:
        h[6] = mid[6]
    flats = flat_lengths(mid, end_l, end_r, n_cells) if flattop else None
    L_bp = float(beampipe_length) if beampipe_length is not None else 2 * mid[5]
    return elliptical_profile_from_half_cells(halves, beampipe, L_bp, flats=flats, name=name)
