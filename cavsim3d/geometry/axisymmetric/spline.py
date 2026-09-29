"""Spline-walled cavity (cavsim2d's ``SplineCavity``)."""

from typing import Optional

import numpy as np

from .base import AxisymmetricGeometry, _beampipe_option
from .profile import Profile

_KIND_ALIASES = {'bspline': 'bspline', 'bezier': 'bezier', 'berzier': 'bezier'}

_CELL_ALIASES = {
    'mid_cell': 'mid_cell', 'mid': 'mid_cell', 'mid cell': 'mid_cell', 'ic': 'mid_cell',
    'end_cell_left': 'end_cell_left', 'end cell left': 'end_cell_left',
    'end_cell': 'end_cell_left', 'end cell': 'end_cell_left',
    'left': 'end_cell_left', 'oc': 'end_cell_left',
    'end_cell_right': 'end_cell_right', 'end cell right': 'end_cell_right',
    'right': 'end_cell_right', 'oc_r': 'end_cell_right',
}


def _poly(cell):
    """Control polygon ``(N, 2)`` from a ``{p0: [z, r], ...}`` dict, by point index."""
    keys = sorted(cell, key=lambda s: int(''.join(c for c in str(s) if c.isdigit()) or 0))
    return np.array([cell[k] for k in keys], dtype=float)


class SplineCavity(AxisymmetricGeometry):
    """A cavity whose wall is a spline through control points.

    Parameters
    ----------
    shape : dict
        - ``'geometry'`` (required): control points ``[z, r]``, keyed
          ``'p0'``, ``'p1'``, ... along the wall. The first and last points are
          the apertures (``p0`` sits at the left iris). Either one control-point
          dict for every cell, or per-cell dicts under ``'mid_cell'`` /
          ``'end_cell_left'`` / ``'end_cell_right'`` (end cells default to the
          mid cell).
        - ``'n_cells'`` (default 1): number of cells.
        - ``'beampipe'`` (default ``'none'``): ``'none'``, ``'left'``,
          ``'right'`` or ``'both'``.
        - ``'beampipe_length'``: default one cell width.
    kind : {'Bezier', 'bspline'}
        ``'Bezier'``: one Bezier curve per cell. ``'bspline'``: one clamped
        cubic B-spline through every cell's poles.
    unit : {'mm', 'm', 'cm', 'um'}
        Unit of every length above (default ``'mm'``).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.

    Examples
    --------
    >>> geom = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
    ...         'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}
    >>> SplineCavity({'geometry': geom, 'n_cells': 2, 'beampipe': 'both'}, maxh=0.03)
    """

    def __init__(self, shape: Optional[dict] = None, kind: str = 'Bezier', unit: str = 'mm',
                 maxh: float = 0.05, config: Optional[dict] = None):
        super().__init__()
        self._require(shape=shape)
        self._set_unit(unit)
        self.kind = kind
        if _KIND_ALIASES.get(str(kind).lower()) is None:
            raise ValueError(f"spline kind must be 'Bezier' or 'bspline', got {kind!r}.")
        self.n_cells = int(shape.get('n_cells', 1))
        self.beampipe = _beampipe_option(shape.get('beampipe', 'none'))
        self.beampipe_length = shape.get('beampipe_length', None)
        self.parameters = shape['geometry']
        self._finish_init(maxh, dict(
            shape={'geometry': self.parameters, 'n_cells': self.n_cells,
                   'beampipe': self.beampipe, 'beampipe_length': self.beampipe_length},
            kind=kind, maxh=maxh))

    def _cell_polys(self):
        """The ``n_cells`` control polygons (in ``unit``) in cell order."""
        geom = self.parameters
        if any(isinstance(v, dict) for v in geom.values()):
            norm = {_CELL_ALIASES.get(str(k).lower().strip(), str(k).lower()): v
                    for k, v in geom.items()}
            mid_src = norm.get('mid_cell') or norm.get('end_cell_left') or norm.get('end_cell_right')
            if mid_src is None:
                raise ValueError('SplineCavity geometry has no mid_cell / end-cell polygon.')
            mid = _poly(mid_src)
            left = _poly(norm['end_cell_left']) if 'end_cell_left' in norm else mid
            right = _poly(norm['end_cell_right']) if 'end_cell_right' in norm else mid
        else:
            mid = left = right = _poly(geom)
        for c in (mid, left, right):
            if c.ndim != 2 or c.shape[0] < 3 or c.shape[1] != 2:
                raise ValueError('each control polygon needs at least 3 [z, r] points.')
        n = self.n_cells
        return [mid] if n == 1 else [left] + [mid] * (n - 2) + [right]

    def control_polygons(self):
        """The per-cell control polygons in metres, placed one after another in z."""
        placed, z_cursor = [], 0.0
        for c in self._cell_polys():
            s = c * self._s
            s[:, 0] += z_cursor - s[0][0]
            placed.append(s)
            z_cursor += s[-1][0] - s[0][0]
        return placed

    def profile(self) -> Profile:
        """The meridian in metres; the wall is an exact spline."""
        kind = _KIND_ALIASES[str(self.kind).lower()]
        placed = self.control_polygons()
        r_ap_l = float(placed[0][0][1])
        r_ap_r = float(placed[-1][-1][1])
        default_bp = float(placed[0][-1][0] - placed[0][0][0])
        L_bp = float(self.beampipe_length) * self._s if self.beampipe_length else default_bp
        L_bp_l = L_bp if self.beampipe in ('both', 'left') else 0.0
        L_bp_r = L_bp if self.beampipe in ('both', 'right') else 0.0

        prof = Profile('spline')
        z0 = -L_bp_l
        prof.start(z0, 0.0)
        prof.line_to(z0, r_ap_l, 'PMC')                     # left aperture
        if L_bp_l > 0:
            prof.line_to(0.0, r_ap_l, 'PEC')                # left beam pipe
        if kind == 'bspline':
            poles = [p for s in placed for p in s[1:].tolist()]
            prof.spline_to(poles, 'PEC', kind='bspline', degree=3)
        else:
            for s in placed:
                prof.spline_to(s[1:].tolist(), 'PEC', kind='bezier')
        z_end = float(placed[-1][-1][0])
        if L_bp_r > 0:
            z_end += L_bp_r
            prof.line_to(z_end, r_ap_r, 'PEC')              # right beam pipe
        prof.line_to(z_end, 0.0, 'PMC')                     # right aperture
        return prof.close('AXI')
