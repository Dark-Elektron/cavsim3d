"""Taper (cavsim2d's ``Taper``)."""

from typing import Optional

import numpy as np

from .base import AxisymmetricGeometry
from .profile import Profile
from .tangency import corner_offset, emit_rounded_wall


class Taper(AxisymmetricGeometry):
    """A conical transition from bore ``R_left`` to bore ``R_right``.

    Parameters
    ----------
    R_left, R_right : float
        Bore radius at the upstream and downstream ends.
    L : float
        Overall axial length.
    straight_left, straight_right : float
        Straight pipe before and after the cone (default 0: the cone spans the
        whole element).
    R_fillet : float
        Radius rounding both transition corners (default 0, sharp); needs a
        straight run on each side.
    chain : int
        Copies chained end-to-end (default 1).
    spacing : float or sequence of float, optional
        Gap between copies when ``chain > 1``.
    unit : {'mm', 'm', 'cm', 'um'}
        Unit of every length above (default ``'mm'``).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.
    """

    def __init__(self, R_left: Optional[float] = None, R_right: Optional[float] = None,
                 L: Optional[float] = None, straight_left: float = 0.0,
                 straight_right: float = 0.0, R_fillet: float = 0.0, chain: int = 1,
                 spacing=None, unit: str = 'mm', maxh: float = 0.05,
                 config: Optional[dict] = None):
        super().__init__()
        self._require(R_left=R_left, R_right=R_right, L=L)
        self._set_unit(unit)
        self.chain = int(chain)
        self.spacing = spacing
        self.parameters = {
            'R_left': float(R_left), 'R_right': float(R_right), 'L': float(L),
            'straight_left': float(straight_left), 'straight_right': float(straight_right),
            'R_fillet': float(R_fillet),
        }
        self.check_feasible()
        self._finish_init(maxh, dict(**self.parameters, chain=self.chain, spacing=spacing,
                                     maxh=maxh))

    @property
    def cone_length(self) -> float:
        """Axial length of the conical run, in ``unit``."""
        p = self.parameters
        return p['L'] - p['straight_left'] - p['straight_right']

    @property
    def half_angle(self) -> float:
        """Cone half-angle from the axis, in degrees (0 for a straight pipe)."""
        p = self.parameters
        dz = self.cone_length
        if dz <= 0:
            return 90.0
        return float(np.degrees(np.arctan2(abs(p['R_right'] - p['R_left']), dz)))

    def _vertices(self):
        """Wall vertices in metres: ``[(z, r, radius), ...]``."""
        p = self.parameters
        half = p['L'] * self._s / 2.0
        fil = p['R_fillet'] * self._s
        return [(-half, p['R_left'] * self._s, 0.0),
                (-half + p['straight_left'] * self._s, p['R_left'] * self._s, fil),
                (half - p['straight_right'] * self._s, p['R_right'] * self._s, fil),
                (half, p['R_right'] * self._s, 0.0)]

    def check_feasible(self):
        """Validate the straight runs and the fillet radius."""
        p = self.parameters
        for key in ('R_left', 'R_right', 'L'):
            if p[key] <= 0:
                raise ValueError(f'{key} must be positive, got {p[key]!r}.')
        for key in ('straight_left', 'straight_right', 'R_fillet'):
            if p[key] < 0:
                raise ValueError(f'{key} must be non-negative, got {p[key]!r}.')
        if self.cone_length < 0:
            raise ValueError(
                f"the straight runs ({p['straight_left']} + {p['straight_right']} {self.unit}) exceed "
                f"the element length L={p['L']} {self.unit}, leaving no room for the cone.")
        if p['R_fillet'] <= 0:
            return True
        verts = self._vertices()
        slant = float(np.hypot(verts[2][0] - verts[1][0], verts[2][1] - verts[1][1]))
        offsets = []
        for i, side in ((1, 'left'), (2, 'right')):
            run = p[f'straight_{side}'] * self._s
            d = corner_offset(verts[i][:2], verts[i - 1][:2], verts[i + 1][:2], verts[i][2])
            offsets.append(d)
            if d > run + 1e-12:
                raise ValueError(
                    f'the {side} fillet does not fit: it reaches {d / self._s:.4g} {self.unit} back '
                    f'along a straight_{side} of {run / self._s:.4g} {self.unit}. Reduce R_fillet, or '
                    f'lengthen straight_{side}.')
        if sum(offsets) > slant + 1e-12:
            raise ValueError(
                f'the two fillets do not both fit on the {slant / self._s:.4g} {self.unit} cone. '
                'Reduce R_fillet, or lengthen the taper.')
        return True

    def profile(self) -> Profile:
        """The meridian in metres; each end is an aperture at its own bore radius."""
        verts = self._vertices()
        (z0, r0, _), (z1, _, _) = verts[0], verts[-1]
        prof = Profile('taper').start(z0, 0.0).line_to(z0, r0, 'PMC')
        emit_rounded_wall(prof, verts, 'PEC')
        prof.line_to(z1, 0.0, 'PMC')
        return self._chained(prof.close('AXI'))
