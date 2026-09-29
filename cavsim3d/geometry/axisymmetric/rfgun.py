"""VHF-type RF gun (cavsim2d's ``RFGun``)."""

import warnings
from typing import Optional

import numpy as np

from .base import AxisymmetricGeometry, _beampipe_option
from .profile import Profile

_GUN_KEYS = ('y1', 'R2', 'T2', 'L3', 'R4', 'L5', 'R6', 'L7', 'R8', 'T9',
             'R10', 'T10', 'L11', 'R12', 'L13', 'R14', 'x')
#: The angles among them (radians, never scaled by ``unit``).
_GUN_ANGLES = ('T2', 'T9', 'T10')


class RFGun(AxisymmetricGeometry):
    """A VHF-type RF gun whose wall is a chain of arcs and straight segments.

    As in cavsim2d, the gun's lengths default to **metres** (``unit='m'``,
    unlike the other models) and its angles are in radians.

    Parameters
    ----------
    shape : dict
        ``{'geometry': {...}}`` (or the geometry dict itself). The keys, in wall
        order from the cathode plane to the exit aperture:

        - ``y1``: cathode-plane aperture radius.
        - ``R2``: radius of the cathode-nose arc; ``T2`` its angle.
        - ``L3``: straight length after that arc (at angle ``T2``).
        - ``R4``: blend-arc radius; ``L5`` the following straight length.
        - ``R6``: radius of the arc into the barrel; ``L7`` the barrel length.
        - ``R8``: nose-cone entry-arc radius; ``T9`` its angle (the closing arc
          radius ``R9`` is derived so the wall stays tangent).
        - ``R10``: exit-nose arc radius; ``T10`` its angle.
        - ``L11``: straight length after the nose; ``R12`` the next arc radius.
        - ``L13``: straight length to the exit; ``R14`` the exit-aperture arc.
        - ``x``: exit drift-tube radius step to the exit aperture.
    beampipe : {'none', 'left', 'both', 'right'}
        ``'left'`` or ``'both'`` adds a cathode-side beam pipe of radius ``y1``
        whose entrance lies upstream of the barrel. The exit drift is always
        built in, so ``'right'`` changes nothing.
    beampipe_length : float, optional
        How far the cathode-pipe entrance lies beyond the barrel's leftmost
        wall; default ``10 * y1``.
    unit : {'m', 'mm', 'cm', 'um'}
        Unit of every length above (default ``'m'``, as in cavsim2d).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.

    Notes
    -----
    The two ports are the cathode plane (or the cathode-pipe entrance) and the
    exit aperture.
    """

    def __init__(self, shape: Optional[dict] = None, beampipe: str = 'none',
                 beampipe_length: Optional[float] = None, unit: str = 'm',
                 maxh: float = 0.05, config: Optional[dict] = None):
        super().__init__()
        self._require(shape=shape)
        self._set_unit(unit)
        geometry = shape.get('geometry', shape)
        missing = [k for k in _GUN_KEYS if k not in geometry]
        if missing:
            raise ValueError(f'RFGun geometry is missing {missing}.')
        self.parameters = {k: float(geometry[k]) for k in _GUN_KEYS}
        self.beampipe = _beampipe_option(beampipe)
        self.beampipe_length = beampipe_length
        self._finish_init(maxh, dict(shape={'geometry': self.parameters},
                                     beampipe=self.beampipe,
                                     beampipe_length=beampipe_length, maxh=maxh))

    def _scaled(self):
        """The gun parameters with every length in metres."""
        return {k: v if k in _GUN_ANGLES else v * self._s
                for k, v in self.parameters.items()}

    @staticmethod
    def _r9(p):
        """Radius of the arc that closes the gun contour, fixed by the others."""
        return (((p['y1'] + p['R2'] * np.sin(p['T2']) + p['L3'] * np.cos(p['T2'])
                  + p['R4'] * np.sin(p['T2']) + p['L5'] + p['R6'])
                 - (p['R14'] + p['L13'] + p['R12'] * np.sin(p['T10'])
                    + p['L11'] * np.cos(p['T10']) + p['R10'] * np.sin(p['T10'])
                    + p['x'] + p['R8'] * (1 - np.sin(p['T9'])))) / np.sin(p['T9']))

    def _walls(self, prof: Profile) -> Profile:
        """Barrel and exit drift, continuing from the cathode aperture ``(0, y1)``."""
        p = self._scaled()
        y1, R2, T2, L3, R4, L5, R6, L7, R8, T9, R10, T10, L11, R12, L13, R14, x = (
            p[k] for k in _GUN_KEYS)
        R9 = self._r9(p)

        z, r = R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)
        prof.circle_arc_to(z, r, center=(-R2, y1), boundary='PEC')

        z, r = z - L3 * np.cos(T2), r + L3 * np.sin(T2)
        prof.line_to(z, r, 'PEC')

        c = (z + R4 * np.cos(T2), r + R4 * np.sin(T2))
        z, r = z - (R4 - R4 * np.cos(T2)), r + R4 * np.sin(T2)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z, r + L5
        prof.line_to(z, r, 'PEC')

        c = (z + R6, r)
        z, r = z + R6, r + R6
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z + L7, r
        prof.line_to(z, r, 'PEC')

        c = (z, r - R8)
        z, r = z + R8 * np.cos(T9), r - (R8 - R8 * np.sin(T9))
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        c = (z - R9 * np.cos(T9), r - R9 * np.sin(T9))
        z, r = z + (R9 - R9 * np.cos(T9)), r - R9 * np.sin(T9)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        c = (z - R10, r)
        z, r = z - (R10 - R10 * np.cos(T10)), r - R10 * np.sin(T10)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z - L11 * np.sin(T10), r - L11 * np.cos(T10)
        prof.line_to(z, r, 'PEC')

        c = (z + R12 * np.cos(T10), r - R12 * np.sin(T10))
        z, r = z - (R12 - R12 * np.cos(T10)), r - R12 * np.sin(T10)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z, r - L13
        prof.line_to(z, r, 'PEC')

        c = (z + R14, r)
        z, r = z + R14, r - R14
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z + 10 * y1, r
        prof.line_to(z, r, 'PEC')

        # R9 is derived so that the wall returns to the axis here (r == x).
        # End exactly on the axis; rounded angles leave a small residual.
        if abs(r - x) > 1e-3 * x:
            warnings.warn(
                f'RFGun contour does not close: the exit aperture ends '
                f'{(r - x) / self._s:.4g} {self.unit} off the axis. Check the '
                'angles (radians) and lengths.', UserWarning, stacklevel=3)
        prof.line_to(z, 0.0, 'PMC')                                # exit aperture
        return prof.close('AXI')

    def profile(self) -> Profile:
        """The meridian in metres, with exact circular arcs."""
        y1 = self.parameters['y1'] * self._s
        if self.beampipe in ('left', 'both'):
            # The cathode pipe's entrance must lie upstream of the barrel, so
            # build the bare gun once to find the barrel's leftmost z.
            base = self._walls(Profile('rfgun').start(0.0, 0.0).line_to(0.0, y1, 'PMC'))
            z_barrel = min(pt[0] for pt in base.points)
            L_bp = (float(self.beampipe_length) * self._s if self.beampipe_length
                    else 10 * y1)
            z_entrance = z_barrel - L_bp
            prof = (Profile('rfgun').start(z_entrance, 0.0)
                    .line_to(z_entrance, y1, 'PMC')                # pipe entrance
                    .line_to(0.0, y1, 'PEC'))                      # pipe wall
            return self._walls(prof)
        return self._walls(Profile('rfgun').start(0.0, 0.0).line_to(0.0, y1, 'PMC'))
