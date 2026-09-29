"""Elliptical cavities (cavsim2d's ``EllipticalCavity`` / ``EllipticalCavityFlatTop``)."""

import warnings
from typing import Optional

import numpy as np

from .base import AxisymmetricGeometry, _beampipe_option
from .contours import elliptical_profile, half_cell_sequence, elliptical_profile_from_half_cells
from .profile import Profile
from .tangency import wall_angle

_CELL_NAMES = ('A', 'B', 'a', 'b', 'Ri', 'L', 'Req')


def _split_cells(mid_cell, end_cell_left, end_cell_right, n_params):
    """``(mid, left, right)`` float arrays; a ``{'IC', 'OC', 'OC_R'}`` dict is accepted."""
    if isinstance(mid_cell, dict):
        end_cell_left = mid_cell.get('OC', end_cell_left)
        end_cell_right = mid_cell.get('OC_R', end_cell_right)
        mid_cell = mid_cell['IC']
    if mid_cell is None:
        raise ValueError('mid_cell is required.')
    mid = np.asarray(mid_cell, dtype=float)[:n_params].copy()
    left = mid.copy() if end_cell_left is None else np.asarray(end_cell_left, dtype=float)[:n_params].copy()
    right = left.copy() if end_cell_right is None else np.asarray(end_cell_right, dtype=float)[:n_params].copy()
    for label, cell in (('mid_cell', mid), ('end_cell_left', left), ('end_cell_right', right)):
        if len(cell) < n_params:
            raise ValueError(f'{label} needs {n_params} parameters, got {len(cell)}.')
    return mid, left, right


def _unify_equator_radius(mid, left, right, n_cells):
    """Force one ``Req`` on every cell, warning if the input disagreed."""
    req = (float(mid[6]), float(left[6]), float(right[6]))
    if np.allclose(req, req[0]):
        return
    canonical, source = (req[0], 'mid-cell') if n_cells >= 2 else (req[1], 'end-cell-left')
    warnings.warn(
        f"Req differs across cells (mid={req[0]}, end-left={req[1]}, end-right={req[2]}). "
        f"Req is shared by every cell; using the {source} value ({canonical}).",
        UserWarning, stacklevel=3)
    for cell in (mid, left, right):
        cell[6] = canonical


class EllipticalCavity(AxisymmetricGeometry):
    r"""A multi-cell elliptical RF cavity, revolved from its cavsim2d meridian.

    Each half-cell is ``[A, B, a, b, Ri, L, Req]`` (mm by default)::

            r
            ^                        equator ellipse
          Req |- - - - - - - - - .--''''''--.   (z semi-axis A, r semi-axis B)
              |                ,-'            \
              |              /                 |
              |            /  <- wall, angle alpha to the z-axis
              |          ,'                    |
              |    _..--'   iris ellipse       |
           Ri |--''         (z semi-axis a,    |
              |              r semi-axis b)     |
              +----------------------------------------> z
              |<--------------- L -------------->|
                       (half-cell length; full cell = 2 L)

    Parameters
    ----------
    n_cells : int
        Number of cells.
    mid_cell, end_cell_left, end_cell_right : sequence of 7 floats
        ``[A, B, a, b, Ri, L, Req]`` of the interior cells and the two end
        cells (an 8th ``alpha`` slot is ignored). The end cells default to the
        mid cell. A dict ``{'IC': mid, 'OC': left, 'OC_R': right}`` is also
        accepted as ``mid_cell``. ``Req`` is shared by every cell.
    beampipe : {'none', 'left', 'right', 'both'}
        Which ends carry a beam pipe of radius ``Ri``.
    beampipe_length : float, optional
        Beam-pipe length; default ``2 * L`` of the mid cell. The ports sit
        at the pipe ends, so a longer pipe lets evanescent fields decay first.
    chain : int
        Copies of the cavity chained end-to-end into one module (default 1).
    spacing : float or sequence of float, optional
        Inter-cavity drift when ``chain > 1`` (iris to iris); ``None``
        keeps the cavity's own beam pipes.
    unit : {'mm', 'm', 'cm', 'um'}
        Unit of every length above (default ``'mm'``).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.

    Examples
    --------
    >>> tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    >>> cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both', maxh=0.02)
    >>> cav.ports
    ['port1', 'port2']
    """

    def __init__(self, n_cells: Optional[int] = None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe: str = 'none',
                 beampipe_length: Optional[float] = None, chain: int = 1, spacing=None,
                 unit: str = 'mm', maxh: float = 0.05, config: Optional[dict] = None):
        super().__init__()
        self._require(n_cells=n_cells, mid_cell=mid_cell)
        self._set_unit(unit)
        self.n_cells = int(n_cells)
        if self.n_cells < 1:
            raise ValueError(f'n_cells must be at least 1, got {n_cells!r}.')
        self.beampipe = _beampipe_option(beampipe)
        self.beampipe_length = beampipe_length
        self.chain = int(chain)
        self.spacing = spacing
        self.mid_cell, self.end_cell_left, self.end_cell_right = _split_cells(
            mid_cell, end_cell_left, end_cell_right, 7)
        _unify_equator_radius(self.mid_cell, self.end_cell_left, self.end_cell_right,
                              self.n_cells)
        self.parameters = {f'{n}_{suf}': float(cell[i])
                           for suf, cell in (('m', self.mid_cell), ('el', self.end_cell_left),
                                             ('er', self.end_cell_right))
                           for i, n in enumerate(_CELL_NAMES)}
        self._finish_init(maxh, dict(
            n_cells=self.n_cells, mid_cell=self.mid_cell, end_cell_left=self.end_cell_left,
            end_cell_right=self.end_cell_right, beampipe=self.beampipe,
            beampipe_length=beampipe_length, chain=self.chain, spacing=spacing, maxh=maxh))

    def half_cells(self) -> np.ndarray:
        """The ``(2 * n_cells, 7)`` half-cell parameters in ``unit``, left to right."""
        halves = half_cell_sequence(self.mid_cell, self.end_cell_left,
                                    self.end_cell_right, self.n_cells)
        return np.array(halves, dtype=float)

    def wall_angles(self) -> dict:
        """Wall inclination ``alpha`` (degrees) of the mid and end cells."""
        return {suf: wall_angle(*cell[:7]) for suf, cell in
                (('m', self.mid_cell), ('el', self.end_cell_left), ('er', self.end_cell_right))}

    def active_length(self) -> float:
        """``2 L n_cells`` [m] with the mid-cell ``L``, per chained cavity: the
        length ``Eacc`` is normalised to (cavsim2d's convention)."""
        return 2 * float(self.mid_cell[5]) * self._s * self.n_cells * self.chain

    def profile(self) -> Profile:
        """The meridian in metres, with exact ellipse arcs."""
        halves = self.half_cells() * self._s
        L_bp = (float(self.beampipe_length) * self._s if self.beampipe_length is not None
                else 2 * self.mid_cell[5] * self._s)
        prof = elliptical_profile_from_half_cells(halves, self.beampipe, L_bp,
                                                  name='elliptical')
        return self._chained(prof)


class EllipticalCavityFlatTop(AxisymmetricGeometry):
    """An elliptical cavity with a straight (flat-top) section at each equator.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    mid_cell, end_cell_left, end_cell_right : sequence of 8 floats
        ``[A, B, a, b, Ri, L, Req, l]``: the seven :class:`EllipticalCavity`
        parameters plus the flat-top length ``l``, so a full cell is
        ``2 * L + l`` long. The end cells default to the mid cell.
    beampipe : {'none', 'left', 'right', 'both'}
        Which ends carry a beam pipe of radius ``Ri``.
    beampipe_length : float, optional
        Beam-pipe length; default ``2 * L`` of the mid cell.
    unit : {'mm', 'm', 'cm', 'um'}
        Unit of every length above (default ``'mm'``).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.

    Examples
    --------
    >>> ft = [42, 42, 12, 19, 35, 57.7, 103.353, 20]    # 20 mm flat top
    >>> EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both', maxh=0.02)
    """

    def __init__(self, n_cells: Optional[int] = None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe: str = 'none',
                 beampipe_length: Optional[float] = None, unit: str = 'mm',
                 maxh: float = 0.05, config: Optional[dict] = None):
        super().__init__()
        self._require(n_cells=n_cells, mid_cell=mid_cell)
        self._set_unit(unit)
        self.n_cells = int(n_cells)
        if self.n_cells < 1:
            raise ValueError(f'n_cells must be at least 1, got {n_cells!r}.')
        self.beampipe = _beampipe_option(beampipe)
        self.beampipe_length = beampipe_length
        self.mid_cell, self.end_cell_left, self.end_cell_right = _split_cells(
            mid_cell, end_cell_left, end_cell_right, 8)
        self.parameters = {f'{n}_{suf}': float(cell[i])
                           for suf, cell in (('m', self.mid_cell), ('el', self.end_cell_left),
                                             ('er', self.end_cell_right))
                           for i, n in enumerate(_CELL_NAMES + ('l',))}
        self._finish_init(maxh, dict(
            n_cells=self.n_cells, mid_cell=self.mid_cell, end_cell_left=self.end_cell_left,
            end_cell_right=self.end_cell_right, beampipe=self.beampipe,
            beampipe_length=beampipe_length, maxh=maxh))

    def active_length(self) -> float:
        """``n_cells (2 L + l)`` [m] with the mid-cell ``L`` and flat ``l``: the
        length ``Eacc`` is normalised to (cavsim2d's convention)."""
        return self.n_cells * (2 * float(self.mid_cell[5]) + float(self.mid_cell[7])) * self._s

    def profile(self) -> Profile:
        """The meridian in metres; each flat top is a straight equator segment."""
        bp = None if self.beampipe_length is None else float(self.beampipe_length) * self._s
        return elliptical_profile(self.mid_cell * self._s, self.end_cell_left * self._s,
                                  self.end_cell_right * self._s, self.n_cells, self.beampipe,
                                  flattop=True, beampipe_length=bp,
                                  name='elliptical_flattop')
