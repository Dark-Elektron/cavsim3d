"""Pillbox cavity (cavsim2d's ``Pillbox``)."""

from typing import Optional

from .base import AxisymmetricGeometry, _beampipe_option
from .profile import Profile


class Pillbox(AxisymmetricGeometry):
    """A right-cylinder (pillbox) cavity with beam apertures, optionally multi-cell.

    Parameters
    ----------
    n_cells : int
        Number of cells (barrels).
    dims : sequence of 5 floats ``[L, Req, Ri, S, L_bp]``
        - ``L``: cell (barrel) length along the axis.
        - ``Req``: cavity radius.
        - ``Ri``: aperture radius, which is also the beam-pipe radius. Must be
          positive: the apertures are the ports.
        - ``S``: straight drift between adjacent cells at radius ``Ri``;
          must be positive when ``n_cells > 1``.
        - ``L_bp``: beam-pipe length at each end selected by ``beampipe``.
    beampipe : {'none', 'left', 'right', 'both'}
        Which ends carry a beam pipe of length ``L_bp``.
    unit : {'mm', 'm', 'cm', 'um'}
        Unit of every length above (default ``'mm'``).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.

    Examples
    --------
    >>> Pillbox(1, [100, 100, 20, 0, 50], beampipe='both', maxh=0.03)
    """

    def __init__(self, n_cells: Optional[int] = None, dims=None, beampipe: str = 'none',
                 unit: str = 'mm', maxh: float = 0.05, config: Optional[dict] = None):
        super().__init__()
        self._require(n_cells=n_cells, dims=dims)
        self._set_unit(unit)
        self.n_cells = int(n_cells)
        if self.n_cells < 1:
            raise ValueError(f'n_cells must be at least 1, got {n_cells!r}.')
        L, Req, Ri, S, L_bp = (float(v) for v in dims)
        if not 0 < Ri < Req:
            raise ValueError(f'need 0 < Ri < Req, got Ri={Ri}, Req={Req} ({self.unit}).')
        if L <= 0 or S < 0 or L_bp < 0:
            raise ValueError(f'need L > 0, S >= 0 and L_bp >= 0, got L={L}, S={S}, L_bp={L_bp}.')
        if self.n_cells > 1 and S <= 0:
            raise ValueError('a multi-cell pillbox needs S > 0: adjacent cells join '
                             'through a drift of length S at radius Ri.')
        self.beampipe = _beampipe_option(beampipe)
        self.parameters = {'L': L, 'Req': Req, 'Ri': Ri, 'S': S, 'L_bp': L_bp}
        self._finish_init(maxh, dict(n_cells=self.n_cells, dims=[L, Req, Ri, S, L_bp],
                                     beampipe=self.beampipe, maxh=maxh))

    def profile(self) -> Profile:
        """The meridian in metres."""
        L, Req, Ri, S, L_bp = (self.parameters[k] * self._s for k in ('L', 'Req', 'Ri', 'S', 'L_bp'))
        n = self.n_cells
        L_bp_l = L_bp if self.beampipe in ('both', 'left') else 0.0
        L_bp_r = L_bp if self.beampipe in ('both', 'right') else 0.0
        shift = (L_bp_l + L_bp_r + n * L + (n - 1) * S) / 2.0

        p = Profile('pillbox')
        z = -shift
        p.start(z, 0.0)
        p.line_to(z, Ri, 'PMC')                  # left aperture
        if L_bp_l > 0:
            z += L_bp_l
            p.line_to(z, Ri, 'PEC')              # left beam pipe
        for cell in range(1, n + 1):
            if cell > 1:                         # inter-cell drift
                z += S
                p.line_to(z, Ri, 'PEC')
            p.line_to(z, Req, 'PEC')             # up the end plate
            z += L
            p.line_to(z, Req, 'PEC')             # along the barrel
            p.line_to(z, Ri, 'PEC')              # down the end plate
        if L_bp_r > 0:
            z += L_bp_r
            p.line_to(z, Ri, 'PEC')              # right beam pipe
        p.line_to(z, 0.0, 'PMC')                 # right aperture
        return p.close('AXI')
