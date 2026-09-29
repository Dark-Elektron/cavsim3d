"""Bellows (cavsim2d's ``Bellows``)."""

from typing import Optional

import numpy as np

from .base import AxisymmetricGeometry
from .profile import Profile
from .tangency import emit_rounded_wall

#: Flats shorter than this (metres) are treated as absent.
FLAT_TOL = 1e-12


class Bellows(AxisymmetricGeometry):
    """A corrugated bellows section of ``N_conv`` convolutions.

    Each convolution is root flat, flank, crest flat, flank, with every corner
    rounded by an exact circular arc.

    Parameters
    ----------
    Ri : float
        Bore (root) radius.
    A : float
        Convolution depth; the crest radius is ``Ri + A``.
    L_p : float
        Convolution period.
    N_conv : int
        Number of convolutions.
    R_root, R_crest : float
        Corner radii at the root and at the crest.
    crest_fraction : float
        Share of the per-period flat length given to the crest (default 0.5).
    flank_angle : float
        Flank angle from the axis in degrees (default 90, vertical flanks).
    L_bp : float
        Straight beam pipe at each end (default 0).
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

    Raises
    ------
    ValueError
        If the corner radii or flanks do not fit, naming the violated constraint.

    Notes
    -----
    netgen refines the mesh to the surface curvature, not only to ``maxh``:
    each rounded corner is a torus section, meshed at about
    ``radius / curvaturesafety`` all the way round the bore. Small corner
    radii are therefore expensive in 3D (one convolution with 1.2 mm corners
    on a 35 mm bore gives ~100k elements at the default ``curvaturesafety=2``,
    ~9k at 0.5). With curved (high-order) elements a lower value is usually
    enough: ``geo.generate_mesh(maxh=..., curvaturesafety=0.5)``.
    """

    def __init__(self, Ri: Optional[float] = None, A: Optional[float] = None,
                 L_p: Optional[float] = None, N_conv: Optional[int] = None,
                 R_root: Optional[float] = None, R_crest: Optional[float] = None,
                 crest_fraction: float = 0.5, flank_angle: float = 90.0, L_bp: float = 0.0,
                 chain: int = 1, spacing=None, unit: str = 'mm', maxh: float = 0.05,
                 config: Optional[dict] = None):
        super().__init__()
        self._require(Ri=Ri, A=A, L_p=L_p, N_conv=N_conv, R_root=R_root, R_crest=R_crest)
        self._set_unit(unit)
        self.N_conv = int(N_conv)
        if self.N_conv < 1:
            raise ValueError(f'N_conv must be at least 1, got {N_conv!r}.')
        self.chain = int(chain)
        self.spacing = spacing
        self.parameters = {
            'Ri': float(Ri), 'A': float(A), 'L_p': float(L_p),
            'R_root': float(R_root), 'R_crest': float(R_crest),
            'crest_fraction': float(crest_fraction),
            'flank_angle': float(flank_angle), 'L_bp': float(L_bp),
        }
        self.check_feasible()
        self._finish_init(maxh, dict(N_conv=self.N_conv, **self.parameters,
                                     chain=self.chain, spacing=spacing, maxh=maxh))

    @property
    def corrugated_length(self) -> float:
        """Axial length of the corrugated run (excluding the end pipes), in ``unit``."""
        return self.N_conv * self.parameters['L_p']

    def _layout(self):
        """Per-period lengths in ``unit``: ``(w_root, w_crest, flank_dz, (d_root, d_crest))``."""
        p = self.parameters
        theta = np.radians(p['flank_angle'])
        flank_dz = 0.0 if abs(theta - np.pi / 2) < 1e-12 else p['A'] / np.tan(theta)
        flats = p['L_p'] - 2.0 * flank_dz
        w_crest = p['crest_fraction'] * flats
        w_root = flats - w_crest
        t = np.tan(theta / 2.0)
        return w_root, w_crest, flank_dz, (p['R_root'] * t, p['R_crest'] * t)

    def check_feasible(self):
        """Validate the corner radii against the flats and the flank."""
        p = self.parameters
        theta = p['flank_angle']
        if not 0.0 < theta <= 90.0:
            raise ValueError(f'flank_angle must be in (0, 90] degrees, got {theta!r}.')
        if not 0.0 < p['crest_fraction'] < 1.0:
            raise ValueError(f"crest_fraction must be strictly between 0 and 1, "
                             f"got {p['crest_fraction']!r}.")
        for key in ('Ri', 'A', 'L_p'):
            if p[key] <= 0:
                raise ValueError(f'{key} must be positive, got {p[key]!r}.')
        for key in ('R_root', 'R_crest', 'L_bp'):
            if p[key] < 0:
                raise ValueError(f'{key} must be non-negative, got {p[key]!r}.')
        w_root, w_crest, _, (d_root, d_crest) = self._layout()
        if w_root < 0 or w_crest < 0:
            raise ValueError(
                f"the flanks do not fit in the period: at flank_angle={theta} deg a depth "
                f"A={p['A']} {self.unit} needs {2 * p['A'] / np.tan(np.radians(theta)):.4g} {self.unit} of "
                f"the L_p={p['L_p']} {self.unit} period, leaving no room for the flats.")
        if 2.0 * d_root > w_root + FLAT_TOL:
            raise ValueError(
                f'root corners do not fit: they need {2 * d_root:.4g} {self.unit} of the '
                f'{w_root:.4g} {self.unit} root flat. Reduce R_root or lengthen L_p.')
        if 2.0 * d_crest > w_crest + FLAT_TOL:
            raise ValueError(
                f'crest corners do not fit: they need {2 * d_crest:.4g} {self.unit} of the '
                f'{w_crest:.4g} {self.unit} crest flat. Reduce R_crest or lengthen L_p.')
        flank = p['A'] / np.sin(np.radians(theta))
        if d_root + d_crest > flank + FLAT_TOL:
            raise ValueError(
                f'the root and crest corners do not both fit on the {flank:.4g} {self.unit} flank: '
                f'they need {d_root + d_crest:.4g} {self.unit}. Reduce R_root / R_crest, or '
                'deepen the convolution (A).')
        return True

    def _vertices(self):
        """Sharp-corner wall vertices in metres: ``[(z, r, radius), ...]``."""
        p = self.parameters
        Ri, A, L_bp = p['Ri'] * self._s, p['A'] * self._s, p['L_bp'] * self._s
        R_root, R_crest = p['R_root'] * self._s, p['R_crest'] * self._s
        w_root, w_crest, flank_dz, _ = self._layout()
        w_root, w_crest, flank_dz = w_root * self._s, w_crest * self._s, flank_dz * self._s
        z = 0.0
        out = [(z, Ri, 0.0)]
        z += w_root / 2.0
        for i in range(self.N_conv):
            out.append((z, Ri, R_root))             # root corner, turning up
            z += flank_dz
            out.append((z, Ri + A, R_crest))        # crest corner, levelling off
            z += w_crest
            out.append((z, Ri + A, R_crest))        # crest corner, turning down
            z += flank_dz
            out.append((z, Ri, R_root))             # root corner, levelling off
            if i < self.N_conv - 1:
                z += w_root
        z += w_root / 2.0
        out.append((z, Ri, 0.0))
        # Centre on z = 0.
        mid = z / 2.0
        out = [(zv - mid, rv, rad) for zv, rv, rad in out]
        if L_bp > 0:
            out = [(out[0][0] - L_bp, Ri, 0.0)] + out + [(out[-1][0] + L_bp, Ri, 0.0)]
        return out

    def profile(self) -> Profile:
        """The meridian in metres; both ends are apertures at the bore radius."""
        verts = self._vertices()
        z0, r0 = verts[0][0], verts[0][1]
        z1 = verts[-1][0]
        prof = Profile('bellows').start(z0, 0.0).line_to(z0, r0, 'PMC')
        emit_rounded_wall(prof, verts, 'PEC')
        prof.line_to(z1, 0.0, 'PMC')
        return self._chained(prof.close('AXI'))
