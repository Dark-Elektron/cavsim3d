"""Beam pipe and beam-line absorber (cavsim2d's ``Beampipe`` / ``BLA``)."""

from typing import Optional

from .base import AxisymmetricGeometry
from .profile import Profile

#: The end-face tags a pipe accepts: an open aperture (a port) or a metal plate.
_END_TAGS = {'pmc': 'PMC', 'pec': 'PEC'}


def _parse_ends(ends):
    """Normalise *ends* to a ``('PMC'|'PEC', 'PMC'|'PEC')`` pair."""
    if isinstance(ends, str):
        ends = (ends, ends)
    try:
        left, right = ends
    except (TypeError, ValueError):
        raise ValueError(f"ends must be 'pmc', 'pec' or a (left, right) pair, got {ends!r}.")
    out = []
    for side, want in (('left', left), ('right', right)):
        key = str(want).lower()
        if key not in _END_TAGS:
            raise ValueError(f"unknown {side} end {want!r}; use 'pmc' or 'pec'.")
        out.append(_END_TAGS[key])
    return tuple(out)


class Beampipe(AxisymmetricGeometry):
    """A straight circular beam pipe of radius *R* and length *L*.

    Centred on ``z = 0``. Unlike :class:`~cavsim3d.geometry.primitives.CircularWaveguide`
    (metres, from ``z = 0``), its dimensions follow cavsim2d (millimetres by
    default), and either end can be closed.

    Parameters
    ----------
    R, L : float
        Pipe radius and length.
    ends : {'pmc', 'pec'} or (str, str)
        ``'pmc'`` (default) makes an end an open aperture, i.e. a port;
        ``'pec'`` closes it with a metal plate. A pair sets the two ends.
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

    def __init__(self, R: Optional[float] = None, L: Optional[float] = None, ends='pmc',
                 chain: int = 1, spacing=None, unit: str = 'mm', maxh: float = 0.05,
                 config: Optional[dict] = None):
        super().__init__()
        self._require(R=R, L=L)
        self._set_unit(unit)
        if float(R) <= 0 or float(L) <= 0:
            raise ValueError(f'R and L must be positive, got R={R}, L={L} ({self.unit}).')
        self.ends = _parse_ends(ends)
        self.chain = int(chain)
        self.spacing = spacing
        self.parameters = {'R': float(R), 'L': float(L)}
        self._finish_init(maxh, dict(R=float(R), L=float(L),
                                     ends=[e.lower() for e in self.ends],
                                     chain=self.chain, spacing=spacing, maxh=maxh))

    def profile(self) -> Profile:
        """The meridian in metres: a rectangle in the (z, r) half-plane."""
        R = self.parameters['R'] * self._s
        half = self.parameters['L'] * self._s / 2.0
        left, right = self.ends
        return self._chained(Profile('beampipe')
                             .start(-half, 0.0)
                             .line_to(-half, R, left)       # left end face
                             .line_to(half, R, 'PEC')       # barrel
                             .line_to(half, 0.0, right)     # right end face
                             .close('AXI'))


#: Runs shorter than this (metres) are not emitted.
_RUN_TOL = 1e-12


class BLA(AxisymmetricGeometry):
    """A beam-line absorber: a beam pipe with a lossy ring set into its wall.

    The wall steps outward over ``absorber_length`` to hold a dielectric ring
    from ``r = R`` to ``R + absorber_thickness``, so the beam still sees the
    clear bore *R*. The ring is a second solid of material *material*, glued
    conformally to the vacuum; both form one domain for the solver.

    Parameters
    ----------
    R, L : float
        Bore radius and overall length.
    absorber_length : float
        Axial length of the ring; at most *L*.
    absorber_thickness : float
        Radial thickness of the ring, measured outward from the bore.
    eps_r, tan_delta : float
        Relative permittivity and loss tangent of the ring (default 10, 0.3).
        ``geo.set_materials({material: {...}})`` overrides them.
    z_absorber : float
        Axial centre of the ring relative to the element centre (default 0).
    material : str
        Mesh material name of the ring (default ``'absorber'``).
    absorber_maxh : float, optional
        Mesh size inside the ring in **metres**; set it when the ring is thin
        compared with ``maxh``.
    ends : {'pmc', 'pec'} or (str, str)
        End faces, as for :class:`Beampipe`.
    unit : {'mm', 'm', 'cm', 'um'}
        Unit of every length above (default ``'mm'``).
    maxh : float
        Mesh size in **metres**, used when the mesh is generated.
    config : dict, optional
        The arguments above as one dict; explicit arguments take precedence.
    """

    def __init__(self, R: Optional[float] = None, L: Optional[float] = None,
                 absorber_length: Optional[float] = None,
                 absorber_thickness: Optional[float] = None, eps_r: float = 10.0,
                 tan_delta: float = 0.3, z_absorber: float = 0.0,
                 material: str = 'absorber', absorber_maxh: Optional[float] = None,
                 ends='pmc', unit: str = 'mm', maxh: float = 0.05,
                 config: Optional[dict] = None):
        super().__init__()
        self._require(R=R, L=L, absorber_length=absorber_length, absorber_thickness=absorber_thickness)
        self._set_unit(unit)
        R, L = float(R), float(L)
        L_abs, t_abs, z_c = float(absorber_length), float(absorber_thickness), float(z_absorber)
        if R <= 0 or L <= 0:
            raise ValueError(f'R and L must be positive, got R={R}, L={L} ({self.unit}).')
        if not 0 < L_abs <= L + 1e-9:
            raise ValueError(f'absorber_length must be in (0, L={L}] {self.unit}, got {absorber_length!r}.')
        if t_abs <= 0:
            raise ValueError(f'absorber_thickness must be positive, got {absorber_thickness!r}.')
        if z_c - L_abs / 2 < -L / 2 - 1e-9 or z_c + L_abs / 2 > L / 2 + 1e-9:
            raise ValueError(
                f'the absorber ring (z = {z_c - L_abs / 2:.4g} to {z_c + L_abs / 2:.4g} {self.unit}) '
                f'does not fit inside the element (z = {-L / 2:.4g} to {L / 2:.4g} {self.unit}).')
        if float(eps_r) <= 0 or float(tan_delta) < 0:
            raise ValueError(f'need eps_r > 0 and tan_delta >= 0, got {eps_r}, {tan_delta}.')
        self.ends = _parse_ends(ends)
        self.material = str(material)
        self.eps_r = float(eps_r)
        self.tan_delta = float(tan_delta)
        self.absorber_maxh = absorber_maxh
        self.parameters = {'R': R, 'L': L, 'absorber_length': L_abs,
                           'absorber_thickness': t_abs, 'z_absorber': z_c}
        # Vacuum and ring are two mesh materials of ONE physical domain.
        self._domain_materials = {'bla': [Profile.default_material, self.material]}
        self._finish_init(maxh, dict(
            R=R, L=L, absorber_length=L_abs, absorber_thickness=t_abs,
            eps_r=self.eps_r, tan_delta=self.tan_delta, z_absorber=z_c,
            material=self.material, absorber_maxh=absorber_maxh,
            ends=[e.lower() for e in self.ends], maxh=maxh))

    @property
    def absorber_span(self):
        """``(z_lo, z_hi)`` of the ring in element coordinates, in ``unit``."""
        p = self.parameters
        return (p['z_absorber'] - p['absorber_length'] / 2.0,
                p['z_absorber'] + p['absorber_length'] / 2.0)

    def profile(self) -> Profile:
        """The meridian in metres, with the ring as a material region."""
        p = self.parameters
        R = p['R'] * self._s
        half = p['L'] * self._s / 2.0
        R_out = R + p['absorber_thickness'] * self._s
        z_lo, z_hi = (v * self._s for v in self.absorber_span)
        left, right = self.ends

        prof = Profile('bla').start(-half, 0.0).line_to(-half, R, left)
        if z_lo + half > _RUN_TOL:
            prof.line_to(z_lo, R, 'PEC')            # bore before the recess
        prof.line_to(z_lo, R_out, 'PEC')            # step out into the wall
        prof.line_to(z_hi, R_out, 'PEC')            # over the ring
        prof.line_to(z_hi, R, 'PEC')                # step back to the bore
        if half - z_hi > _RUN_TOL:
            prof.line_to(half, R, 'PEC')            # bore after the recess
        prof.line_to(half, 0.0, right)
        prof.close('AXI')
        prof.add_region(self.material, z=(z_lo, z_hi), r=(R, R_out))
        return prof

    def build(self) -> None:
        super().build()
        if self.absorber_maxh:
            for solid in self.geo.solids:
                if solid.name == self.material:
                    solid.maxh = float(self.absorber_maxh)

    def get_material(self, domain_name: str) -> dict:
        """The ring's constructor properties unless ``set_materials`` overrides them."""
        props = self._lookup_material(domain_name)
        if props is not None:
            return props
        name = domain_name.split('/', 1)[-1]
        if name == self.material:
            return {**self.MATERIAL_DEFAULTS, 'eps_r': self.eps_r, 'tan_delta': self.tan_delta}
        return dict(self.MATERIAL_DEFAULTS)
