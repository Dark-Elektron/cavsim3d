"""Microstrip transmission line geometry with quasi-TEM ports.

The strip and ground plane are modelled as PEC (their volumes are subtracted
from the air box, so their surfaces become PEC boundaries).  The meshed volume
is the dielectric substrate (FR-4) plus the surrounding air.  Each end face is
split by material into ``portN_substrate`` / ``portN_air`` — a composite,
inhomogeneous quasi-TEM port (see :meth:`FrequencyDomainSolver.solve`'s
``qtem_ports`` option).  The conductor outlines on the port planes are named
``ground_edges`` and ``microstrip_edges`` for the port solver's
``dirichlet_bbnd``.

Default dimensions reproduce the CST ``microstrip_line`` reference model.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from netgen.occ import Rectangle, X, Y, Z, Glue, OCCGeometry
from ngsolve import Mesh

from .base import BaseGeometry


class MicrostripLine(BaseGeometry):
    """PEC-strip microstrip line with inhomogeneous quasi-TEM ports.

    Parameters
    ----------
    L : float
        Line length along the propagation axis (Z) [m].
    W : float
        Box / substrate width (X) [m].
    w : float
        Strip width (X) [m].
    h : float
        Substrate height (Y) [m].
    t : float
        Conductor (strip / ground plate) thickness (Y) [m].
    air_height : float
        Air region height above the substrate, as a multiple of ``h``.
    eps_r : float
        Substrate relative permittivity.
    substrate_name : str
        Mesh material name for the substrate.
    maxh : float
        Maximum mesh element size [m].
    """

    def __init__(
        self,
        L: float = 40e-3,
        W: float = 20e-3,
        w: float = 3.1e-3,
        h: float = 1.6e-3,
        t: float = 0.5e-3,
        air_height: float = 5.0,
        eps_r: float = 4.3,
        substrate_name: str = 'FR-4',
        maxh: float = 1.2e-3,
    ):
        super().__init__()
        self.L, self.W, self.w, self.h, self.t = L, W, w, h, t
        self.air_height = air_height
        self.eps_r = eps_r
        self.substrate_name = substrate_name
        self.maxh = maxh

        # Conductor edge groups on the port planes (for qTEM dirichlet_bbnd)
        self.qtem_conductor_bbnd = 'microstrip_edges|ground_edges'

        # The substrate and air are two mesh materials of ONE physical domain,
        # not a coupled multi-solid structure — declare them as a single domain
        # so the solver treats the ends as external ports (not internal
        # interfaces) and assembles one coupled global system.
        self._domain_materials = {'microstrip': ['air', substrate_name]}

        self.build()
        self.generate_mesh(maxh=maxh)
        self._record('__init__', L=L, W=W, w=w, h=h, t=t,
                     air_height=air_height, eps_r=eps_r, maxh=maxh)

    # ------------------------------------------------------------------
    def build(self) -> None:
        L, W, w, h, t = self.L, self.W, self.w, self.h, self.t
        tol = 1e-6

        def _box(wx, hy, x0, y0):
            return Rectangle(wx, hy).Face().Extrude(L).Move((x0, y0, -L / 2))

        ground_plate = _box(W, t, -W / 2, 0)            # y: 0 .. t   (PEC, removed)
        substrate = _box(W, h, -W / 2, t)               # y: t .. t+h (dielectric)
        strip = _box(w, t, -w / 2, t + h)               # strip       (PEC, removed)
        background = _box(W, self.air_height * h, -W / 2, 0)

        # Meshed volume = dielectric substrate + surrounding air (conductors removed)
        surrounding = background - ground_plate - substrate - strip
        substrate.mat(self.substrate_name)
        surrounding.mat('air')
        geo = Glue([surrounding, substrate])

        # ---- port faces (split by material) ----
        for f in geo.faces:
            c = f.center
            bb = f.bounding_box
            if abs(bb[1][2] - bb[0][2]) < tol and abs(abs(c[2]) - L / 2) < 1e-4:
                pnum = 1 if c[2] < 0 else 2
                f.name = (f"port{pnum}_substrate" if c[1] < t + h - tol
                          else f"port{pnum}_air")
                f.col = (1, 0, 0)

        # ---- PEC faces: ground interface, strip footprint, outer walls ----
        for f in geo.faces:
            if f.name and 'port' in f.name:
                continue
            c = f.center
            bb = f.bounding_box
            xmin, ymin = bb[0][0], bb[0][1]
            xmax, ymax = bb[1][0], bb[1][1]
            if abs(ymax - ymin) < tol and abs(c[1] - t) < 1e-4:
                f.name = 'ground'
            elif (xmax <= w / 2 + tol and xmin >= -w / 2 - tol
                  and ymin >= t + h - tol and ymax <= t + h + t + tol):
                f.name = 'strip'
            else:
                f.name = 'walls'

        # ---- conductor edges on the port planes (qTEM dirichlet_bbnd) ----
        for e in geo.edges:
            bb = e.bounding_box
            c = e.center
            if not (abs(bb[1][2] - bb[0][2]) < tol and abs(abs(c[2]) - L / 2) < 1e-4):
                continue
            xmin, ymin = bb[0][0], bb[0][1]
            xmax, ymax = bb[1][0], bb[1][1]
            if abs(ymax - ymin) < tol and abs(c[1] - t) < 1e-4:
                e.name = 'ground_edges'
            elif ymin >= t + h - tol and xmax <= w / 2 + tol and xmin >= -w / 2 - tol:
                e.name = 'microstrip_edges'

        self.geo = geo
        # PEC on the conductors only.  The outer air-box faces ('walls') are
        # left as the natural (PMC-like) boundary — matching the quasi-TEM port
        # mode solve, whose port-plane wall edges are also natural.  Making the
        # walls PEC in 3D while the port mode leaves them free is inconsistent
        # and reflects power (spurious S11).
        self.bc = 'ground|strip'
        self._bc_explicitly_set = True
        self.invalidate_tag()

    # ------------------------------------------------------------------
    def get_material(self, domain_name: str) -> dict:
        """Material properties per mesh material (defaults to vacuum)."""
        defaults = {"eps_r": 1.0, "mu_r": 1.0, "sigma": 0.0, "tan_delta": 0.0}
        name = domain_name.split('/', 1)[-1] if '/' in domain_name else domain_name
        if name == self.substrate_name:
            return {**defaults, "eps_r": self.eps_r}
        return defaults

    def qtem_voltage_path(self, port: str = 'port1'):
        """Default ground->strip voltage integration path for Z_PV.

        Returns two 3D points (ground-side, strip-side) at the port plane,
        along the strip centreline through the substrate.
        """
        z = -self.L / 2 if str(port).endswith('1') else self.L / 2
        p_ground = (0.0, self.t, z)          # top of ground plate
        p_strip = (0.0, self.t + self.h, z)  # bottom of strip
        return p_ground, p_strip

    # ------------------------------------------------------------------
    def _get_geometry_params(self) -> Dict[str, Any]:
        return {
            'class': 'MicrostripLine',
            'L': float(self.L), 'W': float(self.W), 'w': float(self.w),
            'h': float(self.h), 't': float(self.t),
            'air_height': float(self.air_height), 'eps_r': float(self.eps_r),
            'substrate_name': self.substrate_name, 'bc': self.bc,
        }

    def _get_mesh_params(self) -> Dict[str, Any]:
        return {'maxh': self.maxh, 'nv': self.mesh.nv if self.mesh else None}
