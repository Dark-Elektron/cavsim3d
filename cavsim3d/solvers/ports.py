"""
Port eigenmode solver with analytic and numeric mode computation.

Analytic modes use exact mathematical formulas mapped to GridFunction.
The phase is deterministic from the formula - no mesh-dependent normalization.

OPTIMIZED VERSION (safe precomputation of boundary mass matrix)
"""

import platform
import re
from typing import Dict, Tuple, List, Optional, Literal, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
import scipy.sparse as sp
from scipy.special import jv, yv, jvp, yvp, jn_zeros, jnp_zeros
import numpy.polynomial.chebyshev as cheb

from ngsolve import (
    HCurl, BilinearForm, GridFunction, BND, Cross, Integrate, InnerProduct,
    TaskManager, Preconditioner, solvers, IdentityMatrix, curl, ds,
    CoefficientFunction, specialcf, x, y, z, sin, cos, sqrt, pi,
    H1, grad, dx as dx_vol, ArnoldiSolver
)

from cavsim3d.core.constants import c0, mu0, eps0, Z0
import cavsim3d.utils.printing as pr

# PARDISO is not available on macOS; fall back to UMFPACK
_DIRECT_SOLVER = "umfpack" if platform.system() == "Darwin" else "pardiso"


def make_analytic_port_impedance(params: dict):
    """Rebuild a standalone port wave-impedance function from persisted params.

    Mirrors :meth:`PortEigenmodeSolver.get_port_wave_impedance` exactly, so a
    reloaded reduced model (imported from disk, with no live solver) produces
    identical Z->S.  ``params`` = {'cutoff': {port:{mode:kc}}, 'mtype':
    {port:{mode:'TE'|'TM'|'TEM'}}, 'eps': {port:eps_r}}.
    """
    cutoff = {p: {int(m): v for m, v in d.items()}
              for p, d in params.get('cutoff', {}).items()}
    mtype = {p: {int(m): t for m, t in d.items()}
             for p, d in params.get('mtype', {}).items()}
    eps = dict(params.get('eps', {}))
    zpv = {p: {int(m): complex(v) for m, v in d.items()}
           for p, d in params.get('zpv', {}).items()}

    def impedance(port: str, mode: int, freq: float) -> complex:
        kc = cutoff[port][int(mode)]
        mt = mtype[port][int(mode)]
        eta = Z0 / np.sqrt(eps.get(port, 1.0))
        # Quasi-TEM ports renormalise to their stored power-voltage line impedance.
        if mt == 'qTEM':
            zli = zpv.get(port, {}).get(int(mode))
            return complex(zli) if zli is not None and np.isfinite(zli) else complex(eta)
        if mt == 'TEM':
            return complex(eta)
        wc = kc * c0
        s = 1j * 2 * np.pi * freq
        sqrt_term = np.sqrt(s ** 2 + wc ** 2)
        if mt == 'TE':
            return complex(s * eta / sqrt_term)
        return complex(eta * sqrt_term / s)

    return impedance

def logical_port_name(face_name: str) -> str:
    """Logical port a boundary face belongs to (``port1_substrate`` -> ``port1``).

    Faces sharing a leading ``port<N>`` token form one composite port (e.g. an
    inhomogeneous quasi-TEM microstrip port split by material).  Names without
    that pattern are returned unchanged.
    """
    m = re.match(r'(port\d+)', str(face_name), re.IGNORECASE)
    return m.group(1) if m else str(face_name)


def group_port_faces(boundary_names) -> Dict[str, str]:
    """Map each logical port to a ``|``-joined mesh-region string of its faces.

    ``['port1_substrate','port1_air','port2_substrate','port2_air']`` ->
    ``{'port1': 'port1_air|port1_substrate', 'port2': ...}``.  A simple port
    ``'port1'`` maps to itself, so callers can always resolve a logical port to
    a region usable in ``mesh.Boundaries(...)`` / ``ds(...)``.
    """
    groups: Dict[str, List[str]] = {}
    for b in boundary_names:
        if not b or 'port' not in b.lower():
            continue
        lp = logical_port_name(b)
        groups.setdefault(lp, [])
        if b not in groups[lp]:
            groups[lp].append(b)
    return {lp: '|'.join(sorted(faces)) for lp, faces in groups.items()}


def sorted_logical_ports(region_map: Dict[str, str]) -> List[str]:
    """Logical port names sorted by their numeric index (``port2`` after ``port1``)."""
    def _num(p):
        digits = ''.join(filter(str.isdigit, p))
        return int(digits) if digits else 0
    return sorted(region_map.keys(), key=_num)


class PortGeometryType(Enum):
    """Supported port geometry types."""
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    COAXIAL = "coaxial"
    UNKNOWN = "unknown"


@dataclass
class PortGeometry:
    """Port geometry information."""
    type: PortGeometryType
    center: np.ndarray
    normal: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    area: float
    a: Optional[float] = None
    b: Optional[float] = None
    radius: Optional[float] = None
    inner_radius: Optional[float] = None  # inner radius for coaxial ports
    fit_error: float = 0.0


@dataclass
class AnalyticMode:
    """Analytic mode definition."""
    type: str
    indices: Tuple[int, int]
    kc: float
    polarization: Optional[str] = None
    degeneracy: int = 1


class PortEigenmodeSolver:
    """
    Port eigenmode solver with analytic or numeric computation.
    """

    def __init__(
        self,
        mesh,
        order: int = 3,
        bc: str = 'default',
        mode_source: Literal['analytic', 'numeric'] = 'analytic',
        mode_source_internal: Literal['analytic', 'numeric'] = 'analytic',
        geometry_tolerance: float = 0.05,
        polarization_angle: float = 0.0,
        global_up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        propagation_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        ensure_inward_power: bool = True
    ):
        self.mesh = mesh
        self.order = order
        self.bc = bc
        self.mode_source = mode_source
        self.mode_source_internal = mode_source_internal
        self.geometry_tolerance = geometry_tolerance
        self.polarization_angle = polarization_angle
        self.ensure_inward_power = ensure_inward_power

        self.global_up = np.array(global_up, dtype=float)
        self.global_up /= np.linalg.norm(self.global_up)

        self.propagation_axis = np.array(propagation_axis, dtype=float)
        self.propagation_axis /= np.linalg.norm(self.propagation_axis)

        # Output storage
        self.port_modes: Dict[str, Dict[int, GridFunction]] = {}
        self.port_basis: Dict[str, Dict[int, np.ndarray]] = {}
        self.port_cutoff_kc: Dict[str, Dict[int, float]] = {}
        self.port_cutoff_frequencies: Dict[str, Dict[int, float]] = {}
        self.port_mode_types: Dict[str, Dict[int, str]] = {}

        # Geometric information
        self.port_normals: Dict[str, np.ndarray] = {}
        self.port_polarizations: Dict[str, np.ndarray] = {}
        self.port_orientation_factors: Dict[str, float] = {}
        self.port_tangent_frames: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.port_geometries: Dict[str, PortGeometry] = {}

        # Relative permittivity of the medium filling each port (default
        # vacuum).  Populated by the frequency-domain solver from the
        # material adjacent to each port face; used to scale the medium wave
        # impedance eta = eta0 / sqrt(eps_r).
        self.port_media_eps: Dict[str, float] = {}

        # Phase and polarization tracking
        self.port_phase_signs: Dict[str, Dict[int, float]] = {}
        self.port_mode_polarizations: Dict[str, Dict[int, float]] = {}
        self.port_mode_degeneracies: Dict[str, Dict[int, int]] = {}
        self.port_mode_indices: Dict[str, Dict[int, Tuple[int, int]]] = {}

        # OPTIMIZATION: precomputed mass matrices + keep BilinearForm alive
        self.port_mass_matrices: Dict[str, sp.csr_matrix] = {}
        self.port_mass_forms: Dict[str, BilinearForm] = {}

        # Composite-port support: logical port -> mesh-region string of its
        # faces (e.g. 'port1' -> 'port1_air|port1_substrate').  Empty for the
        # common case where each port is a single boundary.
        self.port_face_region: Dict[str, str] = {}

        # Quasi-TEM (inhomogeneous) port data.  Populated by _solve_port_qtem;
        # beta (propagation constant), eps_eff (effective permittivity), and
        # the power-voltage characteristic (line) impedance used as the S-param
        # renormalisation reference for qTEM modes.
        self.port_beta: Dict[str, Dict[int, complex]] = {}
        self.port_eps_eff: Dict[str, Dict[int, complex]] = {}
        self.port_line_impedance: Dict[str, Dict[int, complex]] = {}

    def _region(self, port: str) -> str:
        """Resolve a logical port to a mesh-region string (identity if simple)."""
        return self.port_face_region.get(port, port)

    # =========================================================================
    # Geometry Detection (unchanged)
    # =========================================================================

    def _compute_port_normal(self, port: str) -> np.ndarray:
        n = specialcf.normal(self.mesh.dim)
        port_region = self.mesh.Boundaries(self._region(port))

        nx = Integrate(n[0], self.mesh, BND, definedon=port_region)
        ny = Integrate(n[1], self.mesh, BND, definedon=port_region)
        nz = Integrate(n[2], self.mesh, BND, definedon=port_region)

        normal = np.array([nx, ny, nz])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            raise ValueError(f"Could not determine normal for port {port}")
        return normal / norm

    def _compute_tangent_frame(self, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g = self.global_up.copy()
        t1 = g - np.dot(g, n) * n
        if np.linalg.norm(t1) < 1e-12:
            g = np.array([1.0, 0.0, 0.0])
            t1 = g - np.dot(g, n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2)
        return t1, t2

    def _compute_port_centroid_and_area(self, port: str) -> Tuple[np.ndarray, float]:
        port_region = self.mesh.Boundaries(self._region(port))
        area = float(Integrate(CoefficientFunction(1.0), self.mesh, BND, definedon=port_region))
        if area < 1e-14:
            raise ValueError(f"Port '{port}' has near-zero area")
        cx = float(Integrate(x, self.mesh, BND, definedon=port_region)) / area
        cy = float(Integrate(y, self.mesh, BND, definedon=port_region)) / area
        cz = float(Integrate(z, self.mesh, BND, definedon=port_region)) / area
        return np.array([cx, cy, cz]), area

    def _compute_orientation_factor(self, port_normal: np.ndarray) -> float:
        inward = -port_normal
        dot = np.dot(inward, self.propagation_axis)
        if np.abs(dot) < 0.1:
            return 1.0
        return 1.0 #np.sign(dot) <- checkthe orientation later

    def _compute_polarization_vector(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        theta = self.polarization_angle
        return np.cos(theta) * t1 + np.sin(theta) * t2

    def _fit_circular(self, center, normal, t1, t2, area, I_uu, I_vv, I_uv):
        """
        Fit circular geometry with improved tolerance handling.
        
        For circular cross-sections (like SRF cavity irises):
        - I_uu and I_vv should be equal (isotropy)
        - I_uv should be zero (centered)
        - I_uu = I_vv = πR⁴/4 for a circle of radius R
        """
        R = np.sqrt(area / np.pi)
        I_expected = np.pi * R**4 / 4

        if I_uu + I_vv < 1e-20:
            return PortGeometry(
                type=PortGeometryType.CIRCULAR, 
                center=center,
                normal=normal, t1=t1, t2=t2, area=area,
                radius=R, fit_error=1.0
            ), 1.0

        # Isotropy check: I_uu ≈ I_vv for a circle
        I_sum = I_uu + I_vv
        isotropy_error = abs(I_uu - I_vv) / I_sum * 2
        
        # Magnitude check: I_avg ≈ πR⁴/4
        I_avg = I_sum / 2
        
        # For a revolved geometry, the actual moment might differ slightly
        # due to mesh discretization. Use a more robust check.
        if I_expected > 1e-20:
            magnitude_error = abs(I_avg - I_expected) / I_expected
        else:
            magnitude_error = 1.0
        
        # Cross-moment check: I_uv ≈ 0 for centered circle
        if I_avg > 1e-20:
            cross_error = abs(I_uv) / I_avg
        else:
            cross_error = 0.0

        # Weighted error - prioritize isotropy for circles
        # A circle MUST be isotropic, but magnitude can vary with mesh
        total_error = (
            isotropy_error * 1.0 +      # Most important for circle
            magnitude_error * 0.3 +      # Less important (mesh-dependent)
            cross_error * 0.5            # Moderate importance
        ) / 1.8  # Normalize

        return PortGeometry(
            type=PortGeometryType.CIRCULAR,
            center=center, normal=normal, t1=t1, t2=t2, area=area,
            radius=R, fit_error=total_error
        ), total_error

    def _fit_coaxial(self, center, normal, t1, t2, area, I_uu, I_vv, I_uv):
        """
        Fit coaxial (annular) geometry.

        For an annular ring with inner radius *a* and outer radius *b*:
          Area  = π (b² - a²)
          I_uu = I_vv = π (b⁴ - a⁴) / 4   (isotropic, like a circle)

        We detect a coaxial port by noticing that:
          1. The cross-section is isotropic (I_uu ≈ I_vv) — same as a circle.
          2. BUT the area / moment relationship does NOT match a solid circle.
             For a solid circle: I / Area² = 1 / (4π).
             For an annulus:     I / Area² = (b² + a²) / (4π(b² - a²))  > 1/(4π).

        So when the circular fit passes the isotropy check but fails the
        magnitude check, the port is likely coaxial.
        """
        if I_uu + I_vv < 1e-20:
            return None, 1.0

        I_sum = I_uu + I_vv
        I_avg = I_sum / 2

        # Must be isotropic
        isotropy_error = abs(I_uu - I_vv) / I_sum * 2
        if isotropy_error > 0.15:
            return None, 1.0

        # Solve for a and b from area and moment:
        #   Area = π (b² - a²)
        #   I    = π (b⁴ - a⁴) / 4 = π (b² - a²)(b² + a²) / 4
        #        = Area * (b² + a²) / 4
        # So:
        #   b² + a² = 4 * I / Area
        #   b² - a² = Area / π
        sum_sq = 4 * I_avg / area        # b² + a²
        diff_sq = area / np.pi           # b² - a²

        b_sq = (sum_sq + diff_sq) / 2
        a_sq = (sum_sq - diff_sq) / 2

        if a_sq <= 0 or b_sq <= 0 or a_sq >= b_sq:
            return None, 1.0

        a = np.sqrt(a_sq)
        b = np.sqrt(b_sq)

        # Reject if inner radius is negligible — this is a solid circle,
        # not a coaxial port (mesh noise can produce tiny positive a_sq).
        if a / b < 0.02:
            return None, 1.0

        # Validate: area and moment should be consistent
        area_check = np.pi * (b**2 - a**2)
        area_error = abs(area_check - area) / area

        I_check = np.pi * (b**4 - a**4) / 4
        moment_error = abs(I_check - I_avg) / I_avg if I_avg > 1e-20 else 1.0

        # Cross-moment should be near zero
        cross_error = abs(I_uv) / I_avg if I_avg > 1e-20 else 0.0

        total_error = (
            isotropy_error * 1.0 +
            area_error * 0.5 +
            moment_error * 0.3 +
            cross_error * 0.5
        ) / 2.3

        geom = PortGeometry(
            type=PortGeometryType.COAXIAL,
            center=center, normal=normal, t1=t1, t2=t2, area=area,
            radius=b, inner_radius=a, fit_error=total_error
        )
        return geom, total_error

    def _fit_rectangular(self, center, normal, t1, t2, area, I_uu, I_vv, I_uv):
        """
        Fit rectangular geometry.
        
        For a rectangle with sides a (along t1) and b (along t2):
        - I_vv = a³b/12 (moment about t1 axis)
        - I_uu = ab³/12 (moment about t2 axis)
        - Area = ab
        """
        if I_uu < 1e-20 or I_vv < 1e-20:
            return PortGeometry(
                type=PortGeometryType.RECTANGULAR, 
                center=center,
                normal=normal, t1=t1, t2=t2, area=area, 
                fit_error=1.0
            ), 1.0

        # From I = (side_perp)² * Area / 12:
        # b² = 12 * I_uu / Area
        # a² = 12 * I_vv / Area
        b_sq = 12 * I_uu / area
        a_sq = 12 * I_vv / area
        
        if b_sq < 0 or a_sq < 0:
            return PortGeometry(
                type=PortGeometryType.RECTANGULAR, 
                center=center,
                normal=normal, t1=t1, t2=t2, area=area, 
                fit_error=1.0
            ), 1.0

        b = np.sqrt(b_sq)
        a = np.sqrt(a_sq)
        
        # Check consistency
        area_error = abs(a * b - area) / area
        cross_error = abs(I_uv) / np.sqrt(I_uu * I_vv) if I_uu * I_vv > 0 else 0
        total_error = max(area_error, cross_error)

        # Ensure a >= b (a is the longer side)
        if b > a:
            a, b = b, a
            t1, t2 = t2, -t1

        return PortGeometry(
            type=PortGeometryType.RECTANGULAR,
            center=center, normal=normal, t1=t1, t2=t2, area=area,
            a=a, b=b, fit_error=total_error
        ), total_error

    def _detect_port_geometry(self, port: str) -> PortGeometry:
        """
        Detect port geometry type (rectangular, circular, or coaxial).

        Uses moments of inertia to distinguish shapes.
        Includes fallback logic for interface ports that may have
        slightly elevated fit errors due to mesh artifacts.
        """
        center, area = self._compute_port_centroid_and_area(port)
        normal = self._compute_port_normal(port)
        t1, t2 = self._compute_tangent_frame(normal)
        port_region = self.mesh.Boundaries(self._region(port))

        # Local coordinates on port plane
        u = (x - center[0]) * t1[0] + (y - center[1]) * t1[1] + (z - center[2]) * t1[2]
        v = (x - center[0]) * t2[0] + (y - center[1]) * t2[1] + (z - center[2]) * t2[2]

        # Compute second moments of area
        I_uu = float(Integrate(v * v, self.mesh, BND, definedon=port_region))
        I_vv = float(Integrate(u * u, self.mesh, BND, definedon=port_region))
        I_uv = float(Integrate(u * v, self.mesh, BND, definedon=port_region))

        # Try all fits
        rect_geom, rect_error = self._fit_rectangular(
            center, normal, t1, t2, area, I_uu, I_vv, I_uv
        )
        circ_geom, circ_error = self._fit_circular(
            center, normal, t1, t2, area, I_uu, I_vv, I_uv
        )
        coax_result = self._fit_coaxial(
            center, normal, t1, t2, area, I_uu, I_vv, I_uv
        )
        coax_geom, coax_error = coax_result if coax_result[0] is not None else (None, 1.0)

        # Decision logic with tolerance
        tol = self.geometry_tolerance

        # Coaxial: isotropic like a circle but moment/area ratio is wrong
        # for a solid disk.  If coaxial fits well and circular does NOT,
        # prefer coaxial.  If both fit, the circular magnitude error will
        # be high while coaxial error will be low.
        if coax_geom is not None and coax_error <= tol:
            # Only prefer coaxial when circular magnitude is off
            if circ_error > tol:
                return coax_geom
            # Both pass — coaxial wins if it has a better fit
            if coax_error < circ_error:
                return coax_geom

        # Strong preference for the better fit
        if rect_error <= tol and circ_error <= tol:
            if rect_error < circ_error:
                return rect_geom
            else:
                return circ_geom

        if rect_error <= tol:
            return rect_geom

        if circ_error <= tol:
            return circ_geom

        # Try coaxial with relaxed tolerance
        if coax_geom is not None and coax_error <= tol * 3:
            return coax_geom

        # Neither passes strict tolerance - try relaxed tolerance
        relaxed_tol = tol * 3

        if circ_error <= relaxed_tol and circ_error < rect_error:
            return PortGeometry(
                type=PortGeometryType.CIRCULAR,
                center=center, normal=normal, t1=t1, t2=t2, area=area,
                radius=circ_geom.radius,
                fit_error=circ_error
            )

        if rect_error <= relaxed_tol:
            return PortGeometry(
                type=PortGeometryType.RECTANGULAR,
                center=center, normal=normal, t1=t1, t2=t2, area=area,
                a=rect_geom.a, b=rect_geom.b,
                fit_error=rect_error
            )
        
        # Last resort: if one error is significantly better, use it
        if circ_error < 0.5 and circ_error < rect_error * 0.5:
            return PortGeometry(
                type=PortGeometryType.CIRCULAR,
                center=center, normal=normal, t1=t1, t2=t2, area=area,
                radius=circ_geom.radius,
                fit_error=circ_error
            )
        
        if rect_error < 0.5 and rect_error < circ_error * 0.5:
            return rect_geom

        # Truly unknown
        return PortGeometry(
            type=PortGeometryType.UNKNOWN,
            center=center, normal=normal, t1=t1, t2=t2, area=area,
            fit_error=min(rect_error, circ_error)
        )

    # =========================================================================
    # Analytic Mode Generation (unchanged)
    # =========================================================================

    def _generate_rectangular_modes(self, geometry: PortGeometry, nmodes: int) -> List[AnalyticMode]:
        a, b = geometry.a, geometry.b
        modes = []
        max_index = int(np.ceil(np.sqrt(nmodes * 4))) + 3

        for m in range(max_index):
            for n in range(max_index):
                if m == 0 and n == 0:
                    continue
                kc = np.pi * np.sqrt((m / a)**2 + (n / b)**2)
                modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc, degeneracy=1))

        for m in range(1, max_index):
            for n in range(1, max_index):
                kc = np.pi * np.sqrt((m / a)**2 + (n / b)**2)
                modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc, degeneracy=1))

        modes.sort(key=lambda mode: mode.kc)
        return modes[:nmodes * 2 + 4]

    def _generate_coaxial_modes(self, geometry: PortGeometry, nmodes: int) -> List[AnalyticMode]:
        """
        Generate analytic modes for a coaxial (annular) waveguide.

        Mode spectrum:
          TEM: kc = 0  (fundamental, unique to multi-conductor)
          TE_mn: kc from  J'_m(kc*b)*Y'_m(kc*a) - J'_m(kc*a)*Y'_m(kc*b) = 0
          TM_mn: kc from  J_m(kc*b)*Y_m(kc*a) - J_m(kc*a)*Y_m(kc*b) = 0

        where a = inner radius, b = outer radius.
        """
        from scipy.special import jv, yv, jvp, yvp
        from scipy.optimize import brentq

        a = geometry.inner_radius
        b = geometry.radius
        modes = []

        # TEM mode — always the fundamental mode of a coaxial line
        modes.append(AnalyticMode(type='TEM', indices=(0, 0), kc=0.0, degeneracy=1))

        max_m = 5
        max_n = 5

        def _find_roots(f, kc_max, n_roots, n_search=500):
            """Find roots of f(kc) in (0, kc_max] by sign-change search."""
            roots = []
            kc_vals = np.linspace(1e-6, kc_max, n_search)
            f_vals = np.array([f(k) for k in kc_vals])
            for i in range(len(f_vals) - 1):
                if np.isfinite(f_vals[i]) and np.isfinite(f_vals[i + 1]):
                    if f_vals[i] * f_vals[i + 1] < 0:
                        try:
                            root = brentq(f, kc_vals[i], kc_vals[i + 1])
                            # Avoid duplicates
                            if not any(abs(root - r) < 1e-8 for r in roots):
                                roots.append(root)
                        except ValueError:
                            pass
                if len(roots) >= n_roots:
                    break
            return roots

        # Upper kc limit — enough to capture nmodes
        kc_max = 50.0 / a  # generous upper bound

        # TM modes: J_m(kc*b)*Y_m(kc*a) - J_m(kc*a)*Y_m(kc*b) = 0
        for m in range(max_m):
            def tm_dispersion(kc, m=m):
                return jv(m, kc * b) * yv(m, kc * a) - jv(m, kc * a) * yv(m, kc * b)

            roots = _find_roots(tm_dispersion, kc_max, max_n)
            for n, kc in enumerate(roots, 1):
                if m == 0:
                    modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc, degeneracy=1))
                else:
                    modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc,
                                              polarization='cos', degeneracy=2))
                    modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc,
                                              polarization='sin', degeneracy=2))

        # TE modes: J'_m(kc*b)*Y'_m(kc*a) - J'_m(kc*a)*Y'_m(kc*b) = 0
        for m in range(max_m):
            def te_dispersion(kc, m=m):
                return jvp(m, kc * b) * yvp(m, kc * a) - jvp(m, kc * a) * yvp(m, kc * b)

            roots = _find_roots(te_dispersion, kc_max, max_n)
            for n, kc in enumerate(roots, 1):
                if m == 0:
                    modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc, degeneracy=1))
                else:
                    modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc,
                                              polarization='cos', degeneracy=2))
                    modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc,
                                              polarization='sin', degeneracy=2))

        modes.sort(key=lambda mode: (mode.kc, mode.type, mode.polarization or ''))
        return modes[:nmodes * 2 + 4]

    def _generate_circular_modes(self, geometry: PortGeometry, nmodes: int) -> List[AnalyticMode]:

        R = geometry.radius
        modes = []
        max_m, max_n = 5, 5

        for m in range(max_m):
            try:
                zeros = jnp_zeros(m, max_n)
                for n, p_mn in enumerate(zeros, 1):
                    kc = float(p_mn) / R
                    if m == 0:
                        modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc, degeneracy=1))
                    else:
                        modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc,
                                                  polarization='cos', degeneracy=2))
                        modes.append(AnalyticMode(type='TE', indices=(m, n), kc=kc,
                                                  polarization='sin', degeneracy=2))
            except Exception:
                pass

        for m in range(max_m):
            try:
                zeros = jn_zeros(m, max_n)
                for n, p_mn in enumerate(zeros, 1):
                    kc = float(p_mn) / R
                    if m == 0:
                        modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc, degeneracy=1))
                    else:
                        modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc,
                                                  polarization='cos', degeneracy=2))
                        modes.append(AnalyticMode(type='TM', indices=(m, n), kc=kc,
                                                  polarization='sin', degeneracy=2))
            except Exception:
                pass

        modes.sort(key=lambda mode: (mode.kc, mode.polarization or ''))
        return modes[:nmodes * 2 + 4]

    def _create_rectangular_mode_cf(self, mode: AnalyticMode, geometry: PortGeometry) -> CoefficientFunction:
        a, b = geometry.a, geometry.b
        m, n = mode.indices
        center = geometry.center
        t1, t2 = geometry.t1, geometry.t2

        origin = center - (a / 2) * t1 - (b / 2) * t2

        u_cf = (x - origin[0]) * t1[0] + (y - origin[1]) * t1[1] + (z - origin[2]) * t1[2]
        v_cf = (x - origin[0]) * t2[0] + (y - origin[1]) * t2[1] + (z - origin[2]) * t2[2]

        if mode.type == 'TE':
            if m == 0:
                E_u = sin(n * pi * v_cf / b)
                E_v = CoefficientFunction(0.0)
            elif n == 0:
                E_u = CoefficientFunction(0.0)
                E_v = -sin(m * pi * u_cf / a)
            else:
                E_u = (n / b) * cos(m * pi * u_cf / a) * sin(n * pi * v_cf / b)
                E_v = -(m / a) * sin(m * pi * u_cf / a) * cos(n * pi * v_cf / b)
        else:
            E_u = (m / a) * cos(m * pi * u_cf / a) * sin(n * pi * v_cf / b)
            E_v = (n / b) * sin(m * pi * u_cf / a) * cos(n * pi * v_cf / b)

        E_x = E_u * t1[0] + E_v * t2[0]
        E_y = E_u * t1[1] + E_v * t2[1]
        E_z = E_u * t1[2] + E_v * t2[2]

        return CoefficientFunction((E_x, E_y, E_z))

    def _create_circular_mode_cf(self, mode: AnalyticMode, geometry: PortGeometry) -> CoefficientFunction:
        from scipy.special import jn_zeros, jnp_zeros

        R = geometry.radius
        center = geometry.center
        normal = geometry.normal
        t1, t2 = geometry.t1, geometry.t2
        m, n = mode.indices

        if mode.type == 'TE':
            p_mn = float(jnp_zeros(m, n)[n - 1])
        else:
            p_mn = float(jn_zeros(m, n)[n - 1])

        global_x = np.array([1.0, 0.0, 0.0])
        global_y = np.array([0.0, 1.0, 0.0])

        e1 = global_x - np.dot(global_x, normal) * normal
        e2 = global_y - np.dot(global_y, normal) * normal

        e1_norm = np.linalg.norm(e1)
        e2_norm = np.linalg.norm(e2)

        if e1_norm > 0.1:
            e1 = e1 / e1_norm
        else:
            e1 = global_y - np.dot(global_y, normal) * normal
            e1 = e1 / np.linalg.norm(e1)

        if e2_norm > 0.1:
            e2 = e2 / e2_norm
            e2 = e2 - np.dot(e2, e1) * e1
            e2_norm = np.linalg.norm(e2)
            if e2_norm > 0.1:
                e2 = e2 / e2_norm
            else:
                e2 = np.cross(normal, e1)
                e2 = e2 / np.linalg.norm(e2)
        else:
            e2 = np.cross(normal, e1)
            e2 = e2 / np.linalg.norm(e2)

        u_global = (x - center[0]) * e1[0] + (y - center[1]) * e1[1] + (z - center[2]) * e1[2]
        v_global = (x - center[0]) * e2[0] + (y - center[1]) * e2[1] + (z - center[2]) * e2[2]

        r_cf = sqrt(u_global * u_global + v_global * v_global + 1e-30)
        rho = r_cf / R

        J_m = self._bessel_chebyshev(m, p_mn * rho, p_mn)
        J_m_prime = self._bessel_prime_chebyshev(m, p_mn * rho, p_mn)

        cos_phi = u_global / r_cf
        sin_phi = v_global / r_cf
        cos_m_phi, sin_m_phi = self._trig_multiple_cf(cos_phi, sin_phi, m)

        if mode.polarization == 'sin':
            angular_1 = sin_m_phi
            angular_2 = -cos_m_phi
        else:
            angular_1 = cos_m_phi
            angular_2 = sin_m_phi

        if mode.type == 'TE':
            if m == 0:
                E_r = CoefficientFunction(0.0)
                E_phi = J_m_prime * (p_mn / R)
            else:
                E_r = (m / r_cf) * J_m * angular_2
                E_phi = (p_mn / R) * J_m_prime * angular_1
        else:
            if m == 0:
                E_r = J_m_prime * (p_mn / R)
                E_phi = CoefficientFunction(0.0)
            else:
                E_r = (p_mn / R) * J_m_prime * angular_1
                E_phi = -(m / r_cf) * J_m * angular_2

        E_e1 = E_r * cos_phi - E_phi * sin_phi
        E_e2 = E_r * sin_phi + E_phi * cos_phi

        E_x = E_e1 * e1[0] + E_e2 * e2[0]
        E_y = E_e1 * e1[1] + E_e2 * e2[1]
        E_z = E_e1 * e1[2] + E_e2 * e2[2]

        return CoefficientFunction((E_x, E_y, E_z))

    def _create_coaxial_mode_cf(self, mode: AnalyticMode, geometry: PortGeometry) -> CoefficientFunction:
        """
        Create CoefficientFunction for a coaxial waveguide mode.

        TEM mode: E_r = 1/r  (radial, azimuthally symmetric)
        TE/TM modes: Bessel combination  C1*J_m(kc*r) + C2*Y_m(kc*r)
        with boundary conditions at r=a and r=b.
        """

        a_inner = geometry.inner_radius
        b_outer = geometry.radius
        center = geometry.center
        normal = geometry.normal
        t1, t2 = geometry.t1, geometry.t2
        m_idx, n_idx = mode.indices

        # Use the port's own pre-computed tangent frame for the local
        # polar coordinate system.  These vectors are already orthonormal
        # and lie in the port plane, regardless of its orientation.
        e1 = t1
        e2 = t2

        u_global = (x - center[0]) * e1[0] + (y - center[1]) * e1[1] + (z - center[2]) * e1[2]
        v_global = (x - center[0]) * e2[0] + (y - center[1]) * e2[1] + (z - center[2]) * e2[2]
        r_cf = sqrt(u_global * u_global + v_global * v_global + 1e-30)
        cos_phi = u_global / r_cf
        sin_phi = v_global / r_cf


        if mode.type == 'TEM':
            # TEM: E_r = 1/r, E_phi = 0  (azimuthally symmetric)
            E_r = 1.0 / r_cf
            E_phi = CoefficientFunction(0.0)

            E_e1 = E_r * cos_phi
            E_e2 = E_r * sin_phi

            E_x = E_e1 * e1[0] + E_e2 * e2[0]
            E_y = E_e1 * e1[1] + E_e2 * e2[1]
            E_z = E_e1 * e1[2] + E_e2 * e2[2]

            return CoefficientFunction((E_x, E_y, E_z))

        # TE/TM modes: radial function is C1*J_m(kc*r) + C2*Y_m(kc*r)
        # with boundary conditions determining C1/C2 ratio.
        kc = mode.kc

        # Approximate the radial Bessel combination via Chebyshev interpolation
        # on the annular region [a, b].
        n_pts = 60
        r_pts = np.linspace(a_inner, b_outer, n_pts)

        if mode.type == 'TM':
            # BC: R(a)=0, R(b)=0  =>  C1*J_m(kc*a) + C2*Y_m(kc*a) = 0
            # => C2 = -C1 * J_m(kc*a) / Y_m(kc*a)
            Ym_a = yv(m_idx, kc * a_inner)
            if abs(Ym_a) < 1e-30:
                return CoefficientFunction((0.0, 0.0, 0.0))
            C1 = 1.0
            C2 = -C1 * jv(m_idx, kc * a_inner) / Ym_a

            R_vals = C1 * jv(m_idx, kc * r_pts) + C2 * yv(m_idx, kc * r_pts)
            Rp_vals = C1 * jvp(m_idx, kc * r_pts) * kc + C2 * yvp(m_idx, kc * r_pts) * kc
        else:  # TE
            # BC: R'(a)=0, R'(b)=0  =>  C1*J'_m(kc*a) + C2*Y'_m(kc*a) = 0
            Ymp_a = yvp(m_idx, kc * a_inner)
            if abs(Ymp_a) < 1e-30:
                return CoefficientFunction((0.0, 0.0, 0.0))
            C1 = 1.0
            C2 = -C1 * jvp(m_idx, kc * a_inner) / Ymp_a

            R_vals = C1 * jv(m_idx, kc * r_pts) + C2 * yv(m_idx, kc * r_pts)
            Rp_vals = C1 * jvp(m_idx, kc * r_pts) * kc + C2 * yvp(m_idx, kc * r_pts) * kc

        # Chebyshev fit on [a, b] mapped to [-1, 1]
        cheb_x = 2 * (r_pts - a_inner) / (b_outer - a_inner) - 1
        R_coeffs = cheb.chebfit(cheb_x, R_vals, min(20, n_pts - 1))
        Rp_coeffs = cheb.chebfit(cheb_x, Rp_vals, min(20, n_pts - 1))

        # Map r_cf to Chebyshev variable
        cheb_var = 2 * (r_cf - a_inner) / (b_outer - a_inner) - 1
        R_cf = self._eval_chebyshev_cf(cheb_var, R_coeffs)
        Rp_cf = self._eval_chebyshev_cf(cheb_var, Rp_coeffs)

        # Angular functions
        cos_m_phi, sin_m_phi = self._trig_multiple_cf(cos_phi, sin_phi, m_idx)

        if mode.polarization == 'sin':
            angular_1 = sin_m_phi
            angular_2 = -cos_m_phi
        else:
            angular_1 = cos_m_phi
            angular_2 = sin_m_phi

        if mode.type == 'TE':
            if m_idx == 0:
                E_r = CoefficientFunction(0.0)
                E_phi = Rp_cf
            else:
                E_r = (m_idx / r_cf) * R_cf * angular_2
                E_phi = Rp_cf * angular_1
        else:  # TM
            if m_idx == 0:
                E_r = Rp_cf
                E_phi = CoefficientFunction(0.0)
            else:
                E_r = Rp_cf * angular_1
                E_phi = -(m_idx / r_cf) * R_cf * angular_2

        E_e1 = E_r * cos_phi - E_phi * sin_phi
        E_e2 = E_r * sin_phi + E_phi * cos_phi

        E_x = E_e1 * e1[0] + E_e2 * e2[0]
        E_y = E_e1 * e1[1] + E_e2 * e2[1]
        E_z = E_e1 * e1[2] + E_e2 * e2[2]

        return CoefficientFunction((E_x, E_y, E_z))

    def _bessel_chebyshev(self, m: int, xi_cf: CoefficientFunction, xi_max: float) -> CoefficientFunction:
        from scipy.special import jv
        import numpy.polynomial.chebyshev as cheb

        n_pts = 50
        rho = np.linspace(0, 1, n_pts)
        xi = xi_max * rho
        J_vals = jv(m, xi)

        cheb_x = 2 * rho - 1
        coeffs = cheb.chebfit(cheb_x, J_vals, min(15, n_pts - 1))

        rho_cf = xi_cf / xi_max
        cheb_var = 2 * rho_cf - 1

        return self._eval_chebyshev_cf(cheb_var, coeffs)

    def _bessel_prime_chebyshev(self, m: int, xi_cf: CoefficientFunction, xi_max: float) -> CoefficientFunction:
        from scipy.special import jvp
        import numpy.polynomial.chebyshev as cheb

        n_pts = 50
        rho = np.linspace(0, 1, n_pts)
        xi = xi_max * rho
        Jp_vals = jvp(m, xi)

        cheb_x = 2 * rho - 1
        coeffs = cheb.chebfit(cheb_x, Jp_vals, min(15, n_pts - 1))

        rho_cf = xi_cf / xi_max
        cheb_var = 2 * rho_cf - 1

        return self._eval_chebyshev_cf(cheb_var, coeffs)

    def _eval_chebyshev_cf(self, x_cf: CoefficientFunction, coeffs: np.ndarray) -> CoefficientFunction:
        n = len(coeffs)
        if n == 0:
            return CoefficientFunction(0.0)
        if n == 1:
            return CoefficientFunction(float(coeffs[0]))

        b_k2 = CoefficientFunction(0.0)
        b_k1 = CoefficientFunction(0.0)

        for k in range(n - 1, 0, -1):
            b_k = float(coeffs[k]) + 2 * x_cf * b_k1 - b_k2
            b_k2 = b_k1
            b_k1 = b_k

        return float(coeffs[0]) + x_cf * b_k1 - b_k2

    def _trig_multiple_cf(self, cos_phi, sin_phi, m: int):
        if m == 0:
            return CoefficientFunction(1.0), CoefficientFunction(0.0)
        if m == 1:
            return cos_phi, sin_phi

        cos_prev2, sin_prev2 = CoefficientFunction(1.0), CoefficientFunction(0.0)
        cos_prev1, sin_prev1 = cos_phi, sin_phi

        for _ in range(2, m + 1):
            cos_curr = 2 * cos_phi * cos_prev1 - cos_prev2
            sin_curr = 2 * cos_phi * sin_prev1 - sin_prev2
            cos_prev2, sin_prev2 = cos_prev1, sin_prev1
            cos_prev1, sin_prev1 = cos_curr, sin_curr

        return cos_prev1, sin_prev1

    # =========================================================================
    # Main Solve Method – with safe precomputation
    # =========================================================================

    def solve(self, nmodes: Union[int, Dict[str, int]] = 1,
              internal_ports: Optional[List[str]] = None,
              qtem_ports: Optional[List[str]] = None,
              port_eps_bnd: Optional[Dict[str, 'CoefficientFunction']] = None,
              port_conductor_bbnd: Optional[Dict[str, str]] = None,
              k0_ref: Optional[float] = None,
              port_voltage_path: Optional[Dict[str, Tuple]] = None,
              port_eps_max: Optional[Dict[str, float]] = None,
              ) -> Tuple[Dict, Dict]:
        pr.running("Calculating Port Eigenmodes...")
        pr.info("=" * 60)
        pr.info(f"  Mode source: {self.mode_source}")
        pr.info(f"  Polarization angle: {np.degrees(self.polarization_angle):.1f}°")
        pr.info(f"  Requested modes per port: {nmodes}")
        pr.info("-" * 60)

        # Group boundary faces into logical ports (composite / inhomogeneous
        # qTEM ports split by material collapse to a single logical port).
        self.port_face_region = group_port_faces(self.mesh.GetBoundaries())
        ports = sorted_logical_ports(self.port_face_region)

        if not ports:
            raise ValueError("No ports found in mesh")

        _qtem = set(qtem_ports or [])
        _eps_bnd = dict(port_eps_bnd or {})
        _cond_bbnd = dict(port_conductor_bbnd or {})
        _vpath = dict(port_voltage_path or {})
        _eps_max = dict(port_eps_max or {})

        # Resolve modes-per-port: an int applies to every port; a dict maps
        # port name -> count (ports not listed default to 1).
        def _nmodes_for(port: str) -> int:
            if isinstance(nmodes, dict):
                return int(nmodes.get(port, nmodes.get('default', 1)))
            return int(nmodes)

        fes_full = HCurl(self.mesh, order=self.order, dirichlet=self.bc)

        # Detect geometry for all ports
        for port in ports:
            geometry = self._detect_port_geometry(port)
            self.port_geometries[port] = geometry
            self.port_normals[port] = geometry.normal
            self.port_tangent_frames[port] = (geometry.t1, geometry.t2)
            self.port_orientation_factors[port] = self._compute_orientation_factor(geometry.normal)
            self.port_polarizations[port] = self._compute_polarization_vector(geometry.t1, geometry.t2)

            pr.info(f"  {port}: {geometry.type.value} (fit error: {geometry.fit_error:.4f})")
            if geometry.type == PortGeometryType.RECTANGULAR:
                pr.debug(f"    a={geometry.a:.6f}, b={geometry.b:.6f}")
            elif geometry.type == PortGeometryType.CIRCULAR:
                pr.debug(f"    R={geometry.radius:.6f}")
            elif geometry.type == PortGeometryType.COAXIAL:
                pr.debug(f"    R_outer={geometry.radius:.6f}, R_inner={geometry.inner_radius:.6f}")

        if self.mode_source == 'analytic':
            unsupported = [p for p, g in self.port_geometries.items()
                           if g.type == PortGeometryType.UNKNOWN and p not in _qtem]
            if unsupported:
                raise ValueError(
                    f"Analytic modes requested but ports {unsupported} have "
                    f"unsupported geometry. Use mode_source='numeric' instead."
                )

        # ────────────────────────────────────────────────────────────────
        # Precompute boundary mass matrix ONCE per port
        # ────────────────────────────────────────────────────────────────
        pr.debug("  Precomputing boundary mass matrices (once per port)...")
        u_full, v_full = fes_full.TnT()
        self.port_mass_matrices.clear()
        self.port_mass_forms.clear()

        for port in ports:
            m_form = BilinearForm(InnerProduct(u_full.Trace(), v_full.Trace()) * ds(self._region(port)))
            with TaskManager():
                m_form.Assemble()
            M_bnd = sp.csr_matrix(m_form.mat.CSR())
            self.port_mass_matrices[port] = M_bnd
            self.port_mass_forms[port] = m_form   # ← keeps C++ matrix alive

        pr.debug(f"    Done for {len(ports)} port(s)")

        # Solve for each port
        pr.debug(f"  Port order: {ports}")
        # Determine which ports are internal (interface) vs external
        _internal_ports = set(internal_ports) if internal_ports else set()

        for port in ports:
            geometry = self.port_geometries[port]
            is_internal = port in _internal_ports

            nm_port = _nmodes_for(port)
            # A composite (multi-face) port is only supported as a quasi-TEM port;
            # the analytic/numeric TE/TM/TEM solvers below take the port name as a
            # single mesh region and would silently see an empty region otherwise.
            if port not in _qtem and self._region(port) != port:
                raise NotImplementedError(
                    f"Port '{port}' groups multiple mesh faces "
                    f"('{self._region(port)}') but is not a quasi-TEM port. "
                    f"Multi-face composite ports are currently supported only as "
                    f"quasi-TEM ports — add '{port}' to qtem_ports, or give the "
                    f"port a single boundary face.")
            if port in _qtem:
                pr.debug(f"  Quasi-TEM calculation for port {port} ({nm_port} mode(s))")
                self._solve_port_qtem(
                    port, nm_port, fes_full,
                    eps_r_bnd=_eps_bnd.get(port),
                    cond_bbnd=_cond_bbnd.get(port),
                    k0_ref=k0_ref,
                    voltage_path=_vpath.get(port),
                    eps_max=_eps_max.get(port),
                )
            elif self.mode_source == 'analytic':
                if is_internal and self.mode_source_internal == 'numeric':
                    pr.debug(f"  Numeric calculation for internal port {port} ({nm_port} mode(s))")
                    self._solve_port_numeric(port, nm_port, fes_full)
                else:
                    pr.debug(f"  Analytic calculation for port {port} ({nm_port} mode(s))")
                    self._solve_port_analytic(port, geometry, nm_port, fes_full)
            else:
                self._solve_port_numeric(port, nm_port, fes_full)

        pr.done(f"Port eigenmodes complete: {sum(len(m) for m in self.port_modes.values())} total modes")

        return self.port_modes, self.port_basis

    def _solve_port_analytic(self, port: str, geometry: PortGeometry, nmodes: int, fes_full: HCurl) -> None:
        fes_port = HCurl(
            self.mesh, order=self.order,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )

        port_region = self.mesh.Boundaries(port)
        sigma = self.port_orientation_factors[port]
        t1, t2 = geometry.t1, geometry.t2

        if geometry.type == PortGeometryType.RECTANGULAR:
            analytic_modes = self._generate_rectangular_modes(geometry, nmodes)
        elif geometry.type == PortGeometryType.COAXIAL:
            analytic_modes = self._generate_coaxial_modes(geometry, nmodes)
        else:
            analytic_modes = self._generate_circular_modes(geometry, nmodes)

        self.port_modes[port] = {}
        self.port_cutoff_kc[port] = {}
        self.port_cutoff_frequencies[port] = {}
        self.port_basis[port] = {}
        self.port_mode_types[port] = {}
        self.port_mode_indices[port] = {}
        self.port_mode_degeneracies[port] = {}
        self.port_phase_signs[port] = {}
        self.port_mode_polarizations[port] = {}

        mode_idx = 0
        for amode in analytic_modes:
            if mode_idx >= nmodes:
                break

            if geometry.type == PortGeometryType.RECTANGULAR:
                mode_cf = self._create_rectangular_mode_cf(amode, geometry)
            elif geometry.type == PortGeometryType.COAXIAL:
                mode_cf = self._create_coaxial_mode_cf(amode, geometry)
            else:
                mode_cf = self._create_circular_mode_cf(amode, geometry)

            mode_gf = GridFunction(fes_port)
            mode_gf.Set(mode_cf, definedon=port_region)

            norm_sq = float(np.real(Integrate(
                InnerProduct(mode_gf, mode_gf),
                self.mesh, BND, definedon=port_region
            )))

            if norm_sq < 1e-15:
                pr.warning(
                    f"Port {port} mode {mode_idx} ({amode.type}_{amode.indices[0]}{amode.indices[1]}) "
                    f"has near-zero norm ({norm_sq:.2e}). "
                    f"The mode field may be misaligned with the port boundary. "
                    f"Check port geometry detection and coordinate system."
                )
                continue

            mode_gf.vec.data /= np.sqrt(norm_sq)

            basis = self._create_basis_vector(mode_gf, port, fes_full)

            self.port_modes[port][mode_idx] = mode_gf
            self.port_cutoff_kc[port][mode_idx] = amode.kc
            self.port_cutoff_frequencies[port][mode_idx] = c0 * amode.kc / (2 * np.pi)
            self.port_basis[port][mode_idx] = basis
            self.port_mode_types[port][mode_idx] = amode.type
            self.port_mode_indices[port][mode_idx] = amode.indices
            self.port_mode_degeneracies[port][mode_idx] = amode.degeneracy
            self.port_phase_signs[port][mode_idx] = 1.0

            if amode.degeneracy > 1:
                if amode.polarization == 'cos':
                    pol_angle = self.polarization_angle
                else:
                    pol_angle = self.polarization_angle + np.pi / 2
            else:
                pol_angle = self.polarization_angle
            self.port_mode_polarizations[port][mode_idx] = pol_angle

            m, n = amode.indices
            mode_name = f"{amode.type}_{m}{n}"
            pol_str = f" ({amode.polarization})" if amode.polarization else ""
            print(f"\t{port} mode {mode_idx}: {mode_name}{pol_str}, "
                  f"kc={amode.kc:.4f}, sigma={sigma:+.0f}")

            mode_idx += 1

    def _solve_port_numeric(self, port: str, nmodes: int, fes_full: HCurl) -> None:
        fes_port = HCurl(
            self.mesh, order=self.order,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )

        geometry = self.port_geometries[port]
        t1, t2 = geometry.t1, geometry.t2
        normal = geometry.normal
        sigma = self.port_orientation_factors[port]

        # Now returns mode types directly
        raw_modes, raw_cutoffs, raw_mode_types = self._solve_eigenvalue_problem(
            fes_port, port, nmodes * 2 + 4)
        
        # Group degenerate modes (pass types through)
        mode_groups = self._group_degenerate_modes(raw_modes, raw_cutoffs, raw_mode_types)

        self.port_modes[port] = {}
        self.port_cutoff_kc[port] = {}
        self.port_cutoff_frequencies[port] = {}
        self.port_basis[port] = {}
        self.port_mode_types[port] = {}
        self.port_phase_signs[port] = {}
        self.port_mode_polarizations[port] = {}
        self.port_mode_degeneracies[port] = {}

        mode_idx = 0
        for kc, group_modes, group_type in mode_groups:
            if mode_idx >= nmodes:
                break

            polarized = self._select_polarized_modes(group_modes, port, t1, t2, self.polarization_angle)

            for aligned_mode, alignment, pol_angle, degeneracy in polarized:
                if mode_idx >= nmodes:
                    break

                phase_sign = self._normalize_mode_phase(aligned_mode, port, t1)
                basis = self._create_basis_vector(aligned_mode, port, fes_full)

                self.port_modes[port][mode_idx] = aligned_mode
                self.port_cutoff_kc[port][mode_idx] = kc
                self.port_cutoff_frequencies[port][mode_idx] = c0 * kc / (2 * np.pi)
                self.port_basis[port][mode_idx] = basis
                self.port_mode_types[port][mode_idx] = group_type
                self.port_phase_signs[port][mode_idx] = phase_sign
                self.port_mode_polarizations[port][mode_idx] = pol_angle
                self.port_mode_degeneracies[port][mode_idx] = degeneracy

                pol_deg = np.degrees(pol_angle) % 360
                degen_str = f", degen={degeneracy}" if degeneracy > 1 else ""
                pol_str = f", pol={pol_deg:.0f}°" if degeneracy > 1 else ""
                
                print(f"\t  {port} mode {mode_idx}: kc={kc:.4f}, "
                    f"type={group_type}, fc={c0 * kc / (2 * np.pi) / 1e9:.4f} GHz, "
                    f"sigma={sigma:+.0f}, phase={'+' if phase_sign > 0 else '-'}"
                    f"{pol_str}{degen_str}")

                mode_idx += 1
        print()

    # =========================================================================
    # Quasi-TEM (inhomogeneous / microstrip) port modes
    # =========================================================================

    def _solve_port_qtem(self, port: str, nmodes: int, fes_full: HCurl,
                         eps_r_bnd: Optional[CoefficientFunction] = None,
                         cond_bbnd: Optional[str] = None,
                         k0_ref: Optional[float] = None,
                         voltage_path: Optional[Tuple] = None,
                         eps_max: Optional[float] = None) -> None:
        """Solve quasi-TEM modes on an inhomogeneous port cross-section.

        Uses the classic mixed HCurl(Et) x H1(Ez) vector-FE formulation, solved
        at a reference wavenumber ``k0_ref`` with the (spatially varying) port
        permittivity ``eps_r_bnd``.  The propagation constant beta is the
        eigenvalue's square root; modes are ordered to match CST — physical
        propagating modes sorted by **descending real(beta)** (the fundamental
        quasi-TEM mode first).  ``nmodes`` of them are kept.

        The transverse field Et is stored as the port mode (feeds the FOM port
        basis B); the power-voltage characteristic impedance Z_PV is stored as
        the S-parameter renormalisation reference (see get_port_wave_impedance).
        """
        region = self._region(port)
        port_region = self.mesh.Boundaries(region)

        if eps_r_bnd is None:
            eps_r_bnd = CoefficientFunction(1.0)
        if k0_ref is None or k0_ref <= 0:
            raise ValueError(
                f"qTEM port '{port}' requires a positive reference wavenumber "
                f"k0_ref (set from the solve frequency range).")

        # Mixed space on the port trace, PEC (Dirichlet) on conductor edges.
        fesEt = HCurl(self.mesh, order=self.order,
                      definedon=port_region, complex=True,
                      dirichlet_bbnd=(cond_bbnd or ''))
        GEt, fesEz = fesEt.CreateGradient()
        fes = fesEt * fesEz
        (Et, p), (Ft, q) = fes.TnT()

        k0 = float(k0_ref)
        a = BilinearForm(fes)
        a += (curl(Et).Trace() * curl(Ft).Trace()
              - k0**2 * eps_r_bnd * Et.Trace() * Ft.Trace()) * ds(region)
        a += -grad(p).Trace() * Ft.Trace() * ds(region)
        a += (grad(p).Trace() * grad(q).Trace()
              - k0**2 * eps_r_bnd * p.Trace() * q.Trace()) * ds(region)

        m = BilinearForm(fes)
        m += -Et.Trace() * Ft.Trace() * ds(region)
        m += Et.Trace() * grad(q).Trace() * ds(region)

        n_eig = max(30, nmodes * 8)
        with TaskManager():
            a.Assemble()
            m.Assemble()
            evecs = GridFunction(fes, multidim=n_eig, name='qtem_modes')
            lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(),
                                list(evecs.vecs), shift=1.0)

        lam = np.array([complex(l) for l in lam])
        beta = np.sqrt(lam)
        eps_eff = lam / k0**2
        if eps_max is None or eps_max <= 0:
            eps_max = self._eps_bnd_max(eps_r_bnd)

        # CST-matching ordering: keep physical propagating modes (real beta
        # dominant, effective permittivity between vacuum and the max material),
        # then sort by descending real(beta) — fundamental quasi-TEM first.
        cand = []
        for i in range(len(lam)):
            b, ee = beta[i], eps_eff[i]
            if (b.real > 1e-3 and abs(b.imag) < 0.3 * abs(b.real)
                    and 1.0 <= ee.real <= eps_max * 1.15):
                cand.append((i, b, ee))
        cand.sort(key=lambda z: -z[1].real)

        # Reset per-port storage
        for d in (self.port_modes, self.port_cutoff_kc, self.port_cutoff_frequencies,
                  self.port_basis, self.port_mode_types, self.port_phase_signs,
                  self.port_mode_polarizations, self.port_mode_degeneracies,
                  self.port_mode_indices, self.port_beta, self.port_eps_eff,
                  self.port_line_impedance):
            d[port] = {}

        fes_port = HCurl(self.mesh, order=self.order, complex=False,
                         definedon=port_region)
        omega = k0 * c0

        kept = 0
        for (idx, b, ee) in cand:
            if kept >= nmodes:
                break
            gf = GridFunction(fes)
            gf.vec.data = evecs.vecs[idx]
            Et_c = gf.components[0]   # complex HCurl transverse field

            # De-phase to a (near-)real field so the FOM port basis stays real.
            Et_real = self._dephase_to_real(Et_c, fes_port, port_region)

            norm_sq = float(np.real(Integrate(
                InnerProduct(Et_real, Et_real), self.mesh, BND,
                definedon=port_region)))
            if norm_sq < 1e-20:
                continue
            Et_real.vec.data /= np.sqrt(norm_sq)

            # Power-voltage characteristic (line) impedance for S renormalisation
            zpv = self._compute_qtem_zpv(Et_c, b.real, omega, port_region,
                                         voltage_path)

            basis = self._create_basis_vector(Et_real, port, fes_full)

            self.port_modes[port][kept] = Et_real
            self.port_basis[port][kept] = basis
            self.port_cutoff_kc[port][kept] = 0.0
            self.port_cutoff_frequencies[port][kept] = 0.0
            self.port_mode_types[port][kept] = 'qTEM'
            self.port_phase_signs[port][kept] = 1.0
            self.port_mode_polarizations[port][kept] = self.polarization_angle
            self.port_mode_degeneracies[port][kept] = 1
            self.port_mode_indices[port][kept] = (0, 0)
            self.port_beta[port][kept] = complex(b)
            self.port_eps_eff[port][kept] = complex(ee)
            self.port_line_impedance[port][kept] = complex(zpv)

            print(f"\t{port} mode {kept}: qTEM, eps_eff={ee.real:.4f}, "
                  f"beta={b.real:.3f} rad/m, Z_PV={zpv.real:.2f} ohm")
            kept += 1

        if kept == 0:
            raise RuntimeError(
                f"qTEM port '{port}': no physical propagating mode found. "
                f"Check the conductor edge groups (dirichlet_bbnd) and the "
                f"port permittivity map.")

    @staticmethod
    def _eps_bnd_max(eps_r_bnd: CoefficientFunction) -> float:
        """Upper bound on the port permittivity for the physical-mode filter.

        The caller (the frequency-domain solver) passes the exact per-port
        maximum; this generous fallback is only used when :meth:`_solve_port_qtem`
        is driven directly without one.
        """
        return 100.0

    def _dephase_to_real(self, Et_complex, fes_port_real: HCurl, port_region) -> GridFunction:
        """Rotate a complex mode by a global phase and take its real part.

        A quasi-TEM cross-section mode is real up to a single global phase e^{i*phi};
        the FOM port basis must be real, so we align the field to the real axis.

        The phase must come from the **non-conjugated** self-product
        integral(Et . Et) = e^{2i*phi} integral|Er|^2 (angle = 2*phi).  NGSolve's
        ``InnerProduct`` is Hermitian (conjugated) and would give a real result
        (angle 0, i.e. no rotation) — which silently drops the mode when the raw
        eigenvector happens to carry a ~90 deg global phase.
        """
        dim = Et_complex.dim
        bilinear = sum(Et_complex[k] * Et_complex[k] for k in range(dim))
        ip = complex(Integrate(bilinear, self.mesh, BND, definedon=port_region))
        phase = 0.5 * np.angle(ip) if abs(ip) > 1e-30 else 0.0
        rot = CoefficientFunction(np.exp(-1j * phase)) * Et_complex
        gf = GridFunction(fes_port_real)
        gf.Set(rot.real, definedon=port_region)
        return gf

    def _compute_qtem_zpv(self, Et_complex, beta_real: float, omega: float,
                          port_region, voltage_path: Optional[Tuple]) -> complex:
        """Power-voltage characteristic impedance Z_PV = |V|^2 / (2 P).

        Quasi-TEM transverse magnetic field Ht = (beta/omega/mu0) (z x Et), so
        the time-average power flow is P = 0.5 (beta/omega/mu0) integral |Et|^2.
        V is the line integral of Et along ``voltage_path`` (ground -> strip).
        Z_PV is invariant to the mode's global scale/phase.
        """
        P = 0.5 * (beta_real / (omega * mu0)) * float(np.real(Integrate(
            InnerProduct(Et_complex, Et_complex), self.mesh, BND,
            definedon=port_region)))
        if P <= 0 or voltage_path is None:
            return complex('nan')

        p0 = np.asarray(voltage_path[0], dtype=float)
        p1 = np.asarray(voltage_path[1], dtype=float)
        seg = p1 - p0
        n_samp = 200
        ts = np.linspace(0.0, 1.0, n_samp)
        V = 0j
        Evals = np.zeros(n_samp, dtype=complex)
        for j, tt in enumerate(ts):
            pt = p0 + tt * seg
            try:
                val = Et_complex(self.mesh(pt[0], pt[1], pt[2], BND))
                Evals[j] = complex(np.dot([complex(v) for v in val], seg)
                                   / (np.linalg.norm(seg) + 1e-30))
            except Exception:
                Evals[j] = 0j
        dl = np.linalg.norm(seg)
        V = np.trapz(Evals, ts) * dl
        # Degenerate path (all point evaluations failed / near-zero voltage):
        # return NaN so the impedance lookup falls back cleanly rather than
        # renormalising S to ~0 ohm.
        if not np.isfinite(V) or abs(V) < 1e-12:
            return complex('nan')
        return complex(abs(V) ** 2 / (2.0 * P))

    # =========================================================================
    # Optimized basis vector creation
    # =========================================================================

    def _create_basis_vector(self, mode_gf: GridFunction, port: str, fes_full: HCurl) -> np.ndarray:
        """Create mass-weighted basis vector using precomputed matrix."""
        port_region = self.mesh.Boundaries(self._region(port))

        # Use precomputed mass matrix
        M_bnd = self.port_mass_matrices[port]

        full_gf = GridFunction(fes_full)
        full_gf.Set(mode_gf, definedon=port_region)

        coeffs = full_gf.vec.FV().NumPy().copy()
        return M_bnd @ coeffs

    # =========================================================================
    # All remaining methods
    # =========================================================================

    # Add to PortEigenmodeSolver class

    def to_save_dict(self) -> Dict:
        """
        Extract all data for serialization (no NGSolve objects).

        Returns
        -------
        dict
            Dictionary containing all port mode data that can be pickled.
        """
        save_data = {
            # Solver configuration
            'order': self.order,
            'bc': self.bc,
            'mode_source': self.mode_source,
            'mode_source_internal': self.mode_source_internal,
            'geometry_tolerance': self.geometry_tolerance,
            'polarization_angle': self.polarization_angle,
            'global_up': self.global_up.tolist(),
            'propagation_axis': self.propagation_axis.tolist(),
            'ensure_inward_power': self.ensure_inward_power,

            # Scalar/simple data (directly picklable)
            'port_cutoff_kc': {p: dict(m) for p, m in self.port_cutoff_kc.items()},
            'port_cutoff_frequencies': {p: dict(m) for p, m in self.port_cutoff_frequencies.items()},
            'port_mode_types': {p: dict(m) for p, m in self.port_mode_types.items()},
            'port_orientation_factors': dict(self.port_orientation_factors),
            'port_phase_signs': {p: dict(m) for p, m in self.port_phase_signs.items()},
            'port_mode_polarizations': {p: dict(m) for p, m in self.port_mode_polarizations.items()},
            'port_mode_degeneracies': {p: dict(m) for p, m in self.port_mode_degeneracies.items()},
            'port_mode_indices': {p: dict(m) for p, m in self.port_mode_indices.items()},

            # Composite / quasi-TEM port data
            'port_face_region': dict(self.port_face_region),
            'port_beta': {p: {m: complex(v) for m, v in d.items()}
                          for p, d in self.port_beta.items()},
            'port_eps_eff': {p: {m: complex(v) for m, v in d.items()}
                             for p, d in self.port_eps_eff.items()},
            'port_line_impedance': {p: {m: complex(v) for m, v in d.items()}
                                    for p, d in self.port_line_impedance.items()},

            # Numpy arrays
            'port_normals': {p: n.tolist() for p, n in self.port_normals.items()},
            'port_polarizations': {p: n.tolist() for p, n in self.port_polarizations.items()},
            'port_tangent_frames': {
                p: (t1.tolist(), t2.tolist())
                for p, (t1, t2) in self.port_tangent_frames.items()
            },

            # Port geometries (dataclass -> dict)
            'port_geometries': {
                p: {
                    'type': g.type.value,
                    'center': g.center.tolist(),
                    'normal': g.normal.tolist(),
                    't1': g.t1.tolist(),
                    't2': g.t2.tolist(),
                    'area': g.area,
                    'a': g.a,
                    'b': g.b,
                    'radius': g.radius,
                    'fit_error': g.fit_error,
                }
                for p, g in self.port_geometries.items()
            },

            # Port basis vectors (already numpy arrays)
            'port_basis': {
                p: {m: b.tolist() for m, b in modes.items()}
                for p, modes in self.port_basis.items()
            },

            # GridFunction vector data (extract numpy arrays)
            'port_modes_vectors': {
                p: {
                    m: gf.vec.FV().NumPy().copy().tolist()
                    for m, gf in modes.items()
                }
                for p, modes in self.port_modes.items()
            },

            # Store which ports exist (for reconstruction order)
            'ports': list(self.port_modes.keys()),
        }

        return save_data

    @classmethod
    def from_save_dict(
            cls,
            data: Dict,
            mesh,
            fes_full: Optional[HCurl] = None
    ) -> 'PortEigenmodeSolver':
        """
        Reconstruct PortEigenmodeSolver from saved data.

        Parameters
        ----------
        data : dict
            Dictionary from to_save_dict()
        mesh : ngsolve.Mesh
            The mesh (must be loaded first)
        fes_full : HCurl, optional
            Full HCurl space for basis vector reconstruction.
            If None, will be created from mesh.

        Returns
        -------
        PortEigenmodeSolver
            Reconstructed solver with all modes restored.
        """
        # Create solver instance with saved configuration
        solver = cls(
            mesh=mesh,
            order=data['order'],
            bc=data['bc'],
            mode_source=data['mode_source'],
            mode_source_internal=data.get('mode_source_internal', data['mode_source']),
            geometry_tolerance=data['geometry_tolerance'],
            polarization_angle=data['polarization_angle'],
            global_up=tuple(data['global_up']),
            propagation_axis=tuple(data['propagation_axis']),
            ensure_inward_power=data['ensure_inward_power'],
        )

        # Restore simple data
        solver.port_cutoff_kc = {p: dict(m) for p, m in data['port_cutoff_kc'].items()}
        solver.port_cutoff_frequencies = {p: dict(m) for p, m in data['port_cutoff_frequencies'].items()}
        solver.port_mode_types = {p: dict(m) for p, m in data['port_mode_types'].items()}
        solver.port_orientation_factors = dict(data['port_orientation_factors'])
        solver.port_phase_signs = {p: dict(m) for p, m in data['port_phase_signs'].items()}
        solver.port_mode_polarizations = {p: dict(m) for p, m in data['port_mode_polarizations'].items()}
        solver.port_mode_degeneracies = {p: dict(m) for p, m in data['port_mode_degeneracies'].items()}
        solver.port_mode_indices = {
            p: {int(m): tuple(idx) for m, idx in modes.items()}
            for p, modes in data['port_mode_indices'].items()
        }

        # Restore composite / quasi-TEM port data
        solver.port_face_region = dict(data.get('port_face_region', {}))
        solver.port_beta = {p: {int(m): complex(v) for m, v in d.items()}
                            for p, d in data.get('port_beta', {}).items()}
        solver.port_eps_eff = {p: {int(m): complex(v) for m, v in d.items()}
                               for p, d in data.get('port_eps_eff', {}).items()}
        solver.port_line_impedance = {p: {int(m): complex(v) for m, v in d.items()}
                                      for p, d in data.get('port_line_impedance', {}).items()}

        # Restore numpy arrays
        solver.port_normals = {p: np.array(n) for p, n in data['port_normals'].items()}
        solver.port_polarizations = {p: np.array(n) for p, n in data['port_polarizations'].items()}
        solver.port_tangent_frames = {
            p: (np.array(t1), np.array(t2))
            for p, (t1, t2) in data['port_tangent_frames'].items()
        }

        # Restore port geometries
        for p, gdata in data['port_geometries'].items():
            solver.port_geometries[p] = PortGeometry(
                type=PortGeometryType(gdata['type']),
                center=np.array(gdata['center']),
                normal=np.array(gdata['normal']),
                t1=np.array(gdata['t1']),
                t2=np.array(gdata['t2']),
                area=gdata['area'],
                a=gdata['a'],
                b=gdata['b'],
                radius=gdata['radius'],
                fit_error=gdata['fit_error'],
            )

        # Create full FES if not provided
        if fes_full is None:
            fes_full = HCurl(mesh, order=data['order'], dirichlet=data['bc'])

        # Precompute mass matrices for basis vector creation
        ports = data['ports']
        u_full, v_full = fes_full.TnT()

        for port in ports:
            m_form = BilinearForm(InnerProduct(u_full.Trace(), v_full.Trace()) * ds(solver._region(port)))
            with TaskManager():
                m_form.Assemble()
            M_bnd = sp.csr_matrix(m_form.mat.CSR())
            solver.port_mass_matrices[port] = M_bnd
            solver.port_mass_forms[port] = m_form

        # Restore GridFunctions and basis vectors
        for port in ports:
            # Create port-specific FES
            fes_port = HCurl(
                mesh, order=data['order'],
                dirichlet=data['bc'],
                definedon=mesh.Boundaries(solver._region(port))
            )

            solver.port_modes[port] = {}
            solver.port_basis[port] = {}

            port_modes_data = data['port_modes_vectors'].get(port, {})
            port_basis_data = data['port_basis'].get(port, {})

            for mode_str, vec_data in port_modes_data.items():
                mode = int(mode_str)

                # Reconstruct GridFunction
                mode_gf = GridFunction(fes_port)
                vec_array = np.array(vec_data)

                # Check size compatibility
                if len(vec_array) == mode_gf.vec.size:
                    mode_gf.vec.FV().NumPy()[:] = vec_array
                else:
                    print(f"Warning: Vector size mismatch for {port} mode {mode}. "
                          f"Expected {mode_gf.vec.size}, got {len(vec_array)}. "
                          f"Mode will need to be recomputed.")
                    # Set to zero - mode needs recomputation
                    mode_gf.vec[:] = 0

                solver.port_modes[port][mode] = mode_gf

                # Reconstruct basis vector using mass matrix
                # (more reliable than loading saved basis)
                solver.port_basis[port][mode] = solver._create_basis_vector(
                    mode_gf, port, fes_full
                )

        print(f"Restored PortEigenmodeSolver with {len(ports)} ports, "
              f"{sum(len(m) for m in solver.port_modes.values())} total modes")

        return solver

    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save port mode data to a pickle file."""
        import pickle
        filepath = Path(filepath)
        data = self.to_save_dict()
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved port modes to {filepath}")

    @classmethod
    def load_from_file(
            cls,
            filepath: Union[str, Path],
            mesh,
            fes_full: Optional[HCurl] = None
    ) -> 'PortEigenmodeSolver':
        """Load port mode data from a pickle file."""
        import pickle
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls.from_save_dict(data, mesh, fes_full)

    def _solve_eigenvalue_problem(self, fes_port, port, nmodes):
        """
        Solve for both TE and TM modes on a port surface.

        TE modes: HCurl curl-curl formulation with gradient projection
        TM modes: H1 scalar formulation for Ez, then Et = -grad(Ez)

        Returns modes sorted by increasing cutoff frequency.
        """
        port_region = self.mesh.Boundaries(port)

        # ========== TEM Mode ==========
        tem_mode = self._solve_tem_mode(port)

        # ========== TE Modes ==========
        te_modes, te_cutoffs = self._solve_te_modes(port, nmodes + 5)

        # ========== TM Modes ==========
        tm_modes, tm_cutoffs = self._solve_tm_modes(port, nmodes + 5)

        # Combine all modes
        all_modes = []
        if tem_mode is not None:
            all_modes.append((0.0, tem_mode, 'TEM'))
        for mode, kc in zip(te_modes, te_cutoffs):
            all_modes.append((kc, mode, 'TE'))
        for mode, kc in zip(tm_modes, tm_cutoffs):
            all_modes.append((kc, mode, 'TM'))

        # Sort by cutoff frequency (ascending)
        all_modes.sort(key=lambda x: x[0])

        # Unpack into separate lists
        modes = [m for _, m, _ in all_modes]
        cutoffs = [k for k, _, _ in all_modes]
        mode_types = [t for _, _, t in all_modes]

        return modes, cutoffs, mode_types

    def _solve_te_modes(self, port: str, nmodes: int):
        """
        Solve for TE modes using HCurl curl-curl formulation.

        TE modes have Hz ≠ 0, Ez = 0.
        On the port surface, we solve:
            curl curl Et = kc² Et
        with tangential E boundary condition (Dirichlet in HCurl).

        Requires gradient projection to remove curl-free null space.

        Returns
        -------
        modes : List[GridFunction]
            TE mode E-field patterns (HCurl GridFunctions), sorted by kc
        cutoffs : List[float]
            Cutoff wavenumbers kc, sorted ascending
        """

        port_region = self.mesh.Boundaries(port)

        # HCurl space on port surface with Dirichlet BC on waveguide walls
        fes_te = HCurl(
            self.mesh, order=self.order,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )

        u, v = fes_te.TnT()

        # Bilinear forms for curl-curl eigenvalue problem
        a = BilinearForm(curl(u.Trace()) * curl(v.Trace()) * ds(port))
        m = BilinearForm(u.Trace() * v.Trace() * ds(port))
        apre = BilinearForm(
            (curl(u).Trace() * curl(v).Trace() + u.Trace() * v.Trace()) * ds(port)
        )
        pre = Preconditioner(apre, type="direct", inverse=_DIRECT_SOLVER)

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            # Gradient projection to remove null-space (curl-free fields)
            # These are gradients of H1 functions that satisfy the Dirichlet BC
            G, fes_h1 = fes_te.CreateGradient()
            GT = G.CreateTranspose()
            math1 = GT @ m.mat @ G
            invh1 = math1.Inverse(inverse=_DIRECT_SOLVER, freedofs=fes_h1.FreeDofs())
            proj = IdentityMatrix(fes_te.ndof) - G @ invh1 @ GT @ m.mat
            projpre = proj @ pre.mat

            evals, evecs = solvers.PINVIT(
                a.mat, m.mat, pre=projpre,
                num=nmodes, maxit=50, printrates=False
            )

        # Collect valid modes with their cutoffs
        mode_data = []  # List of (kc, GridFunction)

        for i, ev in enumerate(evals):
            if ev > 1e-6:  # Skip null-space modes
                kc = np.sqrt(ev)

                mode = GridFunction(fes_te)
                mode.vec.data = evecs[i]

                # Normalize
                norm_sq = float(np.real(Integrate(
                    InnerProduct(mode, mode), self.mesh, BND, definedon=port_region
                )))

                if norm_sq > 1e-15:
                    mode.vec.data /= np.sqrt(norm_sq)
                    mode_data.append((kc, mode))

        # Sort by cutoff frequency (ascending)
        mode_data.sort(key=lambda x: x[0])

        # Unpack
        modes = [m for _, m in mode_data]
        cutoffs = [k for k, _ in mode_data]

        return modes, cutoffs

    def _solve_tm_modes(self, port: str, nmodes: int):
        """
        Solve for TM modes using scalar H1 formulation for Ez.

        TM modes have Ez ≠ 0, Hz = 0.
        On the port surface, we solve:
            -∇²Ez = kc² Ez
        with Dirichlet BC (Ez = 0 on waveguide walls).

        Then the transverse E field is: Et = -∇Ez

        No gradient projection needed - Dirichlet BC removes the constant mode.

        Returns
        -------
        modes : List[GridFunction]
            TM mode E-field patterns (HCurl GridFunctions), sorted by kc
        cutoffs : List[float]
            Cutoff wavenumbers kc, sorted ascending
        """
        from ngsolve import H1, grad

        port_region = self.mesh.Boundaries(port)
        geometry = self.port_geometries[port]

        # H1 space on port surface with Dirichlet BC on waveguide walls
        fes_h1 = H1(
            self.mesh, order=self.order + 1,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )

        # Check if we have any free DOFs (if all are constrained, no TM modes)
        n_free = sum(1 for i in range(fes_h1.ndof) if fes_h1.FreeDofs()[i])
        if n_free < 2:
            return [], []

        u, v = fes_h1.TnT()

        # Bilinear forms for scalar Helmholtz eigenvalue problem
        # Weak form: ∫∇u·∇v = kc² ∫u·v
        a = BilinearForm(InnerProduct(grad(u).Trace(), grad(v).Trace()) * ds(port))
        m = BilinearForm(u.Trace() * v.Trace() * ds(port))
        apre = BilinearForm(
            (InnerProduct(grad(u).Trace(), grad(v).Trace()) + u.Trace() * v.Trace()) * ds(port)
        )
        pre = Preconditioner(apre, type="direct", inverse=_DIRECT_SOLVER)

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            # No gradient projection needed - Dirichlet BC handles constant mode
            evals, evecs = solvers.PINVIT(
                a.mat, m.mat, pre=pre.mat,
                num=min(nmodes, n_free - 1),
                maxit=50, printrates=False
            )

        # HCurl space for storing the transverse E field
        fes_hcurl = HCurl(
            self.mesh, order=self.order,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )

        # Collect valid modes with their cutoffs
        mode_data = []  # List of (kc, GridFunction)

        for i, ev in enumerate(evals):
            if ev > 1e-6:  # Skip near-zero eigenvalues
                kc = np.sqrt(ev)

                # Create Ez GridFunction
                Ez = GridFunction(fes_h1)
                Ez.vec.data = evecs[i]

                # Compute Et = -∇Ez (transverse E field for TM mode)
                Et_cf = -grad(Ez)

                # Project to HCurl space
                Et = GridFunction(fes_hcurl)
                Et.Set(Et_cf, definedon=port_region)

                # Normalize
                norm_sq = float(np.real(Integrate(
                    InnerProduct(Et, Et), self.mesh, BND, definedon=port_region
                )))

                if norm_sq > 1e-15:
                    Et.vec.data /= np.sqrt(norm_sq)
                    mode_data.append((kc, Et))

        # Sort by cutoff frequency (ascending)
        mode_data.sort(key=lambda x: x[0])

        # Unpack
        modes = [m for _, m in mode_data]
        cutoffs = [k for k, _ in mode_data]

        return modes, cutoffs

    def _solve_tem_mode(self, port: str):
        """
        Solve for TEM mode on a port cross-section.

        TEM modes exist only on multi-conductor ports (e.g. coaxial).
        The transverse E field is Et = -∇φ where φ satisfies Laplace's
        equation with distinct potentials on inner and outer conductors.

        We detect a TEM mode by solving the Laplace eigenvalue problem
        (same as TM but looking for a near-zero eigenvalue that is NOT
        the trivial constant mode removed by Dirichlet BC).  If the port
        has only a single conductor boundary the Dirichlet BC removes all
        constant modes and no TEM mode exists.

        For a coaxial port the inner conductor has Dirichlet BC (φ=0)
        while the outer conductor also has Dirichlet BC.  The TEM potential
        is the harmonic function that is 1 on the inner conductor and 0 on
        the outer (or vice-versa).  We approximate this by solving the
        Laplace problem with an inhomogeneous Dirichlet lift.

        Returns
        -------
        mode : GridFunction or None
            TEM mode E-field pattern (HCurl), or None if no TEM mode exists.
        """

        port_region = self.mesh.Boundaries(port)

        # H1 space on port surface with Dirichlet BC on waveguide walls
        fes_h1 = H1(
            self.mesh, order=self.order + 1,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )

        n_free = sum(1 for i in range(fes_h1.ndof) if fes_h1.FreeDofs()[i])
        if n_free < 2:
            return None

        u_h1, v_h1 = fes_h1.TnT()

        # Solve the same Laplace eigenvalue problem as TM modes
        # but look for eigenvalues very close to zero.
        # A near-zero eigenvalue (but non-trivial) indicates a TEM mode.
        a = BilinearForm(InnerProduct(grad(u_h1).Trace(), grad(v_h1).Trace()) * ds(port))
        m = BilinearForm(u_h1.Trace() * v_h1.Trace() * ds(port))
        apre = BilinearForm(
            (InnerProduct(grad(u_h1).Trace(), grad(v_h1).Trace()) + u_h1.Trace() * v_h1.Trace()) * ds(port)
        )
        pre = Preconditioner(apre, type="direct", inverse=_DIRECT_SOLVER)

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            evals, evecs = solvers.PINVIT(
                a.mat, m.mat, pre=pre.mat,
                num=min(3, n_free - 1),
                maxit=50, printrates=False
            )

        # Look for a near-zero eigenvalue — this is the TEM mode.
        # The Dirichlet BC removes the trivial constant, so a near-zero
        # eigenvalue means a harmonic function with non-trivial gradient
        # exists (multi-conductor topology).
        tem_threshold = 1e-4  # eigenvalue threshold for "near zero"

        for i, ev in enumerate(evals):
            if ev < tem_threshold and ev >= 0:
                # Found a TEM candidate — compute Et = -∇φ
                phi = GridFunction(fes_h1)
                phi.vec.data = evecs[i]

                Et_cf = -grad(phi)

                # Project to HCurl space
                fes_hcurl = HCurl(
                    self.mesh, order=self.order,
                    dirichlet=self.bc,
                    definedon=self.mesh.Boundaries(port)
                )
                Et = GridFunction(fes_hcurl)
                Et.Set(Et_cf, definedon=port_region)

                # Normalize
                norm_sq = float(np.real(Integrate(
                    InnerProduct(Et, Et), self.mesh, BND, definedon=port_region
                )))

                if norm_sq > 1e-15:
                    Et.vec.data /= np.sqrt(norm_sq)
                    return Et

        return None

    def _classify_mode_type(self, mode, port, normal):
        port_region = self.mesh.Boundaries(port)
        n_cf = CoefficientFunction(tuple(normal))
        E_n_sq = float(np.real(Integrate(InnerProduct(mode, n_cf)**2, self.mesh, BND, definedon=port_region)))
        E_tot_sq = float(np.real(Integrate(InnerProduct(mode, mode), self.mesh, BND, definedon=port_region)))
        return 'TM' if E_tot_sq > 1e-15 and E_n_sq / E_tot_sq >= 0.01 else 'TE'

    def _group_degenerate_modes(self, modes, cutoffs, types, tol=1e-3):
        if not modes:
            return []
        sorted_data = sorted(zip(cutoffs, modes, types), key=lambda x: x[0])
        groups = []
        curr_kc, curr_group, curr_types = sorted_data[0][0], [sorted_data[0][1]], [sorted_data[0][2]]
        for kc, mode, mtype in sorted_data[1:]:
            if curr_kc > 1e-10 and abs(kc - curr_kc) / curr_kc < tol:
                curr_group.append(mode)
                curr_types.append(mtype)
            else:
                groups.append((curr_kc, curr_group, max(set(curr_types), key=curr_types.count)))
                curr_kc, curr_group, curr_types = kc, [mode], [mtype]
        groups.append((curr_kc, curr_group, max(set(curr_types), key=curr_types.count)))
        return groups

    def _select_polarized_modes(self, modes, port, t1, t2, angle):
        port_region = self.mesh.Boundaries(port)
        if len(modes) == 1:
            return [(modes[0], 1.0, angle, 1)]
        elif len(modes) == 2:
            results = []
            t1_cf = CoefficientFunction(tuple(t1))
            t2_cf = CoefficientFunction(tuple(t2))
            c = [[float(np.real(Integrate(InnerProduct(m, t1_cf), self.mesh, BND, definedon=port_region))),
                  float(np.real(Integrate(InnerProduct(m, t2_cf), self.mesh, BND, definedon=port_region)))]
                 for m in modes]
            A = np.array(c).T
            for offset in [0, np.pi / 2]:
                ang = angle + offset
                target = np.array([np.cos(ang), np.sin(ang)])
                coeffs = np.linalg.lstsq(A, target, rcond=None)[0]
                coeffs /= np.linalg.norm(coeffs) + 1e-12
                aligned = GridFunction(modes[0].space)
                aligned.vec.data = coeffs[0] * modes[0].vec + coeffs[1] * modes[1].vec
                norm_sq = float(np.real(Integrate(InnerProduct(aligned, aligned), self.mesh, BND, definedon=port_region)))
                if norm_sq > 1e-15:
                    aligned.vec.data /= np.sqrt(norm_sq)
                results.append((aligned, 1.0, ang, 2))
            return results
        return [(modes[0], 1.0, angle, 1)]

    def _normalize_mode_phase(self, mode, port, t1):
        port_region = self.mesh.Boundaries(port)
        center, _ = self._compute_port_centroid_and_area(port)
        t1_cf = CoefficientFunction(tuple(t1))
        w = (x - center[0]) * t1[0] + (y - center[1]) * t1[1] + (z - center[2]) * t1[2]
        proj = float(np.real(Integrate(InnerProduct(mode, t1_cf) * w, self.mesh, BND, definedon=port_region)))
        if proj < 0:
            mode.vec.data *= -1
            return -1.0
        return 1.0

    # ────────────────────────────────────────────────────────────────────────
    # Wave Impedance & Utility Methods
    # ────────────────────────────────────────────────────────────────────────

    def _port_media_eps_for(self, port) -> float:
        """Relative permittivity of the medium filling a (possibly composite) port.

        ``port_media_eps`` is keyed by mesh FACE name; a composite qTEM port
        (``port1`` = ``port1_substrate|port1_air``) resolves to the max eps over
        its member faces.  Defaults to vacuum (1.0).
        """
        eps_map = getattr(self, 'port_media_eps', {}) or {}
        if not eps_map:
            return 1.0
        key = str(port)
        if key in eps_map:
            return float(eps_map[key])
        region = self.port_face_region.get(key, key)
        faces = region.split('|')
        vals = [float(eps_map[f]) for f in faces if f in eps_map]
        return max(vals) if vals else 1.0

    def get_port_wave_impedance(self, port: str, mode: int, freq: float) -> complex:
        # Robust lookup: handle cases where port keys might be ints or strings
        try:
            p_key = port
            if p_key not in self.port_cutoff_kc:
                # Try converting to int if it's a digit string like '1', '2'
                if isinstance(port, str) and port.isdigit():
                    p_key = int(port)
                elif isinstance(port, str) and port.lower().startswith('port'):
                    # Try extracting the number: 'port1' -> 1
                    try:
                        p_key = int(port[4:])
                    except ValueError:
                        pass
            
            kc = self.port_cutoff_kc[p_key][mode]
            mode_type = self.port_mode_types[p_key][mode]
        except KeyError:
            # Fallback: if 'port1' is requested but keys are 'port1', 1, etc.
            # and our logic above didn't find it, just raise a more helpful error
            raise KeyError(f"Port '{port}' not found in solver data. Available: {list(self.port_cutoff_kc.keys())}")

        # Quasi-TEM ports renormalise S to their power-voltage line impedance
        # (matches CST's reference impedance), not the analytic wave impedance.
        if mode_type == 'qTEM':
            zpv = (self.port_line_impedance.get(p_key, {}) or {}).get(mode)
            if zpv is not None and np.isfinite(zpv) and complex(zpv).real > 1e-6:
                return complex(zpv)
            # Fallback: medium wave impedance eta = eta0 / sqrt(eps_r).  Resolve
            # eps from the port's member faces (port_media_eps is keyed by FACE
            # name, e.g. 'port1_substrate', not the logical port 'port1').
            return complex(Z0 / np.sqrt(self._port_media_eps_for(p_key)))

        wc = kc * c0
        s = 1j * 2 * np.pi * freq
        # Medium wave impedance eta = eta0 / sqrt(eps_r).  The FOM normalises
        # its port modes to the wave impedance, so the Z->S reference must be
        # the medium wave impedance (eta), NOT the coaxial line impedance --
        # the latter is inconsistent with the FOM's Z normalisation.  eps_r of
        # the medium filling the port is looked up per port (default vacuum),
        # so dielectric-filled couplers are handled correctly.
        eps_r = (getattr(self, 'port_media_eps', {}) or {}).get(port, 1.0)
        eta = Z0 / np.sqrt(eps_r)
        if mode_type == 'TEM':
            return complex(eta)
        sqrt_term = np.sqrt(s**2 + wc**2)
        if mode_type == 'TE':
            return complex(s * eta / sqrt_term)
        else:
            return complex(eta * sqrt_term / s)

    def get_port_wave_impedance_matrix(self, freq: float) -> np.ndarray:
        impedances = []
        for port in sorted(self.port_modes.keys()):
            for mode in sorted(self.port_modes[port].keys()):
                impedances.append(self.get_port_wave_impedance(port, mode, freq))
        return np.diag(impedances)

    def get_propagation_constant(self, port: str, mode: int, freq: float) -> complex:
        # Robust lookup
        try:
            p_key = port
            if p_key not in self.port_cutoff_kc:
                if isinstance(port, str) and port.isdigit():
                    p_key = int(port)
                elif isinstance(port, str) and port.lower().startswith('port'):
                    try:
                        p_key = int(port[4:])
                    except ValueError:
                        pass
            kc = self.port_cutoff_kc[p_key][mode]
        except KeyError:
            raise KeyError(f"Port '{port}' not found in solver data. Available: {list(self.port_cutoff_kc.keys())}")

        wc = kc * c0
        s = 1j * 2 * np.pi * freq
        return complex(np.sqrt(s**2 + wc**2) / c0)

    def get_cutoff_frequency(self, port: str, mode: int = 0) -> float:
        if port not in self.port_cutoff_kc:
            raise KeyError(f"Port {port} not found")
        if mode not in self.port_cutoff_kc[port]:
            raise KeyError(f"Mode {mode} not found for port {port}")
        return c0 * self.port_cutoff_kc[port][mode] / (2 * np.pi)

    def get_cutoff_frequencies_dict(self) -> Dict[str, Dict[int, float]]:
        return {
            port: {mode: c0 * kc / (2 * np.pi) for mode, kc in modes.items()}
            for port, modes in self.port_cutoff_kc.items()
        }

    def get_polarization_info(self) -> Dict[str, Dict]:
        info = {}
        for port in self.port_normals:
            t1, t2 = self.port_tangent_frames[port]
            info[port] = {
                'normal': self.port_normals[port],
                'tangent1': t1,
                'tangent2': t2,
                'polarization': self.port_polarizations.get(port),
                'polarization_angle': self.polarization_angle,
                'orientation_factor': self.port_orientation_factors[port],
                'phase_signs': self.port_phase_signs.get(port, {}),
                'mode_polarizations': self.port_mode_polarizations.get(port, {})
            }
        return info

    def get_mode_info(self) -> Dict[str, Dict[int, Dict]]:
        info = {}
        for port in self.port_modes:
            info[port] = {}
            for mode in self.port_modes[port]:
                kc = self.port_cutoff_kc[port][mode]
                fc = c0 * kc / (2 * np.pi)
                info[port][mode] = {
                    'type': self.port_mode_types[port][mode],
                    'kc': kc,
                    'fc_Hz': fc,
                    'fc_GHz': fc / 1e9,
                    'orientation_factor': self.port_orientation_factors[port],
                    'phase_sign': self.port_phase_signs.get(port, {}).get(mode, 1.0),
                    'degeneracy': self.port_mode_degeneracies.get(port, {}).get(mode, 1),
                    'indices': self.port_mode_indices.get(port, {}).get(mode),
                    'polarization_angle_rad': self.port_mode_polarizations.get(port, {}).get(mode),
                    'polarization_angle_deg': np.degrees(self.port_mode_polarizations.get(port, {}).get(mode, 0)) % 360
                }
        return info

    def get_geometry_info(self) -> Dict[str, Dict]:
        info = {}
        for port, geom in self.port_geometries.items():
            info[port] = {
                'type': geom.type.value,
                'center': geom.center.tolist(),
                'normal': geom.normal.tolist(),
                't1': geom.t1.tolist(),
                't2': geom.t2.tolist(),
                'area': geom.area,
                'fit_error': geom.fit_error
            }
            if geom.type == PortGeometryType.RECTANGULAR:
                info[port]['a'] = geom.a
                info[port]['b'] = geom.b
            elif geom.type == PortGeometryType.CIRCULAR:
                info[port]['radius'] = geom.radius
        return info

    def get_port_wave_impedance_matrix_with_info(self, freq: float) -> Tuple[np.ndarray, List[Dict]]:
        impedances = []
        mode_info = []
        for port in sorted(self.port_modes.keys()):
            for mode in sorted(self.port_modes[port].keys()):
                Zw = self.get_port_wave_impedance(port, mode, freq)
                impedances.append(Zw)
                mode_info.append({
                    'port': port,
                    'mode': mode,
                    'type': self.port_mode_types[port][mode],
                    'Z': Zw,
                    'indices': self.port_mode_indices.get(port, {}).get(mode),
                    'polarization_deg': np.degrees(self.port_mode_polarizations.get(port, {}).get(mode, 0)) % 360,
                    'degeneracy': self.port_mode_degeneracies.get(port, {}).get(mode, 1)
                })
        return np.diag(impedances), mode_info

    def get_num_modes(self, port: str) -> int:
        return len(self.port_modes.get(port, {}))

    def get_total_num_modes(self) -> int:
        return sum(len(modes) for modes in self.port_modes.values())

    def get_mode_name(self, port: str, mode: int) -> str:
        mtype = self.port_mode_types[port][mode]
        indices = self.port_mode_indices.get(port, {}).get(mode)
        if indices is not None:
            return f"{mtype}_{indices[0]}{indices[1]}"
        return f"{mtype}_mode{mode}"

    def get_mode_degeneracy(self, port: str, mode: int) -> int:
        return self.port_mode_degeneracies.get(port, {}).get(mode, 1)

    def is_mode_degenerate(self, port: str, mode: int) -> bool:
        return self.get_mode_degeneracy(port, mode) > 1

    def print_mode_summary(self) -> None:
        print(f"\n{'=' * 70}")
        print(f"Port Mode Summary (mode_source={self.mode_source})")
        print(f"{'=' * 70}")
        for port in sorted(self.port_modes.keys()):
            geom = self.port_geometries[port]
            print(f"\n{port}: {geom.type.value}")
            if geom.type == PortGeometryType.RECTANGULAR:
                print(f"  a={geom.a:.6f}, b={geom.b:.6f}")
            elif geom.type == PortGeometryType.CIRCULAR:
                print(f"  R={geom.radius:.6f}")
            print(f"  sigma={self.port_orientation_factors[port]:+.0f}")
            print(f"  {'Mode':<6}{'Type':<6}{'Indices':<10}{'fc [GHz]':<12}{'kc':<10}")
            print(f"  {'-' * 44}")
            for mode in sorted(self.port_modes[port].keys()):
                kc = self.port_cutoff_kc[port][mode]
                fc = c0 * kc / (2 * np.pi) / 1e9
                mtype = self.port_mode_types[port][mode]
                idx = self.port_mode_indices.get(port, {}).get(mode)
                idx_str = f"({idx[0]},{idx[1]})" if idx else "-"
                print(f"  {mode:<6}{mtype:<6}{idx_str:<10}{fc:<12.4f}{kc:<10.4f}")
        print(f"{'=' * 70}")

    def print_info(self) -> None:
        print("\n" + "=" * 70)
        print("Port Eigenmode Solver Information")
        print("=" * 70)

        print(f"Mode source:          {self.mode_source.upper()}")
        print(f"Polynomial order:     {self.order}")
        print(f"Dirichlet BC label:   {self.bc}")
        print(f"Polarization angle:   {np.degrees(self.polarization_angle):.1f}°")
        print(f"Ensure inward power:  {self.ensure_inward_power}")
        print(f"Global up direction:  {self.global_up}")
        print(f"Propagation axis:     {self.propagation_axis}")
        print(f"Geometry tolerance:   {self.geometry_tolerance:.4f}")

        ports = sorted(self.port_modes.keys())
        n_ports = len(ports)

        if n_ports == 0:
            print("\nNo ports have been solved yet.")
            print("=" * 70)
            return

        total_modes = self.get_total_num_modes()

        print(f"\nNumber of detected/solved ports: {n_ports}")
        print(f"Total number of modes:           {total_modes}")

        print("\n" + "-" * 70)
        print("Per-port summary:")
        print("-" * 70)

        for port in ports:
            geom = self.port_geometries.get(port, None)
            if geom is None:
                print(f"  {port}:  (geometry not detected)")
                continue

            print(f"  {port}:")
            print(f"    Geometry:       {geom.type.value}")
            print(f"    Fit error:      {geom.fit_error:.4f}")

            if geom.type == PortGeometryType.RECTANGULAR:
                print(f"    Dimensions:     a = {geom.a:.6f}, b = {geom.b:.6f}")
            elif geom.type == PortGeometryType.CIRCULAR:
                print(f"    Radius:         R = {geom.radius:.6f}")

            print(f"    Area:           {geom.area:.6e}")
            print(f"    Orientation σ:  {self.port_orientation_factors.get(port, 0):+.0f}")
            print(f"    Normal:         {self.port_normals.get(port, np.array([np.nan]*3))}")

            n_modes_port = self.get_num_modes(port)
            if n_modes_port > 0:
                print(f"    Modes found:    {n_modes_port}")
                print(f"    Lowest fc:      {self.get_cutoff_frequency(port, 0)/1e9:.4f} GHz")
                if n_modes_port > 1:
                    print(f"    Highest fc:     {self.get_cutoff_frequency(port, n_modes_port-1)/1e9:.4f} GHz")

                mode_list = []
                for m in range(min(4, n_modes_port)):
                    name = self.get_mode_name(port, m)
                    typ = self.port_mode_types[port].get(m, "?")
                    kc = self.port_cutoff_kc[port].get(m, np.nan)
                    fc = c0 * kc / (2 * np.pi) / 1e9 if not np.isnan(kc) else np.nan
                    degen = self.port_mode_degeneracies[port].get(m, 1)
                    pol = self.port_mode_polarizations[port].get(m, None)
                    pol_str = f" pol={np.degrees(pol):.0f}°" if pol is not None else ""
                    degen_str = f" (degen={degen})" if degen > 1 else ""
                    mode_list.append(f"{name} ({typ}, {fc:.3f} GHz{degen_str}{pol_str})")

                if n_modes_port > 4:
                    mode_list.append("…")
                    last_name = self.get_mode_name(port, n_modes_port-1)
                    last_fc = self.get_cutoff_frequency(port, n_modes_port-1)/1e9
                    mode_list.append(f"{last_name} ({last_fc:.3f} GHz)")

                print("    Modes (example):")
                for line in mode_list:
                    print(f"      {line}")

            else:
                print("    No modes computed yet for this port")

            print()

        print("-" * 70)
        print(f"Total modes across all ports: {total_modes}")
        print(f"Analytic modes used:          {self.mode_source == 'analytic'}")
        print(f"Precomputed mass matrices:    {len(self.port_mass_matrices)} ports")

        if self.mode_source == 'analytic':
            print("Note: Phases are deterministic from analytic formulas (no mesh-dependent sign flip)")
        else:
            print("Note: Numeric modes have phase normalized w.r.t. tangent1 projection")

        print("=" * 70)