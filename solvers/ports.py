"""
Port eigenmode solver with analytic and numeric mode computation.

Analytic modes use exact mathematical formulas mapped to GridFunction.
The phase is deterministic from the formula - no mesh-dependent normalization.

OPTIMIZED VERSION (safe precomputation of boundary mass matrix)
"""

from typing import Dict, Tuple, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np
import scipy.sparse as sp

from ngsolve import (
    HCurl, BilinearForm, GridFunction, BND, Cross, Integrate, InnerProduct,
    TaskManager, Preconditioner, solvers, IdentityMatrix, curl, ds,
    CoefficientFunction, specialcf, x, y, z, sin, cos, sqrt, pi
)

from core.constants import c0, mu0, eps0, Z0


class PortGeometryType(Enum):
    """Supported port geometry types."""
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
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
        mode_source_internal: Literal['analytic', 'numeric'] = 'analytical',
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

        # Phase and polarization tracking
        self.port_phase_signs: Dict[str, Dict[int, float]] = {}
        self.port_mode_polarizations: Dict[str, Dict[int, float]] = {}
        self.port_mode_degeneracies: Dict[str, Dict[int, int]] = {}
        self.port_mode_indices: Dict[str, Dict[int, Tuple[int, int]]] = {}

        # OPTIMIZATION: precomputed mass matrices + keep BilinearForm alive
        self.port_mass_matrices: Dict[str, sp.csr_matrix] = {}
        self.port_mass_forms: Dict[str, BilinearForm] = {}

    # =========================================================================
    # Geometry Detection (unchanged)
    # =========================================================================

    def _compute_port_normal(self, port: str) -> np.ndarray:
        n = specialcf.normal(self.mesh.dim)
        port_region = self.mesh.Boundaries(port)

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
        port_region = self.mesh.Boundaries(port)
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
        return np.sign(dot)

    def _compute_polarization_vector(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        theta = self.polarization_angle
        return np.cos(theta) * t1 + np.sin(theta) * t2

    def _detect_port_geometry(self, port: str) -> PortGeometry:
        center, area = self._compute_port_centroid_and_area(port)
        normal = self._compute_port_normal(port)
        t1, t2 = self._compute_tangent_frame(normal)
        port_region = self.mesh.Boundaries(port)

        u = (x - center[0]) * t1[0] + (y - center[1]) * t1[1] + (z - center[2]) * t1[2]
        v = (x - center[0]) * t2[0] + (y - center[1]) * t2[1] + (z - center[2]) * t2[2]

        I_uu = float(Integrate(v * v, self.mesh, BND, definedon=port_region))
        I_vv = float(Integrate(u * u, self.mesh, BND, definedon=port_region))
        I_uv = float(Integrate(u * v, self.mesh, BND, definedon=port_region))

        rect_geom, rect_error = self._fit_rectangular(center, normal, t1, t2, area, I_uu, I_vv, I_uv)
        circ_geom, circ_error = self._fit_circular(center, normal, t1, t2, area, I_uu, I_vv, I_uv)

        if rect_error <= self.geometry_tolerance and rect_error <= circ_error:
            return rect_geom
        elif circ_error <= self.geometry_tolerance:
            return circ_geom
        else:
            return PortGeometry(
                type=PortGeometryType.UNKNOWN,
                center=center, normal=normal, t1=t1, t2=t2, area=area,
                fit_error=min(rect_error, circ_error)
            )

    def _fit_rectangular(self, center, normal, t1, t2, area, I_uu, I_vv, I_uv):
        if I_uu < 1e-20 or I_vv < 1e-20:
            return PortGeometry(type=PortGeometryType.RECTANGULAR, center=center,
                               normal=normal, t1=t1, t2=t2, area=area, fit_error=1.0), 1.0

        b_sq = 12 * I_uu / area
        a_sq = 12 * I_vv / area
        if b_sq < 0 or a_sq < 0:
            return PortGeometry(type=PortGeometryType.RECTANGULAR, center=center,
                               normal=normal, t1=t1, t2=t2, area=area, fit_error=1.0), 1.0

        b = np.sqrt(b_sq)
        a = np.sqrt(a_sq)
        area_error = abs(a * b - area) / area
        cross_error = abs(I_uv) / np.sqrt(I_uu * I_vv) if I_uu * I_vv > 0 else 0
        total_error = max(area_error, cross_error)

        if b > a:
            a, b = b, a
            t1, t2 = t2, -t1

        return PortGeometry(
            type=PortGeometryType.RECTANGULAR,
            center=center, normal=normal, t1=t1, t2=t2, area=area,
            a=a, b=b, fit_error=total_error
        ), total_error

    def _fit_circular(self, center, normal, t1, t2, area, I_uu, I_vv, I_uv):
        R = np.sqrt(area / np.pi)
        I_expected = np.pi * R**4 / 4

        if I_uu + I_vv < 1e-20:
            return PortGeometry(type=PortGeometryType.CIRCULAR, center=center,
                               normal=normal, t1=t1, t2=t2, area=area, fit_error=1.0), 1.0

        isotropy_error = abs(I_uu - I_vv) / (I_uu + I_vv) * 2
        I_avg = (I_uu + I_vv) / 2
        magnitude_error = abs(I_avg - I_expected) / I_expected if I_expected > 0 else 1.0
        cross_error = abs(I_uv) / I_avg if I_avg > 0 else 0
        total_error = max(isotropy_error, magnitude_error, cross_error)

        return PortGeometry(
            type=PortGeometryType.CIRCULAR,
            center=center, normal=normal, t1=t1, t2=t2, area=area,
            radius=R, fit_error=total_error
        ), total_error

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

    def _generate_circular_modes(self, geometry: PortGeometry, nmodes: int) -> List[AnalyticMode]:
        from scipy.special import jn_zeros, jnp_zeros

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

    def solve(self, nmodes: int = 1) -> Tuple[Dict, Dict]:
        print("\n\t" + "=" * 60)
        print("\tCalculating Port Eigenmodes...")
        print("\t" + "=" * 60)
        print(f"\t  Mode source: {self.mode_source}")
        print(f"\t  Polarization angle: {np.degrees(self.polarization_angle):.1f}°")
        print(f"\t  Requested modes per port: {nmodes}")
        print("\t" + "-" * 60)

        ports = [b for b in self.mesh.GetBoundaries() if 'port' in b.lower()]
        if not ports:
            raise ValueError("No ports found in mesh")

        fes_full = HCurl(self.mesh, order=self.order, dirichlet=self.bc)

        # Detect geometry for all ports
        for port in ports:
            geometry = self._detect_port_geometry(port)
            self.port_geometries[port] = geometry
            self.port_normals[port] = geometry.normal
            self.port_tangent_frames[port] = (geometry.t1, geometry.t2)
            self.port_orientation_factors[port] = self._compute_orientation_factor(geometry.normal)
            self.port_polarizations[port] = self._compute_polarization_vector(geometry.t1, geometry.t2)

            print(f"\t  {port}: {geometry.type.value} (fit error: {geometry.fit_error:.4f})")
            if geometry.type == PortGeometryType.RECTANGULAR:
                print(f"\t    a={geometry.a:.6f}, b={geometry.b:.6f}")
            elif geometry.type == PortGeometryType.CIRCULAR:
                print(f"\t    R={geometry.radius:.6f}")

        if self.mode_source == 'analytic':
            unsupported = [p for p, g in self.port_geometries.items() if g.type == PortGeometryType.UNKNOWN]
            if unsupported:
                raise ValueError(
                    f"Analytic modes requested but ports {unsupported} have "
                    f"unsupported geometry. Use mode_source='numeric' instead."
                )

        # ────────────────────────────────────────────────────────────────
        # OPTIMIZATION: Precompute boundary mass matrix ONCE per port
        # ────────────────────────────────────────────────────────────────
        print("\t  Precomputing boundary mass matrices (once per port)...")
        u_full, v_full = fes_full.TnT()
        self.port_mass_matrices.clear()
        self.port_mass_forms.clear()

        for port in ports:
            m_form = BilinearForm(InnerProduct(u_full.Trace(), v_full.Trace()) * ds(port))
            with TaskManager():
                m_form.Assemble()
            M_bnd = sp.csr_matrix(m_form.mat.CSR())
            self.port_mass_matrices[port] = M_bnd
            self.port_mass_forms[port] = m_form   # ← keeps C++ matrix alive

        print(f"\t    Done for {len(ports)} port(s)")

        # Solve for each port
        for port in ports:
            geometry = self.port_geometries[port]
            if self.mode_source == 'analytic':
                if self.mode_source_internal == 'numeric' and (port != ports[0] and port != ports[-1]):
                    self._solve_port_numeric(port, nmodes, fes_full)
                else:
                    self._solve_port_analytic(port, geometry, nmodes, fes_full)
            else:
                self._solve_port_numeric(port, nmodes, fes_full)

        print("\t" + "-" * 60)
        print(f"\tTotal modes: {sum(len(m) for m in self.port_modes.values())}")
        print("\t" + "=" * 60)

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
            else:
                mode_cf = self._create_circular_mode_cf(amode, geometry)

            mode_gf = GridFunction(fes_port)
            mode_gf.Set(mode_cf, definedon=port_region)

            norm_sq = float(np.real(Integrate(
                InnerProduct(mode_gf, mode_gf),
                self.mesh, BND, definedon=port_region
            )))

            if norm_sq < 1e-15:
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
                  f"kc={amode.kc:.4f}, σ={sigma:+.0f}")

            mode_idx += 1
        print('\n')

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
                    f"σ={sigma:+.0f}, phase={'+' if phase_sign > 0 else '-'}"
                    f"{pol_str}{degen_str}")

                mode_idx += 1
        print()
    
    # =========================================================================
    # Optimized basis vector creation
    # =========================================================================

    def _create_basis_vector(self, mode_gf: GridFunction, port: str, fes_full: HCurl) -> np.ndarray:
        """Create mass-weighted basis vector using precomputed matrix."""
        port_region = self.mesh.Boundaries(port)

        # Use precomputed mass matrix
        M_bnd = self.port_mass_matrices[port]

        full_gf = GridFunction(fes_full)
        full_gf.Set(mode_gf, definedon=port_region)

        coeffs = full_gf.vec.FV().NumPy().copy()
        return M_bnd @ coeffs

    # =========================================================================
    # All remaining methods unchanged
    # =========================================================================

    def _solve_eigenvalue_problem(self, fes_port, port, nmodes):
        """Solve for both TE and TM modes using curl-curl formulation."""
        port_region = self.mesh.Boundaries(port)
        
        # ========== TE Modes: curl-curl for E with Dirichlet BC ==========
        te_modes, te_cutoffs = self._solve_te_modes(port, nmodes + 5)
        
        # ========== TM Modes: curl-curl for H with natural BC ==========
        tm_modes, tm_cutoffs = self._solve_tm_modes(port, nmodes + 5)
        
        # Combine and sort by cutoff frequency
        all_modes = []
        for mode, kc in zip(te_modes, te_cutoffs):
            all_modes.append((kc, mode, 'TE'))
        for mode, kc in zip(tm_modes, tm_cutoffs):
            all_modes.append((kc, mode, 'TM'))
        
        all_modes.sort(key=lambda x: x[0])
        
        modes = [m for _, m, _ in all_modes]
        cutoffs = [k for k, _, _ in all_modes]
        mode_types = [t for _, _, t in all_modes]
        
        return modes, cutoffs, mode_types


    def _solve_te_modes(self, port: str, nmodes: int):
        """
        Solve for TE modes using curl-curl formulation for E.
        
        TE modes: E is purely transverse, n × E = 0 on PEC (Dirichlet BC)
        """
        # HCurl space WITH Dirichlet BC on conducting walls
        fes_te = HCurl(
            self.mesh, order=self.order,
            dirichlet=self.bc,
            definedon=self.mesh.Boundaries(port)
        )
        
        port_region = self.mesh.Boundaries(port)
        u, v = fes_te.TnT()
        
        a = BilinearForm(curl(u.Trace()) * curl(v.Trace()) * ds(port))
        m = BilinearForm(u.Trace() * v.Trace() * ds(port))
        apre = BilinearForm((curl(u).Trace() * curl(v).Trace() + u.Trace() * v.Trace()) * ds(port))
        pre = Preconditioner(apre, type="multigrid")
        
        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()
            
            # Gradient projection to remove null-space
            G, fes_h1 = fes_te.CreateGradient()
            GT = G.CreateTranspose()
            math1 = GT @ m.mat @ G
            invh1 = math1.Inverse(freedofs=fes_h1.FreeDofs())
            proj = IdentityMatrix(fes_te.ndof) - G @ invh1 @ GT @ m.mat
            projpre = proj @ pre
            
            evals, evecs = solvers.PINVIT(
                a.mat, m.mat, pre=projpre, num=nmodes, maxit=50, printrates=False)
        
        modes, cutoffs = [], []
        for i, ev in enumerate(evals):
            if ev > 1e-6:
                mode = GridFunction(fes_te)
                mode.vec.data = evecs[i]
                norm_sq = float(np.real(Integrate(
                    InnerProduct(mode, mode), self.mesh, BND, definedon=port_region)))
                if norm_sq > 1e-15:
                    mode.vec.data /= np.sqrt(norm_sq)
                    modes.append(mode)
                    cutoffs.append(np.sqrt(ev))
        
        return modes, cutoffs


    def _solve_tm_modes(self, port: str, nmodes: int):
        """
        Solve for TM modes using curl-curl formulation for H.
        
        TM modes: H is purely transverse, n × H = 0 on PEC (natural BC, no Dirichlet)
        
        The eigenvalue problem is the same:
            curl curl H = kc² H
        
        But with natural BC instead of essential BC.
        After solving for H, we can compute E_t from H if needed.
        """
        # HCurl space WITHOUT Dirichlet BC (natural BC for H on PEC)
        fes_tm = HCurl(
            self.mesh, order=self.order,
            # No dirichlet parameter - natural BC
            definedon=self.mesh.Boundaries(port)
        )
        
        port_region = self.mesh.Boundaries(port)
        u, v = fes_tm.TnT()
        
        a = BilinearForm(curl(u.Trace()) * curl(v.Trace()) * ds(port))
        m = BilinearForm(u.Trace() * v.Trace() * ds(port))
        apre = BilinearForm((curl(u).Trace() * curl(v).Trace() + u.Trace() * v.Trace()) * ds(port))
        pre = Preconditioner(apre, type="multigrid")
        
        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()
            
            # Gradient projection still needed to remove null-space
            G, fes_h1 = fes_tm.CreateGradient()
            GT = G.CreateTranspose()
            math1 = GT @ m.mat @ G
            invh1 = math1.Inverse(freedofs=fes_h1.FreeDofs())
            proj = IdentityMatrix(fes_tm.ndof) - G @ invh1 @ GT @ m.mat
            projpre = proj @ pre
            
            evals, evecs = solvers.PINVIT(
                a.mat, m.mat, pre=projpre, num=nmodes, maxit=50, printrates=False)
        
        # For TM modes, we solved for H but we need E for port excitation
        # E_t and H_t are related by: E_t = -Z_TM * (n × H_t)
        # For mode patterns, we can use n × H directly (rotation by 90°)
        
        normal = self.port_normals[port]
        n_cf = CoefficientFunction(tuple(normal))
        
        modes, cutoffs = [], []
        for i, ev in enumerate(evals):
            if ev > 1e-6:
                H_mode = GridFunction(fes_tm)
                H_mode.vec.data = evecs[i]
                
                # Convert H to E: E_t ∝ n × H_t
                # Create E mode in same space
                E_cf = Cross(n_cf, H_mode)
                
                E_mode = GridFunction(fes_tm)
                E_mode.Set(E_cf, definedon=port_region)
                
                # Normalize
                norm_sq = float(np.real(Integrate(
                    InnerProduct(E_mode, E_mode), self.mesh, BND, definedon=port_region)))
                
                if norm_sq > 1e-15:
                    E_mode.vec.data /= np.sqrt(norm_sq)
                    modes.append(E_mode)
                    cutoffs.append(np.sqrt(ev))
        
        return modes, cutoffs

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
    # Wave Impedance & Utility Methods (unchanged)
    # ────────────────────────────────────────────────────────────────────────

    def get_port_wave_impedance(self, port: str, mode: int, freq: float) -> complex:
        kc = self.port_cutoff_kc[port][mode]
        wc = kc * c0
        s = 1j * 2 * np.pi * freq
        sqrt_term = np.sqrt(s**2 + wc**2)
        mode_type = self.port_mode_types[port][mode]
        if mode_type == 'TE':
            return complex(s * Z0 / sqrt_term)
        else:
            return complex(Z0 * sqrt_term / s)

    def get_port_wave_impedance_matrix(self, freq: float) -> np.ndarray:
        impedances = []
        for port in sorted(self.port_modes.keys()):
            for mode in sorted(self.port_modes[port].keys()):
                impedances.append(self.get_port_wave_impedance(port, mode, freq))
        print(impedances)
        return np.diag(impedances)

    def get_propagation_constant(self, port: str, mode: int, freq: float) -> complex:
        kc = self.port_cutoff_kc[port][mode]
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
            print(f"  σ={self.port_orientation_factors[port]:+.0f}")
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