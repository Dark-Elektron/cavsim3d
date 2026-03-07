"""Frequency-domain electromagnetic solver for single and compound structures."""

from typing import Dict, List, Optional, Tuple, Union, Literal
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sl
from solvers.eigen_mixin import FDSEigenMixin

from ngsolve import (
    HCurl, BilinearForm, LinearForm, GridFunction, BND,
    Integrate, InnerProduct, TaskManager, curl, dx, ds,
    BoundaryFromVolumeCF, CoefficientFunction, Norm, BaseVector
)
from ngsolve.webgui import Draw

# Iterative: preconditioner MUST be registered before assembly
from ngsolve import Preconditioner as NGPreconditioner
from ngsolve import InnerProduct as NGInnerProduct
from ngsolve.krylovspace import GMRes

from core.constants import mu0, eps0, c0, Z0
from solvers.base import BaseEMSolver, ParameterConverter
from solvers.ports import PortEigenmodeSolver


class FrequencyDomainSolver(BaseEMSolver, FDSEigenMixin):
    """
    Frequency-domain solver for electromagnetic problems.

    Handles both single-domain and compound (multi-domain) structures.
    For compound structures, two methods are available for computing
    global S/Z-parameters:

    1. **Coupled** (default): Solves the full system with all domains coupled.
       Most accurate, captures all inter-domain reflections and resonances.

    2. **Cascade**: Solves each domain independently, then cascades S-matrices.
       Faster but assumes negligible reflections at internal ports.

    Conventions:
    - Time convention: exp(+jωt)
    - Z-parameters: V_n = Σ_m Z_nm * I_m
    - Modes normalized to ∫|E_t|² dS = 1

    Parameters
    ----------
    geometry : BaseGeometry
        Geometry object with mesh
    order : int
        Polynomial order for finite elements
    bc : str, optional
        Boundary condition specification. If None, uses geometry.bc
    use_wave_impedance : bool
        Whether to use frequency-dependent wave impedance for S-parameters

    Examples
    --------
    >>> # Single domain solve
    >>> fds = FrequencyDomainSolver(geometry, order=3)
    >>> fds.assemble_matrices(nportmodes=1)
    >>> fds.solve(1, 10, 100)
    >>> fds.plot_s_parameters()

    >>> # Compound structure with coupled method (default, most accurate)
    >>> fds = FrequencyDomainSolver(compound_geometry, order=3)
    >>> fds.solve(1, 10, 100, global_method='coupled')

    >>> # Compound structure with cascade method (faster, less accurate)
    >>> fds.solve(1, 10, 100, global_method='cascade')

    >>> # Per-domain results only (for ROM training)
    >>> fds.solve(1, 10, 100, per_domain=True, global_method=None)

    >>> # Compare both methods
    >>> comparison = fds.compare_methods(1, 10, 100)
    """

    # --- Iterative solver defaults ---
    AUTO_DOF_THRESHOLD = 50_000
    DEFAULT_ITERATIVE_OPTS = {
        'precond': 'multigrid',
        'maxsteps': 5000,
        'tol': 1e-6,
        'printrates': False,
    }

    def __init__(
        self,
        geometry,
        order: int = 3,
        bc: Optional[str] = None,
        use_wave_impedance: bool = True
    ):
        super().__init__()

        self.geometry = geometry
        self.mesh = geometry.mesh
        self.bc = bc if bc is not None else getattr(geometry, 'bc', None)
        self.order = order
        self.use_wave_impedance = use_wave_impedance

        # Detect structure topology
        self.domains = self._detect_domains()
        self._ports = self._detect_ports()
        self.domain_port_map = self._build_domain_port_map()
        self.n_domains = len(self.domains)
        self._n_ports = len(self._ports)
        self.is_compound = self.n_domains > 1

        # External ports (for compound structures, first and last only)
        self._external_ports: List[str] = self._identify_external_ports()
        self._internal_ports: List[str] = self._identify_internal_ports()

        # Print structure info
        self._print_structure_info()

        # Per-domain storage
        self._fes: Dict[str, HCurl] = {}
        self.M: Dict[str, sp.csr_matrix] = {}
        self.K: Dict[str, sp.csr_matrix] = {}
        self.B: Dict[str, np.ndarray] = {}

        # Global (coupled) storage
        self._fes_global: Optional[HCurl] = None
        self.M_global: Optional[sp.csr_matrix] = None
        self.K_global: Optional[sp.csr_matrix] = None
        self.B_global: Optional[np.ndarray] = None

        # Port modes (shared across domains)
        self.port_solver = PortEigenmodeSolver(self.mesh, order, self.bc)
        self.port_modes: Dict[str, Dict[int, CoefficientFunction]] = None
        self.port_basis: Dict[str, Dict[int, np.ndarray]] = None
        self._n_modes_per_port: int = None

        # Per-domain results (internal storage)
        self._Z_per_domain: Dict[str, Dict[str, np.ndarray]] = {}
        self._S_per_domain: Dict[str, Dict[str, np.ndarray]] = {}

        # Global results storage
        self._Z_global_coupled: Optional[np.ndarray] = None
        self._S_global_coupled: Optional[np.ndarray] = None
        self._Z_global_cascade: Optional[np.ndarray] = None
        self._S_global_cascade: Optional[np.ndarray] = None

        # Track which method was used for current results
        self._current_global_method: Optional[str] = None

        # Snapshots storage
        self.snapshots: Dict[str, np.ndarray] = {}

        # Assembly state flags
        self._global_matrices_assembled: bool = False
        self._per_domain_matrices_assembled: bool = False

        # Result-object caches (built lazily via .fom / .foms)
        self._fom_cache = None
        self._foms_cache = None

        # Validate boundary conditions
        self._validate_boundary_conditions()

    def _validate_boundary_conditions(self) -> None:
        """
        Validate that boundary conditions are properly set.

        Issues warnings (not errors) for common misconfigurations so that
        the user can fix them before calling :meth:`solve`.
        """
        boundaries = list(self.mesh.GetBoundaries())
        ports = [b for b in boundaries if 'port' in b.lower()]
        unnamed = [b for b in boundaries if b in ('', None)]
        unique_boundaries = sorted(set(b for b in boundaries if b))

        if not ports:
            warnings.warn(
                f"\n  No port boundaries detected in mesh.\n"
                f"  Boundaries found: {unique_boundaries or '(none)'}\n"
                f"  The solver requires at least one port to compute "
                f"Z/S parameters.\n"
                f"  Fix: call geo.define_ports(zmin=True, zmax=True) "
                f"before generate_mesh().",
                UserWarning,
                stacklevel=2,
            )

        if unnamed:
            warnings.warn(
                f"\n  {len(unnamed)} boundary face(s) have no name and will "
                f"NOT have PEC conditions applied.\n"
                f"  All boundaries: {boundaries}\n"
                f"  Fix: call geo.define_ports() or rebuild the geometry.",
                UserWarning,
                stacklevel=2,
            )

        if self.bc in (None, ''):
            warnings.warn(
                f"\n  No boundary conditions set (bc={self.bc!r}). All "
                f"boundaries will be unconstrained (no PEC walls).\n"
                f"  All boundaries: {unique_boundaries or '(none)'}\n"
                f"  Fix: call geo.define_ports() to set up the geometry "
                f"correctly, which also sets bc='wall'.",
                UserWarning,
                stacklevel=2,
            )

    # === BaseEMSolver abstract implementations ===

    # --- Result-object navigation (additive, lazy-cached) ---

    @property
    def fom(self):
        """
        Global FOM result for this solver as a :class:`~solvers.results.FOMResult`.

        Returns the coupled (global) solve result for single-solid structures,
        or the coupled global solve for multi-solid structures.

        Requires that :meth:`solve` has been called first.

        Example
        -------
        >>> fig, ax = fds.fom.plot_s()
        >>> fig, ax = fds.fom.rom.plot_s(ax=ax)
        """
        if self._fom_cache is None:
            if self._Z_matrix is None:
                raise RuntimeError(
                    "No global FOM results available. "
                    "Call fds.solve() first (global_method='coupled' or 'concatenate')."
                )
            from solvers.results import build_fom_result
            self._fom_cache = build_fom_result(self, domain='global')
        return self._fom_cache

    @property
    def foms(self):
        """
        Per-domain FOM results as a :class:`~solvers.results.FOMCollection`.

        Only available for **multi-solid** (compound) structures.
        Requires that :meth:`solve` was called with ``per_domain=True``.

        Example
        -------
        >>> fds.foms[0].plot_s()            # first domain
        >>> fds.foms.concat.plot_s()        # concatenated FOM
        >>> fds.foms.roms.concat.rom.plot_s()  # full chain
        """
        if self._foms_cache is None:
            from solvers.results import build_fom_collection
            self._foms_cache = build_fom_collection(self)
        return self._foms_cache

    @property
    def n_ports(self) -> int:
        """Number of ports (external ports only for compound structures after solve)."""
        if self._current_global_method is not None and self.is_compound:
            return len(self._external_ports)
        return self._n_ports

    @property
    def ports(self) -> List[str]:
        """List of port names (external only for compound structures after solve)."""
        if self._current_global_method is not None and self.is_compound:
            return self._external_ports.copy()
        return self._ports.copy()

    @property
    def all_ports(self) -> List[str]:
        """List of all port names including internal ports."""
        return self._ports.copy()

    @property
    def external_ports(self) -> List[str]:
        """List of external port names."""
        return self._external_ports.copy()

    @property
    def internal_ports(self) -> List[str]:
        """List of internal port names (empty for single-domain)."""
        return self._internal_ports.copy()

    def _identify_external_ports(self) -> List[str]:
        """Identify external ports (first and last for compound structures)."""
        if not self.is_compound:
            return self._ports.copy()
        return [self._ports[0], self._ports[-1]]

    def _identify_internal_ports(self) -> List[str]:
        """Identify internal ports (between domains in compound structures)."""
        if not self.is_compound:
            return []
        return self._ports[1:-1]

    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        """Get port wave impedance."""
        if self.use_wave_impedance and self.port_modes is not None:
            return self.port_solver.get_port_wave_impedance(port, mode, freq)
        return Z0

    # === Structure detection ===

    def _detect_domains(self) -> List[str]:
        """Detect domains from mesh materials."""
        materials = list(self.mesh.GetMaterials())

        # Look for 'cell' naming convention from split geometry
        domains = sorted(
            [m for m in materials if 'cell' in m.lower()],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )

        if not domains:
            # Single domain case
            domains = ['default'] if 'default' in materials else [materials[0]]

        return domains

    def _detect_ports(self) -> List[str]:
        """Detect ports from mesh boundaries."""
        boundaries = list(self.mesh.GetBoundaries())
        ports = [b for b in boundaries if 'port' in b.lower()]

        def get_port_number(port_name):
            digits = ''.join(filter(str.isdigit, port_name))
            return int(digits) if digits else 0

        return sorted(ports, key=get_port_number)

    def _build_domain_port_map(self) -> Dict[str, List[str]]:
        """Map each domain to its adjacent ports."""
        mapping = {}
        n_domains = len(self.domains)
        n_ports = len(self._ports)

        if n_domains == 1:
            mapping[self.domains[0]] = self._ports
        else:
            # Multiple domains - sequential mapping
            if n_ports != n_domains + 1:
                print(f"Warning: Expected {n_domains + 1} ports for {n_domains} domains, "
                      f"found {n_ports}. Using sequential assignment.")

            for i, domain in enumerate(self.domains):
                if i + 1 < n_ports:
                    mapping[domain] = [self._ports[i], self._ports[i + 1]]
                else:
                    mapping[domain] = [self._ports[i]]

        return mapping

    def _print_structure_info(self) -> None:
        """Print detected structure information."""
        print("\n" + "=" * 60)
        print("Structure Topology")
        print("=" * 60)
        print(f"Type: {'Compound' if self.is_compound else 'Single'} structure")
        print(f"Domains ({self.n_domains}): {self.domains}")
        print(f"Total Ports ({self._n_ports}): {self._ports}")

        if self.is_compound:
            print(f"External Ports ({len(self._external_ports)}): {self._external_ports}")
            print(f"Internal Ports ({len(self._internal_ports)}): {self._internal_ports}")

        print("\nDomain-Port Mapping:")
        for domain, ports in self.domain_port_map.items():
            port_types = []
            for p in ports:
                if p in self._external_ports:
                    if p == self._ports[0]:
                        port_types.append(f"{p} (external, input)")
                    else:
                        port_types.append(f"{p} (external, output)")
                else:
                    port_types.append(f"{p} (internal)")
            print(f"  {domain}: {port_types}")
        print("=" * 60)

    # === Matrix assembly ===

    def assemble_matrices(
        self,
        nportmodes: int = 1,
        assemble_global: bool = True,
        assemble_per_domain: bool = True
    ) -> Dict[str, Tuple]:
        """
        Assemble frequency-independent system matrices.

        Parameters
        ----------
        nportmodes : int
            Number of modes to compute per port
        assemble_global : bool
            Assemble global (full-structure) matrices for coupled solve.
            Required for global_method='coupled'.
        assemble_per_domain : bool
            Assemble per-domain matrices. Required for per_domain=True
            or global_method='cascade'.

        Returns
        -------
        dict
            Dictionary summarizing assembled matrices
        """
        print("\n" + "=" * 60)
        print("Assembling Matrices...")
        print("=" * 60)

        # For single-domain, per-domain and global are the same
        if not self.is_compound:
            assemble_global = True
            assemble_per_domain = True

        # Solve port eigenmodes first (shared across all assemblies)
        if self.port_modes is None:
            print("\nSolving port eigenmodes...")
            self.port_modes, self.port_basis = self.port_solver.solve(nmodes=nportmodes)
            self._n_modes_per_port = nportmodes

        # Assemble per-domain matrices if requested
        if assemble_per_domain and not self._per_domain_matrices_assembled:
            self._assemble_per_domain_matrices()
            self._per_domain_matrices_assembled = True

        # Assemble global matrices if requested
        if assemble_global and not self._global_matrices_assembled:
            self._assemble_global_matrices()
            self._global_matrices_assembled = True

        return self._get_matrix_summary()

    def _assemble_per_domain_matrices(self) -> None:
        """Assemble matrices for each domain independently."""
        print("\n--- Assembling Per-Domain Matrices ---")

        for domain in self.domains:
            print(f"\nDomain: {domain}")

            # Create FES for this domain
            fes = HCurl(
                self.mesh,
                order=self.order,
                dirichlet=self.bc,
                definedon=self.mesh.Materials(domain)
            )
            self._fes[domain] = fes

            u, v = fes.TnT()

            # Stiffness and mass matrices
            k_form = BilinearForm((1 / mu0) * curl(u) * curl(v) * dx(domain))
            m_form = BilinearForm(eps0 * u * v * dx(domain))

            with TaskManager():
                k_form.Assemble()
                m_form.Assemble()

            self.K[domain] = sp.csr_matrix(k_form.mat.CSR()).copy()
            self.M[domain] = sp.csr_matrix(m_form.mat.CSR()).copy()

            # Port basis matrix for this domain
            self._construct_domain_basis_matrix(domain, fes)

            print(f"  FES ndof: {fes.ndof}")
            print(f"  K shape: {self.K[domain].shape}, nnz: {self.K[domain].nnz}")
            print(f"  M shape: {self.M[domain].shape}, nnz: {self.M[domain].nnz}")
            print(f"  B shape: {self.B[domain].shape}")

    def _assemble_global_matrices(self) -> None:
        """Assemble matrices for the full (coupled) structure."""
        print("\n--- Assembling Global Matrices (Coupled System) ---")

        # Create FES for entire mesh
        self._fes_global = HCurl(
            self.mesh,
            order=self.order,
            dirichlet=self.bc
        )
        fes = self._fes_global
        u, v = fes.TnT()

        # Stiffness and mass matrices for full structure
        k_form = BilinearForm((1 / mu0) * curl(u) * curl(v) * dx)
        m_form = BilinearForm(eps0 * u * v * dx)

        with TaskManager():
            k_form.Assemble()
            m_form.Assemble()

        self.K_global = sp.csr_matrix(k_form.mat.CSR()).copy()
        self.M_global = sp.csr_matrix(m_form.mat.CSR()).copy()

        # Port basis matrix for external ports
        self._construct_global_basis_matrix(fes)

        print(f"  Global FES ndof: {fes.ndof}")
        print(f"  K_global shape: {self.K_global.shape}, nnz: {self.K_global.nnz}")
        print(f"  M_global shape: {self.M_global.shape}, nnz: {self.M_global.nnz}")
        print(f"  B_global shape: {self.B_global.shape}")

    def _construct_domain_basis_matrix(self, domain: str, fes: HCurl) -> None:
        """
        Construct port basis matrix B for a specific domain.

        B is mass-weighted on the boundary:
            b = M_bnd @ (port_mode embedded in fes)
        and is additionally multiplied by the port orientation factor sigma
        to keep FOM/ROM conventions consistent.
        """
        domain_ports = self.domain_port_map[domain]
        basis_vectors = []

        u, v = fes.TnT()

        for port in domain_ports:
            if port not in self.port_basis:
                continue

            # Boundary mass matrix for this port
            m_bnd_form = BilinearForm(InnerProduct(u.Trace(), v.Trace()) * ds(port))
            with TaskManager():
                m_bnd_form.Assemble()
            M_bnd = sp.csr_matrix(m_bnd_form.mat.CSR())

            # Port sign convention
            # sigma = float(self.port_solver.port_orientation_factors.get(port, 1.0))

            for mode in sorted(self.port_basis[port].keys()):
                port_mode_cf = self.port_modes[port][mode]

                gf = GridFunction(fes)
                gf.Set(port_mode_cf, definedon=self.mesh.Boundaries(port))

                b_coeffs = gf.vec.FV().NumPy().copy()
                b_weighted = M_bnd @ b_coeffs

                # # Apply sigma so ROM uses same port convention as FOM
                # b_weighted *= sigma

                basis_vectors.append(b_weighted)

        if basis_vectors:
            self.B[domain] = np.array(basis_vectors).T
        else:
            self.B[domain] = np.zeros((fes.ndof, 0))

    def _construct_global_basis_matrix(self, fes: HCurl) -> None:
        """
        Construct port basis matrix for external ports in the global system.

        Uses the same boundary mass-weighting
        """
        target_ports = self._external_ports if self.is_compound else self._ports

        basis_vectors = []
        u, v = fes.TnT()

        for port in target_ports:
            if port not in self.port_basis:
                continue

            m_bnd_form = BilinearForm(InnerProduct(u.Trace(), v.Trace()) * ds(port))
            with TaskManager():
                m_bnd_form.Assemble()
            M_bnd = sp.csr_matrix(m_bnd_form.mat.CSR())

            # sigma = float(self.port_solver.port_orientation_factors.get(port, 1.0))

            for mode in sorted(self.port_basis[port].keys()):
                port_mode_cf = self.port_modes[port][mode]

                gf = GridFunction(fes)
                gf.Set(port_mode_cf, definedon=self.mesh.Boundaries(port))

                b_coeffs = gf.vec.FV().NumPy().copy()
                b_weighted = M_bnd @ b_coeffs

                # # Apply sigma so ROM uses same port convention as FOM
                # b_weighted *= sigma

                basis_vectors.append(b_weighted)

        self.B_global = np.array(basis_vectors).T if basis_vectors else np.zeros((fes.ndof, 0))

    def _get_matrix_summary(self) -> Dict:
        """Return summary of assembled matrices."""
        summary = {'per_domain': {}, 'global': None}

        for d in self.domains:
            if d in self.M:
                summary['per_domain'][d] = {
                    'M_shape': self.M[d].shape,
                    'K_shape': self.K[d].shape,
                    'B_shape': self.B[d].shape,
                    'ndof': self._fes[d].ndof if d in self._fes else None
                }

        if self.M_global is not None:
            summary['global'] = {
                'M_shape': self.M_global.shape,
                'K_shape': self.K_global.shape,
                'B_shape': self.B_global.shape,
                'ndof': self._fes_global.ndof if self._fes_global else None
            }

        return summary

    # === Iterative solver helpers ===

    def _resolve_solver_type(self, solver_type: str, fes) -> str:
        """Resolve 'auto' to 'direct' or 'iterative' based on DOF count."""
        if solver_type != 'auto':
            return solver_type
        ndof = fes.ndof if fes is not None else 0
        chosen = 'iterative' if ndof > self.AUTO_DOF_THRESHOLD else 'direct'
        print(f"  Auto solver: {ndof} DOFs → '{chosen}' "
              f"(threshold: {self.AUTO_DOF_THRESHOLD})")
        return chosen

    def _merge_iterative_opts(self, user_opts: Optional[Dict]) -> Dict:
        """Merge user-supplied iterative options with defaults."""
        opts = dict(self.DEFAULT_ITERATIVE_OPTS)
        if user_opts:
            opts.update(user_opts)
        return opts

    def _prepare_iterative(self, fes, opts: Dict):
        """Prepare FES for iterative solve (logging only, no mesh mutation)."""
        print(f"  Iterative mode: precond='{opts.get('precond', 'multigrid')}', "
              f"tol={opts.get('tol', 1e-6)}, maxsteps={opts.get('maxsteps', 5000)}")
        print(f"  FES ndof: {fes.ndof}")
        return fes

    def _solve_system(self, fes, a_form, f_vec, precond, opts: Dict):
        """
        Solve a_form * x = f_vec using direct or iterative method.

        Parameters
        ----------
        fes : NGSolve FESpace
        a_form : BilinearForm  (NOT yet assembled for iterative path)
        f_vec : BaseVector
        solver_type : 'direct' or 'iterative'
        opts : dict with iterative solver options

        Returns
        -------
        x : BaseVector  (solution)
        iters : int     (0 for direct, GMRES steps for iterative)
        residual : float  (relative residual ||Ax-b||/||b||, 0.0 for direct)
        """

        # Count iterations via callback
        iter_count = [0]
        def _count_iter(sol_vec):
            iter_count[0] += 1

        # GMRes is a function: GMRes(A, b, pre=...) → solution vector
        x = GridFunction(fes)
        sol = GMRes(
            A=a_form.mat,
            b=f_vec,
            pre=precond.mat,
            maxsteps=opts['maxsteps'],
            tol=opts['tol'],
            printrates=opts['printrates'],
            callback=_count_iter,
        )
        x.vec.data = sol
        iters = iter_count[0]

        # Compute residual on FREE DOFs only
        r = x.vec.CreateVector()
        r.data = a_form.mat * x.vec - f_vec
        
        # Get numpy arrays
        r_np = r.FV().NumPy()
        f_np = f_vec.FV().NumPy()
        
        # Extract free DOF indices
        free_idx = np.array([i for i in range(fes.ndof) if fes.FreeDofs()[i]])
        
        # Compute norms on free DOFs only
        r_free = r_np[free_idx]
        f_free = f_np[free_idx]
        
        r_norm_free = np.linalg.norm(r_free)
        b_norm_free = np.linalg.norm(f_free)
        
        rel_res = r_norm_free / b_norm_free if b_norm_free > 0 else r_norm_free

        return x.vec, iters, rel_res

    # === Frequency domain solve ===

    def solve(
            self,
            fmin: float,
            fmax: float,
            nsamples: int = 100,
            store_snapshots: bool = True,
            compute_s_params: bool = True,
            per_domain: bool = True,
            global_method: Literal['coupled', None] = 'coupled',
            solver_type: str = 'iterative',
            iterative_opts: Optional[Dict] = None,
    ) -> Dict:
        """
        Solve frequency sweep.

        For compound structures, three methods are available for computing
        global (full-structure) S/Z-parameters:

        Parameters
        ----------
        fmin : float
            Minimum frequency in GHz
        fmax : float
            Maximum frequency in GHz
        nsamples : int
            Number of frequency samples
        store_snapshots : bool
            Store solution vectors for ROM or field visualization
        compute_s_params : bool
            Compute S-parameters from Z-parameters
        per_domain : bool
            If True, solve each domain independently and store per-domain Z/S.
        global_method : {'coupled', None}
            Method for computing global (full-structure) S/Z parameters:

            - 'coupled': Solve entire structure as one system (default).
                         Most accurate, captures all inter-domain interactions.

            - None: Skip global computation, only compute per-domain if requested.

        solver_type : {'auto', 'direct', 'iterative'}
            Linear system solver strategy:

            - 'auto': Use direct for small problems (< AUTO_DOF_THRESHOLD DOFs),
                      iterative for larger ones.
            - 'direct': Always use sparse direct factorisation.
            - 'iterative': Always use iterative solver (preconditioner).

        iterative_opts : dict, optional
            Options for the iterative solver, merged with DEFAULT_ITERATIVE_OPTS::

                {
                    'precond': 'multigrid',      # preconditioner type
                    'maxsteps': 5000,        # max GMRES iterations
                    'tol': 1e-6,            # convergence tolerance
                    'printrates': False,     # print GMRES residuals
                }

        Returns
        -------
        dict
            Results dictionary
        """
        self.frequencies = np.linspace(fmin, fmax, nsamples) * 1e9

        # Merge iterative options with defaults
        iter_opts = self._merge_iterative_opts(iterative_opts)

        # Normalize options for single-domain structures
        if not self.is_compound:
            per_domain = False
            global_method = 'coupled'

        # Cascade and concatenate removed — only 'coupled' or None
        if global_method is not None and global_method != 'coupled':
            raise ValueError(
                f"Unknown global_method '{global_method}'. "
                f"Use 'coupled' or None."
            )

        # Validate options
        if global_method is None and not per_domain:
            raise ValueError(
                "At least one of 'per_domain=True' or 'global_method' must be specified."
            )

        # Ensure required matrices are assembled
        self._ensure_matrices_assembled(per_domain, global_method)

        # Clear previous results
        self._clear_results()

        # Print solve configuration
        self._print_solve_config(per_domain, global_method)

        # Solve per-domain if requested
        if per_domain:
            print("\n" + "-" * 40)
            print("Per-Domain Solve")
            print("-" * 40)
            self._solve_per_domain(store_snapshots,
                                   solver_type=solver_type,
                                   iter_opts=iter_opts)
        else:
        # Compute global results
        # if global_method == 'coupled':
            print("\n" + "-" * 40)
            print("Global Coupled Solve")
            print("-" * 40)
            self._solve_global_coupled(store_snapshots,
                                       solver_type=solver_type,
                                       iter_opts=iter_opts)
            self._current_global_method = 'coupled'

        # Compute S-parameters from Z
        if compute_s_params:
            if global_method is not None:
                self._compute_s_from_z()
            if per_domain:
                self._compute_per_domain_s_from_z()

        self._invalidate_cache()

        return self._build_results_dict(compute_s_params, per_domain, global_method)

    def _ensure_matrices_assembled(
        self,
        per_domain: bool,
        global_method: Optional[str]
    ) -> None:
        """Ensure required matrices are assembled."""
        needs_global = (global_method == 'coupled')
        needs_per_domain = per_domain

        # Check if port modes exist
        if self.port_modes is None:
            self.assemble_matrices(
                assemble_global=needs_global,
                assemble_per_domain=needs_per_domain
            )
            return

        # Assemble missing matrices
        if needs_global and not self._global_matrices_assembled:
            self._assemble_global_matrices()
            self._global_matrices_assembled = True

        if needs_per_domain and not self._per_domain_matrices_assembled:
            self._assemble_per_domain_matrices()
            self._per_domain_matrices_assembled = True

    def _clear_results(self) -> None:
        """Clear previous solve results."""
        self._Z_per_domain = {}
        self._S_per_domain = {}
        self._Z_global_coupled = None
        self._S_global_coupled = None
        self._current_global_method = None
        self.snapshots = {}
        self._residuals = {}
        # Invalidate result-object caches
        self._fom_cache = None
        self._foms_cache = None

    def _print_solve_config(
        self,
        per_domain: bool,
        global_method: Optional[str]
    ) -> None:
        """Print solve configuration."""
        print("\n" + "=" * 60)
        print("Frequency Domain Solve Configuration")
        print("=" * 60)
        print(f"Frequency range: {self.frequencies[0]/1e9:.4f} - {self.frequencies[-1]/1e9:.4f} GHz")
        print(f"Number of samples: {len(self.frequencies)}")
        print(f"Structure type: {'Compound' if self.is_compound else 'Single'}")
        print(f"Per-domain solve: {per_domain}")
        print(f"Global method: {global_method}")
        if self.is_compound:
            print(f"External ports: {self._external_ports}")
            print(f"Internal ports: {self._internal_ports}")
        print("=" * 60)

    def _solve_per_domain(
        self,
        store_snapshots: bool,
        solver_type: str = 'auto',
        iter_opts: Optional[Dict] = None,
    ) -> None:
        """Solve each domain independently using NGSolve with fast Z-extraction."""
        import time

        iter_opts = iter_opts or self._merge_iterative_opts(None)

        for domain in self.domains:
            t_domain_start = time.time()
            print(f"\nSolving domain: {domain}")

            domain_ports = self.domain_port_map[domain]
            fes = self._fes[domain]

            # Resolve solver type for this domain
            st = self._resolve_solver_type(solver_type, fes)
            print(f"  Solver type: {st}")

            # Prepare multigrid if iterative
            if st == 'iterative':
                fes = self._prepare_iterative(fes, iter_opts)
                self._fes[domain] = fes

            # Get pre-assembled B matrix for fast Z extraction
            # B has shape (ndof, n_port_modes) where columns correspond to port excitations
            B = self.B[domain]  # Assumes B is assembled during _ensure_matrices_assembled

            # Build excitation ordering: list of (port_idx, port_name, mode_idx)
            excitation_keys = []
            for pm, port_m in enumerate(domain_ports):
                if port_m not in self.port_modes:
                    continue
                for mode_m in sorted(self.port_modes[port_m].keys()):
                    excitation_keys.append((pm, port_m, mode_m))

            n_excitations = len(excitation_keys)
            n_freqs = len(self.frequencies)

            # Pre-allocate Z matrix for this domain: (n_freqs, n_excitations, n_excitations)
            Z_matrix = np.zeros((n_freqs, n_excitations, n_excitations), dtype=complex)
            snapshots_list = []

            total_iter_steps = 0
            freq_iters = []
            freq_residuals = []

            for kk, freq in enumerate(self.frequencies):
                omega = 2 * np.pi * freq

                u, v = fes.TnT()

                # System matrix: A(ω) = K - ω²M
                a_form = BilinearForm(
                    (1 / mu0) * curl(u) * curl(v) * dx(domain)
                    - omega ** 2 * eps0 * u * v * dx(domain)
                )

                with TaskManager():
                    if st == 'direct':
                        a_form.Assemble()
                        inv_a = a_form.mat.Inverse(freedofs=fes.FreeDofs())
                    else:
                        precond = NGPreconditioner(a_form, iter_opts['precond'])
                        a_form.Assemble()

                # Solve for ALL port excitations at this frequency
                # Store solutions as columns: x_all has shape (ndof, n_excitations)
                x_all = np.zeros((fes.ndof, n_excitations), dtype=complex)

                for col, (pm, port_m, mode_m) in enumerate(excitation_keys):
                    # RHS: excitation at port_m
                    f = LinearForm(fes)
                    f += omega * InnerProduct(
                        self.port_modes[port_m][mode_m], v.Trace()
                    ) * ds(port_m)

                    with TaskManager():
                        f.Assemble()

                    E = GridFunction(fes)

                    if st == 'direct':
                        E.vec.data = inv_a * f.vec
                        freq_iters.append(0)
                        freq_residuals.append(0.0)
                    else:
                        sol, iters, res = self._solve_system(
                            fes, a_form, f.vec, precond, iter_opts
                        )
                        E.vec.data = sol
                        total_iter_steps += iters
                        freq_iters.append(iters)
                        freq_residuals.append(res)

                    # Store solution vector
                    x_all[:, col] = E.vec.FV().NumPy()

                # ============================================================
                # Fast Z extraction: Z_ij = 1j * B^H @ x
                # B^H has shape (n_excitations, ndof), x_all has shape (ndof, n_excitations)
                # Result: (n_excitations, n_excitations)
                # ============================================================
                Z_matrix[kk, :, :] = 1j * (B.T.conj() @ x_all)

                # Store snapshots if requested
                if store_snapshots:
                    for col in range(n_excitations):
                        snapshots_list.append(x_all[:, col].copy())

                # Progress reporting
                if (kk + 1) % max(1, n_freqs // 5) == 0 or kk == n_freqs - 1:
                    elapsed = time.time() - t_domain_start
                    print(f"    [{kk + 1}/{n_freqs}] {elapsed:.1f}s elapsed")

            # Convert Z_matrix to dict format for compatibility
            self._Z_per_domain[domain] = {}
            for col, (pm, port_m, mode_m) in enumerate(excitation_keys):
                for row, (pn, port_n, mode_n) in enumerate(excitation_keys):
                    key = f"{pn + 1}({mode_n + 1}){pm + 1}({mode_m + 1})"
                    self._Z_per_domain[domain][key] = Z_matrix[:, row, col]

            if store_snapshots and snapshots_list:
                self.snapshots[domain] = np.array(snapshots_list).T

            t_elapsed = time.time() - t_domain_start
            msg = f"  Completed: {len(domain_ports)} ports, {n_freqs} frequencies in {t_elapsed:.2f}s"
            if st == 'iterative':
                msg += f" (total iteration steps: {total_iter_steps})"
            print(msg)

            # Store convergence info
            self._store_residuals(domain, n_freqs, freq_iters, freq_residuals, st)

    def _solve_global_coupled(
        self,
        store_snapshots: bool,
        solver_type: str = 'auto',
        iter_opts: Optional[Dict] = None,
    ) -> None:
        """Solve entire structure as one coupled system with fast Z-extraction."""
        import time

        t_start = time.time()

        if iter_opts is None:
            iter_opts = self._merge_iterative_opts(None)

        if self._fes_global is None:
            self._assemble_global_matrices()

        fes = self._fes_global
        st = self._resolve_solver_type(solver_type, fes)

        if st == 'iterative':
            fes = self._prepare_iterative(fes, iter_opts)
            self._fes_global = fes

        # For global solve, use external ports only for compound structures
        target_ports = self._external_ports if self.is_compound else self._ports
        n_modes = self._n_modes_per_port or 1

        # Build excitation ordering
        excitation_keys = []
        for pm, port_m in enumerate(target_ports):
            if port_m not in self.port_modes:
                continue
            for mode_m in sorted(self.port_modes[port_m].keys()):
                excitation_keys.append((pm, port_m, mode_m))

        n_excitations = len(excitation_keys)
        n_freqs = len(self.frequencies)

        # Get global B matrix
        B = self.B_global  # Assumes assembled during _assemble_global_matrices

        self._Z_matrix = np.zeros((n_freqs, n_excitations, n_excitations), dtype=complex)
        snapshots_list = []
        total_iter_steps = 0
        freq_iters = []
        freq_residuals = []

        for kk, freq in enumerate(self.frequencies):
            if kk % max(1, n_freqs // 10) == 0:
                print(f"  Frequency {kk + 1}/{n_freqs}: {freq / 1e9:.4f} GHz")

            omega = 2 * np.pi * freq

            u, v = fes.TnT()
            a_form = BilinearForm(
                (1 / mu0) * curl(u) * curl(v) * dx
                - omega ** 2 * eps0 * u * v * dx
            )

            with TaskManager():
                if st == 'direct':
                    a_form.Assemble()
                    inv_a = a_form.mat.Inverse(freedofs=fes.FreeDofs())
                else:
                    precond = NGPreconditioner(a_form, iter_opts['precond'])
                    a_form.Assemble()

            # Solve for all port excitations
            x_all = np.zeros((fes.ndof, n_excitations), dtype=complex)

            for col, (pm, port_m, mode_m) in enumerate(excitation_keys):
                f = LinearForm(fes)
                f += omega * InnerProduct(
                    self.port_modes[port_m][mode_m], v.Trace()
                ) * ds(port_m)

                with TaskManager():
                    f.Assemble()

                E = GridFunction(fes)

                if st == 'direct':
                    E.vec.data = inv_a * f.vec
                    freq_iters.append(0)
                    freq_residuals.append(0.0)
                else:
                    sol, iters, res = self._solve_system(
                        fes, a_form, f.vec, precond, iter_opts
                    )
                    E.vec.data = sol
                    total_iter_steps += iters
                    freq_iters.append(iters)
                    freq_residuals.append(res)

                x_all[:, col] = E.vec.FV().NumPy()

            # Fast Z extraction
            self._Z_matrix[kk, :, :] = 1j * (B.T.conj() @ x_all)

            if store_snapshots:
                for col in range(n_excitations):
                    snapshots_list.append(x_all[:, col].copy())

        if store_snapshots and snapshots_list:
            self.snapshots["global"] = np.array(snapshots_list).T

        self._Z_global_coupled = self._Z_matrix.copy()

        t_elapsed = time.time() - t_start
        msg = f"\nCoupled solve complete: {len(target_ports)} external ports in {t_elapsed:.2f}s"
        if st == 'iterative':
            msg += f" (total iteration steps: {total_iter_steps})"
        print(msg)

        self._store_residuals('global', n_freqs, freq_iters, freq_residuals, st)

    def _store_residuals(
        self,
        key: str,
        n_freqs: int,
        freq_iters: List[int],
        freq_residuals: List[float],
        solver_type: str,
    ) -> None:
        """Helper to store convergence info."""
        if not hasattr(self, '_residuals'):
            self._residuals = {}

        if freq_iters:
            n_excitations = len(freq_iters) // n_freqs
            raw_iters = np.array(freq_iters).reshape(n_freqs, n_excitations)
            raw_res = np.array(freq_residuals).reshape(n_freqs, n_excitations)
        else:
            raw_iters = np.zeros((n_freqs, 1))
            raw_res = np.zeros((n_freqs, 1))

        self._residuals[key] = {
            'frequencies': self.frequencies.copy(),
            'iterations': raw_iters.mean(axis=1),
            'residuals': raw_res.min(axis=1),
            'iterations_per_excitation': raw_iters,
            'residuals_per_excitation': raw_res,
            'solver_type': solver_type,
        }

    # def _solve_global_coupled(
    #     self, store_snapshots: bool,
    #     solver_type: str = 'auto',
    #     iter_opts: Optional[Dict] = None,
    # ) -> None:
    #     """Solve entire structure as one coupled system (with consistent port sign convention)."""
    #     # Optional: Conj is available in most NGSolve builds; keep a safe fallback.
    #     try:
    #         from ngsolve import Conj  # type: ignore
    #     except Exception:
    #         Conj = None  # noqa: N816

    #     if iter_opts is None:
    #         iter_opts = self._merge_iterative_opts(None)

    #     if self._fes_global is None:
    #         self._assemble_global_matrices()

    #     fes = self._fes_global

    #     # Resolve solver type
    #     st = self._resolve_solver_type(solver_type, fes)

    #     # Prepare multigrid if iterative
    #     if st == 'iterative':
    #         fes = self._prepare_iterative(fes, iter_opts)
    #         self._fes_global = fes   # update stored ref

    #     # For global solve, use external ports only for compound structures
    #     target_ports = self._external_ports if self.is_compound else self._ports
    #     n_ports = len(target_ports)
    #     n_freqs = len(self.frequencies)
    #     n_modes = self._n_modes_per_port or 1

    #     self._Z_matrix = np.zeros((n_freqs, n_ports * n_modes, n_ports * n_modes), dtype=complex)
    #     snapshots_list = []
    #     total_iter_steps = 0
    #     freq_iters = []
    #     freq_residuals = []

    #     for kk, freq in enumerate(self.frequencies):
    #         if kk % max(1, n_freqs // 10) == 0:
    #             print(f"  Frequency {kk + 1}/{n_freqs}: {freq / 1e9:.4f} GHz")

    #         omega = 2 * np.pi * freq

    #         u, v = fes.TnT()
    #         a_form = BilinearForm(
    #             (1 / mu0) * curl(u) * curl(v) * dx
    #             - omega ** 2 * eps0 * u * v * dx
    #         )

    #         # Direct: assemble + invert once per frequency
    #         with TaskManager():
    #             if st == 'direct':
    #                 a_form.Assemble()
    #                 inv_a = a_form.mat.Inverse(freedofs=fes.FreeDofs())
    #             else:
    #                 precond = NGPreconditioner(a_form, iter_opts['precond'])
    #                 a_form.Assemble()

    #             # Solve for each external port excitation
    #             for pm, port_m in enumerate(target_ports):
    #                 if port_m not in self.port_modes:
    #                     continue

    #                 for mode_m in sorted(self.port_modes[port_m].keys()):
    #                     f = LinearForm(fes)

    #                     f += omega * InnerProduct(
    #                         self.port_modes[port_m][mode_m], v.Trace()
    #                     ) * ds(port_m)

    #                     with TaskManager():
    #                         f.Assemble()

    #                     E = GridFunction(fes)

    #                     if st == 'direct':
    #                         E.vec.data = inv_a * f.vec
    #                         # Explicitly record zero iters/residuals for direct solver
    #                         freq_iters.append(0)
    #                         freq_residuals.append(0.0)
    #                     else:
    #                         sol, iters, res = self._solve_system(
    #                             fes, a_form, f.vec, precond, iter_opts
    #                         )
    #                         E.vec.data = sol
    #                         total_iter_steps += iters
    #                         freq_iters.append(iters)
    #                         freq_residuals.append(res)

    #                     if store_snapshots:
    #                         snapshots_list.append(E.vec.FV().NumPy().copy())

    #                     # Extract Z-parameters at external ports
    #                     for pn, port_n in enumerate(target_ports):
    #                         if port_n not in self.port_modes:
    #                             continue

    #                         port_region_n = self.mesh.Boundaries(port_n)

    #                         for mode_n in sorted(self.port_modes[port_n].keys()):
    #                             row = pn * n_modes + mode_n
    #                             col = pm * n_modes + mode_m

    #                             test_mode = self.port_modes[port_n][mode_n]
    #                             if Conj is not None:
    #                                 test_mode = Conj(test_mode)

    #                             z_val = Integrate(
    #                                 InnerProduct(E, test_mode),
    #                                 self.mesh, BND,
    #                                 definedon=port_region_n
    #                             )

    #                             self._Z_matrix[kk, row, col] = 1j * z_val

    #     if store_snapshots and snapshots_list:
    #         self.snapshots["global"] = np.array(snapshots_list).T

    #     # Store coupled result separately
    #     self._Z_global_coupled = self._Z_matrix.copy()

    #     msg = f"\nCoupled solve complete: {n_ports} external ports"
    #     if st == 'iterative':
    #         msg += f" (total CG steps: {total_iter_steps})"
    #     print(msg)

    #     # Store residual info for the global solve
    #     # freq_iters/freq_residuals have one entry per (freq, port, mode)
    #     # Reshape to (n_freqs, n_excitations), then:
    #     # - iterations: mean across excitations
    #     # - residuals: min across excitations (best/final per-frequency residual)
    #     if not hasattr(self, '_residuals'):
    #         self._residuals = {}
    #     if freq_iters:
    #         raw_iters = np.array(freq_iters).reshape(n_freqs, -1)
    #         raw_res = np.array(freq_residuals).reshape(n_freqs, -1)
    #     else:
    #         raw_iters = np.zeros((n_freqs, 1))
    #         raw_res = np.zeros((n_freqs, 1))
    #     self._residuals['global'] = {
    #         'frequencies': self.frequencies.copy(),
    #         'iterations': raw_iters.mean(axis=1),
    #         'residuals': raw_res.min(axis=1),
    #         'iterations_per_excitation': raw_iters,
    #         'residuals_per_excitation': raw_res,
    #         'solver_type': st,
    #     }

    def _compute_per_domain_s_from_z(self) -> None:
        """Compute per-domain S-matrices from per-domain Z data."""
        n_freqs = len(self.frequencies)
        n_modes = self._n_modes_per_port or 1

        for domain in self.domains:
            if domain not in self._Z_per_domain:
                continue

            domain_ports = self.domain_port_map[domain]
            n_ports_d = len(domain_ports)

            # Build Z-matrix for this domain
            Z_d = np.zeros((n_freqs, n_ports_d * n_modes, n_ports_d * n_modes), dtype=complex)

            for i in range(n_ports_d):
                for j in range(n_ports_d):
                    for mi in range(n_modes):
                        for mj in range(n_modes):
                            key = f'{i + 1}({mi + 1}){j + 1}({mj + 1})'
                            if key in self._Z_per_domain[domain]:
                                row = i * n_modes + mi
                                col = j * n_modes + mj
                                Z_d[:, row, col] = self._Z_per_domain[domain][key]

            # Convert to S using each port's impedance
            self._S_per_domain[domain] = {}

            for k in range(n_freqs):
                freq = self.frequencies[k]
                Z0_d = np.diag([
                    self._get_port_impedance(p, m, freq)
                    for p in domain_ports
                    for m in range(n_modes)
                ])
                S_d = ParameterConverter.z_to_s(Z_d[k], Z0_d)

                # Store in dictionary format
                for i in range(n_ports_d):
                    for j in range(n_ports_d):
                        for mi in range(n_modes):
                            for mj in range(n_modes):
                                key = f'{i + 1}({mi + 1}){j + 1}({mj + 1})'
                                if key not in self._S_per_domain[domain]:
                                    self._S_per_domain[domain][key] = np.zeros(n_freqs, dtype=complex)
                                row = i * n_modes + mi
                                col = j * n_modes + mj
                                self._S_per_domain[domain][key][k] = S_d[row, col]

    def _get_per_domain_s_matrices(self) -> Dict[str, np.ndarray]:
        """Get per-domain S-matrices as arrays.

        Returns
        -------
        dict
            {domain: S_array} where S_array is (n_freqs, n_ports_d, n_ports_d)
        """
        n_freqs = len(self.frequencies)
        n_modes = self._n_modes_per_port or 1
        domain_S = {}

        for domain in self.domains:
            if domain not in self._S_per_domain:
                continue

            domain_ports = self.domain_port_map[domain]
            n_ports_d = len(domain_ports)

            S_d = np.zeros((n_freqs, n_ports_d * n_modes, n_ports_d * n_modes), dtype=complex)

            for i in range(n_ports_d):
                for j in range(n_ports_d):
                    for mi in range(n_modes):
                        for mj in range(n_modes):
                            key = f'{i + 1}({mi + 1}){j + 1}({mj + 1})'
                            if key in self._S_per_domain[domain]:
                                row = i * n_modes + mi
                                col = j * n_modes + mj
                                S_d[:, row, col] = self._S_per_domain[domain][key]

            domain_S[domain] = S_d

        return domain_S

    def _build_results_dict(
        self,
        compute_s_params: bool,
        per_domain: bool,
        global_method: Optional[str]
    ) -> Dict:
        """Build comprehensive results dictionary."""
        results = {
            'frequencies': self.frequencies,
            'method': global_method,
            'is_compound': self.is_compound,
            'n_domains': self.n_domains,
            'domains': self.domains,
            'all_ports': self._ports,
            'external_ports': self._external_ports,
            'internal_ports': self._internal_ports,
        }

        # Global results
        if global_method is not None:
            results['Z'] = self._Z_matrix
            results['S'] = self._S_matrix if compute_s_params else None
            results['Z_dict'] = self.Z_dict
            results['S_dict'] = self.S_dict if compute_s_params else None
            results['ports'] = self.ports  # External ports for compounds

        # Per-domain results
        if per_domain:
            results['Z_per_domain'] = self._Z_per_domain.copy()
            results['S_per_domain'] = self._S_per_domain.copy() if compute_s_params else None
            results['domain_port_map'] = self.domain_port_map.copy()

        # Snapshots
        results['snapshots'] = self.snapshots.copy()

        # Residuals
        results['residuals'] = self._residuals.copy()

        return results

    # === Method comparison ===

    def compare_methods(
            self,
            fmin: float,
            fmax: float,
            nsamples: int = 100,
            store_snapshots: bool = False,
            methods: List[str] = None
    ) -> Dict:
        """
        Solve using multiple methods for comparison.

        Parameters
        ----------
        fmin, fmax : float
            Frequency range in GHz
        nsamples : int
            Number of frequency samples
        store_snapshots : bool
            Whether to store solution snapshots
        methods : list of str, optional
            Methods to compare. Default: ['coupled', 'cascade', 'concatenate']

        Returns
        -------
        dict
            Comparison results including all solutions and error metrics
        """
        if not self.is_compound:
            raise ValueError(
                "Method comparison only meaningful for compound structures."
            )

        if methods is None:
            methods = ['coupled']

        print("\n" + "=" * 60)
        print(f"Comparing Methods: {methods}")
        print("=" * 60)

        results = {'frequencies': None}

        # Solve with coupled method first (reference)
        if 'coupled' in methods:
            print("\n[1] Solving with COUPLED method (reference)...")
            self.solve(
                fmin, fmax, nsamples,
                store_snapshots=store_snapshots,
                per_domain=True,
                global_method='coupled'
            )
            results['Z_coupled'] = self._Z_matrix.copy()
            results['S_coupled'] = self._S_matrix.copy()
            results['frequencies'] = self.frequencies.copy()

        # Compute other methods from per-domain results
        # Restore coupled as primary if available
        if 'coupled' in methods:
            self._Z_matrix = results['Z_coupled']
            self._S_matrix = results['S_coupled']
            self._current_global_method = 'coupled'

        # Compute error metrics
        print("\n" + "=" * 60)
        print("Method Comparison Results")
        print("=" * 60)

        reference = 'coupled' if 'coupled' in methods else methods[0]
        Z_ref = results.get(f'Z_{reference}')
        S_ref = results.get(f'S_{reference}')

        for method in methods:
            if method == reference:
                continue

            Z_m = results.get(f'Z_{method}')
            S_m = results.get(f'S_{method}')

            if Z_m is not None and Z_ref is not None:
                Z_diff = np.abs(Z_ref - Z_m)
                S_diff = np.abs(S_ref - S_m) if S_m is not None else None

                results[f'Z_diff_{method}'] = Z_diff
                results[f'S_diff_{method}'] = S_diff

                print(f"\n{method.upper()} vs {reference.upper()}:")
                print(f"  Max |ΔZ|: {np.max(Z_diff):.4e}")
                print(f"  Mean |ΔZ|: {np.mean(Z_diff):.4e}")
                if S_diff is not None:
                    print(f"  Max |ΔS|: {np.max(S_diff):.4e}")
                    print(f"  Mean |ΔS|: {np.mean(S_diff):.4e}")

        # Recommendation
        if 'concatenate' in methods and 'coupled' in methods:
            concat_err = np.max(results.get('S_diff_concatenate', [0]))
            if concat_err < 1e-10:
                print("\n✓ Concatenate matches coupled (as expected for linear systems)")
            else:
                print(f"\n⚠ Concatenate differs from coupled by {concat_err:.2e}")
                print("  This may indicate numerical issues or implementation bugs.")

        if 'cascade' in methods:
            cascade_err = np.max(results.get('S_diff_cascade', [0]))
            if cascade_err < 0.01:
                print("\n✓ Cascade method is appropriate for this structure.")
            elif cascade_err < 0.1:
                print("\n⚠ Cascade shows moderate error - use concatenate/coupled for accuracy.")
            else:
                print("\n✗ Cascade NOT appropriate - significant inter-domain reflections.")

        print("=" * 60)

        return results

    def _print_comparison_summary(
            self,
            results: Dict,
            methods: List[str],
            reference_method: str
    ) -> None:
        """Print comparison summary table."""
        print("\n" + "=" * 60)
        print("Method Comparison Results")
        print("=" * 60)

        for method in methods:
            if method == reference_method:
                continue

            Z_diff = results.get(f'Z_diff_{method}')
            S_diff = results.get(f'S_diff_{method}')

            if Z_diff is not None:
                print(f"\n{method.upper()} vs {reference_method.upper()}:")
                print(f"  {'Metric':<30} {'Value':>15}")
                print(f"  {'-' * 45}")
                print(f"  {'Max |ΔZ|':<30} {np.max(Z_diff):>15.4e}")
                print(f"  {'Mean |ΔZ|':<30} {np.mean(Z_diff):>15.4e}")
                print(f"  {'RMS |ΔZ|':<30} {np.sqrt(np.mean(Z_diff ** 2)):>15.4e}")

                if S_diff is not None:
                    print(f"  {'Max |ΔS|':<30} {np.max(S_diff):>15.4e}")
                    print(f"  {'Mean |ΔS|':<30} {np.mean(S_diff):>15.4e}")
                    print(f"  {'RMS |ΔS|':<30} {np.sqrt(np.mean(S_diff ** 2)):>15.4e}")

        # Recommendations
        print("\n" + "-" * 60)
        print("Recommendations:")

        if 'S_diff_concatenate' in results:
            concat_err = np.max(results['S_diff_concatenate'])
            if concat_err < 1e-10:
                print("✓ CONCATENATE matches COUPLED (expected for linear systems)")
            elif concat_err < 1e-6:
                print(f"✓ CONCATENATE closely matches COUPLED (max |ΔS| = {concat_err:.2e})")
            else:
                print(f"⚠ CONCATENATE differs from COUPLED (max |ΔS| = {concat_err:.2e})")
                print("  This may indicate numerical issues.")

        if 'S_diff_cascade' in results:
            cascade_err = np.max(results['S_diff_cascade'])
            if cascade_err < 0.01:
                print("✓ CASCADE is appropriate (negligible inter-domain reflections)")
            elif cascade_err < 0.1:
                print(f"⚠ CASCADE shows moderate error (max |ΔS| = {cascade_err:.2f})")
                print("  Use CONCATENATE or COUPLED for better accuracy.")
            else:
                print(f"✗ CASCADE NOT recommended (max |ΔS| = {cascade_err:.2f})")
                print("  Significant inter-domain reflections detected.")

        print("=" * 60)

    def _plot_method_comparison(
            self,
            results: Dict,
            methods: List[str],
            reference_method: str,
            plot_params: List[str] = None,
            figsize: Tuple[float, float] = None,
            db_scale: bool = True,
            show_phase: bool = True,
            show_error: bool = True,
            save_path: str = None
    ) -> Dict:
        """
        Generate comparison plots for different methods.

        Returns
        -------
        dict
            Dictionary of figure handles
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        frequencies = results['frequencies'] / 1e9  # GHz
        n_freqs = len(frequencies)

        # Determine S-parameters to plot
        S_ref = results.get(f'S_{reference_method}')
        if S_ref is None:
            raise ValueError(f"No S-parameters for reference method '{reference_method}'")

        n_ports = S_ref.shape[1]

        if plot_params is None:
            # # Default: S11, S21 for 2-port; all Sii and Si1 for n-port
            # if n_ports == 2:
            #     plot_params = ['S11', 'S21', 'S12', 'S22']
            # else:
            plot_params = [f'S{i + 1}{i + 1}' for i in range(n_ports)]  # Diagonal
            plot_params += [f'S{i + 1}1' for i in range(1, n_ports)]  # First column

        # Parse S-parameter indices
        param_indices = []
        for param in plot_params:
            if param.upper().startswith('S') and len(param) >= 3:
                try:
                    i = int(param[1]) - 1
                    j = int(param[2]) - 1
                    if 0 <= i < n_ports and 0 <= j < n_ports:
                        param_indices.append((param.upper(), i, j))
                except ValueError:
                    print(f"Warning: Could not parse S-parameter '{param}'")

        n_params = len(param_indices)
        if n_params == 0:
            print("Warning: No valid S-parameters to plot")
            return {}

        # Color scheme for methods
        method_colors = {
            'coupled': '#1f77b4',  # Blue
            'cascade': '#ff7f0e',  # Orange
            'concatenate': '#2ca02c',  # Green
        }
        method_styles = {
            'coupled': '-',
            'cascade': '--',
            'concatenate': ':',
        }
        method_markers = {
            'coupled': None,
            'cascade': None,
            'concatenate': None,
        }

        figures = {}

        # ========== Figure 1: S-Parameter Comparison ==========
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        if show_phase:
            n_rows *= 2  # Double rows for phase

        if figsize is None:
            fig_width = 6 * n_cols
            fig_height = 3 * n_rows
        else:
            fig_width, fig_height = figsize

        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        fig1.suptitle('S-Parameter Comparison: Methods', fontsize=14, fontweight='bold')

        for idx, (param_name, i, j) in enumerate(param_indices):
            if show_phase:
                ax_mag = axes1[2 * (idx // n_cols), idx % n_cols]
                ax_phase = axes1[2 * (idx // n_cols) + 1, idx % n_cols]
            else:
                ax_mag = axes1[idx // n_cols, idx % n_cols]
                ax_phase = None

            for method in methods:
                S_m = results.get(f'S_{method}')
                if S_m is None:
                    continue

                s_data = S_m[:, i, j]
                color = method_colors.get(method, 'gray')
                style = method_styles.get(method, '-')
                marker = method_markers.get(method)
                lw = 2.5 if method == reference_method else 1.5
                alpha = 1.0 if method == reference_method else 0.8

                # Magnitude
                if db_scale:
                    mag = 20 * np.log10(np.abs(s_data) + 1e-12)
                    ylabel_mag = f'|{param_name}| (dB)'
                else:
                    mag = np.abs(s_data)
                    ylabel_mag = f'|{param_name}|'

                ax_mag.plot(
                    frequencies, mag,
                    linestyle=style, color=color, linewidth=lw, alpha=alpha,
                    marker=marker, markevery=max(1, n_freqs // 20),
                    label=method.capitalize()
                )

                # Phase
                if ax_phase is not None:
                    phase = np.angle(s_data, deg=True)
                    ax_phase.plot(
                        frequencies, phase,
                        linestyle=style, color=color, linewidth=lw, alpha=alpha,
                        marker=marker, markevery=max(1, n_freqs // 20),
                        label=method.capitalize()
                    )

            ax_mag.set_ylabel(ylabel_mag)
            ax_mag.set_title(param_name)
            ax_mag.grid(True, alpha=0.3)
            ax_mag.legend(loc='best', fontsize=8)

            if ax_phase is not None:
                ax_phase.set_xlabel('Frequency (GHz)')
                ax_phase.set_ylabel(f'∠{param_name} (°)')
                ax_phase.grid(True, alpha=0.3)
            else:
                ax_mag.set_xlabel('Frequency (GHz)')

        # Hide unused subplots
        total_plots = n_rows * n_cols
        used_plots = n_params * (2 if show_phase else 1)
        for idx in range(used_plots, total_plots):
            axes1.flat[idx].set_visible(False)

        fig1.tight_layout()
        figures['s_parameters'] = fig1

        # ========== Figure 2: Error Comparison ==========
        if show_error:
            other_methods = [m for m in methods if m != reference_method]
            if other_methods:
                n_other = len(other_methods)
                n_error_cols = min(2, n_params)
                n_error_rows = (n_params + n_error_cols - 1) // n_error_cols

                fig2, axes2 = plt.subplots(
                    n_error_rows, n_error_cols,
                    figsize=(6 * n_error_cols, 3 * n_error_rows),
                    squeeze=False
                )
                fig2.suptitle(
                    f'S-Parameter Error vs {reference_method.capitalize()}',
                    fontsize=14, fontweight='bold'
                )

                for idx, (param_name, i, j) in enumerate(param_indices):
                    ax = axes2[idx // n_error_cols, idx % n_error_cols]

                    for method in other_methods:
                        S_diff = results.get(f'S_diff_{method}')
                        if S_diff is None:
                            continue

                        err = S_diff[:, i, j]
                        color = method_colors.get(method, 'gray')
                        style = method_styles.get(method, '-')

                        ax.semilogy(
                            frequencies, err + 1e-16,
                            linestyle=style, color=color, linewidth=1.5,
                            label=f'{method.capitalize()}'
                        )

                    ax.set_xlabel('Frequency (GHz)')
                    ax.set_ylabel(f'|Δ{param_name}|')
                    ax.set_title(f'{param_name} Error')
                    ax.grid(True, alpha=0.3, which='both')
                    ax.legend(loc='best', fontsize=8)

                # Hide unused
                for idx in range(n_params, n_error_rows * n_error_cols):
                    axes2.flat[idx].set_visible(False)

                fig2.tight_layout()
                figures['error'] = fig2

        # ========== Figure 3: Summary Statistics ==========
        other_methods = [m for m in methods if m != reference_method]
        if other_methods and show_error:
            fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
            fig3.suptitle('Error Summary Statistics', fontsize=14, fontweight='bold')

            # Plot 3a: Max error per S-parameter
            ax3a = axes3[0]
            x_pos = np.arange(n_params)
            width = 0.8 / len(other_methods)

            for m_idx, method in enumerate(other_methods):
                S_diff = results.get(f'S_diff_{method}')
                if S_diff is None:
                    continue

                max_errs = [np.max(S_diff[:, i, j]) for _, i, j in param_indices]
                color = method_colors.get(method, 'gray')
                ax3a.bar(
                    x_pos + m_idx * width, max_errs, width,
                    label=method.capitalize(), color=color, alpha=0.8
                )

            ax3a.set_xticks(x_pos + width * (len(other_methods) - 1) / 2)
            ax3a.set_xticklabels([p[0] for p in param_indices])
            ax3a.set_ylabel('Max |ΔS|')
            ax3a.set_title('Maximum Error per Parameter')
            ax3a.legend()
            ax3a.set_yscale('log')
            ax3a.grid(True, alpha=0.3, axis='y')

            # Plot 3b: RMS error per S-parameter
            ax3b = axes3[1]
            for m_idx, method in enumerate(other_methods):
                S_diff = results.get(f'S_diff_{method}')
                if S_diff is None:
                    continue

                rms_errs = [np.sqrt(np.mean(S_diff[:, i, j] ** 2)) for _, i, j in param_indices]
                color = method_colors.get(method, 'gray')
                ax3b.bar(
                    x_pos + m_idx * width, rms_errs, width,
                    label=method.capitalize(), color=color, alpha=0.8
                )

            ax3b.set_xticks(x_pos + width * (len(other_methods) - 1) / 2)
            ax3b.set_xticklabels([p[0] for p in param_indices])
            ax3b.set_ylabel('RMS |ΔS|')
            ax3b.set_title('RMS Error per Parameter')
            ax3b.legend()
            ax3b.set_yscale('log')
            ax3b.grid(True, alpha=0.3, axis='y')

            # Plot 3c: Error vs frequency (aggregated)
            ax3c = axes3[2]
            for method in other_methods:
                S_diff = results.get(f'S_diff_{method}')
                if S_diff is None:
                    continue

                # Frobenius norm at each frequency
                frob_err = np.sqrt(np.sum(np.abs(S_diff) ** 2, axis=(1, 2)))
                color = method_colors.get(method, 'gray')
                style = method_styles.get(method, '-')

                ax3c.semilogy(
                    frequencies, frob_err + 1e-16,
                    linestyle=style, color=color, linewidth=2,
                    label=method.capitalize()
                )

            ax3c.set_xlabel('Frequency (GHz)')
            ax3c.set_ylabel('||ΔS||_F')
            ax3c.set_title('Frobenius Norm Error vs Frequency')
            ax3c.legend()
            ax3c.grid(True, alpha=0.3, which='both')

            fig3.tight_layout()
            figures['summary'] = fig3

        # ========== Figure 4: Smith Chart (optional for 2-port) ==========
        if n_ports == 2:
            fig4 = self._plot_smith_chart_comparison(results, methods, method_colors)
            if fig4 is not None:
                figures['smith_chart'] = fig4

        # ========== Figure 5: Polar Plot of S21 ==========
        if n_ports >= 2:
            fig5 = self._plot_polar_comparison(results, methods, method_colors, param='S21')
            if fig5 is not None:
                figures['polar'] = fig5

        # Save if requested
        if save_path is not None:
            for name, fig in figures.items():
                path = f"{save_path}_{name}.png"
                fig.savefig(path, dpi=150, bbox_inches='tight')
                print(f"Saved: {path}")

        plt.show()

        return figures

    def _plot_smith_chart_comparison(
            self,
            results: Dict,
            methods: List[str],
            method_colors: Dict[str, str]
    ) -> Optional['plt.Figure']:
        """Plot S11 on Smith chart for all methods."""
        import matplotlib.pyplot as plt

        try:
            # Check if we have S11
            S_first = results.get(f'S_{methods[0]}')
            if S_first is None or S_first.shape[1] < 1:
                return None
        except Exception:
            return None

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        # Draw Smith chart background
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(theta, np.ones_like(theta), 'k-', linewidth=0.5, alpha=0.3)

        # Constant resistance circles (simplified)
        for r in [0.2, 0.5, 1.0, 2.0]:
            # Circle center and radius in Γ plane
            center = r / (1 + r)
            radius = 1 / (1 + r)
            # Parametric
            phi = np.linspace(0, 2 * np.pi, 100)
            x = center + radius * np.cos(phi)
            y = radius * np.sin(phi)
            # Convert to polar
            r_polar = np.sqrt(x ** 2 + y ** 2)
            theta_polar = np.arctan2(y, x)
            mask = r_polar <= 1.0
            ax.plot(theta_polar[mask], r_polar[mask], 'gray', linewidth=0.3, alpha=0.5)

        # Plot S11 for each method
        for method in methods:
            S_m = results.get(f'S_{method}')
            if S_m is None:
                continue

            s11 = S_m[:, 0, 0]
            r = np.abs(s11)
            theta = np.angle(s11)
            color = method_colors.get(method, 'gray')
            lw = 2.0 if method == results.get('reference_method') else 1.2

            ax.plot(theta, r, color=color, linewidth=lw, label=method.capitalize())

            # Mark start and end points
            ax.plot(theta[0], r[0], 'o', color=color, markersize=6)
            ax.plot(theta[-1], r[-1], 's', color=color, markersize=6)

        ax.set_title('S11 on Smith Chart', fontsize=12, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        fig.tight_layout()
        return fig

    def _plot_polar_comparison(
            self,
            results: Dict,
            methods: List[str],
            method_colors: Dict[str, str],
            param: str = 'S21'
    ) -> Optional['plt.Figure']:
        """Plot S-parameter in polar form."""
        import matplotlib.pyplot as plt

        # Parse parameter
        try:
            i = int(param[1]) - 1
            j = int(param[2]) - 1
        except (ValueError, IndexError):
            return None

        S_first = results.get(f'S_{methods[0]}')
        if S_first is None or S_first.shape[1] <= max(i, j):
            return None

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

        for method in methods:
            S_m = results.get(f'S_{method}')
            if S_m is None:
                continue

            s_data = S_m[:, i, j]
            r = np.abs(s_data)
            theta = np.angle(s_data)
            color = method_colors.get(method, 'gray')
            lw = 2.0 if method == results.get('reference_method') else 1.2

            ax.plot(theta, r, color=color, linewidth=lw, label=method.capitalize())

            # Mark start (circle) and end (square)
            ax.plot(theta[0], r[0], 'o', color=color, markersize=8)
            ax.plot(theta[-1], r[-1], 's', color=color, markersize=8)

        ax.set_title(f'{param} Polar Plot', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        fig.tight_layout()
        return fig

    def plot_s_parameters_comparison(
            self,
            results: Dict = None,
            params: List[str] = None,
            db_scale: bool = True,
            show_phase: bool = True,
            figsize: Tuple[float, float] = None,
            title: str = None
    ) -> Tuple['plt.Figure', np.ndarray]:
        """
        Plot S-parameters from comparison results or current solution.

        Parameters
        ----------
        results : dict, optional
            Results from compare_methods(). If None, uses current solution.
        params : list of str, optional
            S-parameters to plot. Default: ['S11', 'S21'] for 2-port.
        db_scale : bool
            Plot magnitude in dB
        show_phase : bool
            Include phase subplot
        figsize : tuple, optional
            Figure size
        title : str, optional
            Figure title

        Returns
        -------
        fig, axes
        """
        import matplotlib.pyplot as plt

        if results is None:
            # Use current solution
            if self._S_matrix is None:
                raise ValueError("No S-parameters available. Call solve() first.")

            results = {
                'frequencies': self.frequencies,
                f'S_{self._current_global_method or "solution"}': self._S_matrix,
                'methods': [self._current_global_method or 'solution']
            }

        methods = results.get('methods', ['solution'])
        frequencies = results['frequencies'] / 1e9

        # Get first available S-matrix to determine size
        S_first = None
        for m in methods:
            S_first = results.get(f'S_{m}')
            if S_first is not None:
                break

        if S_first is None:
            raise ValueError("No S-parameters in results")

        n_ports = S_first.shape[1]

        # Default parameters
        if params is None:
            if n_ports == 2:
                params = ['S11', 'S21']
            else:
                params = [f'S{i + 1}{j + 1}' for i in range(min(2, n_ports)) for j in range(min(2, n_ports))]

        # Parse parameters
        param_list = []
        for p in params:
            try:
                i = int(p[1]) - 1
                j = int(p[2]) - 1
                if 0 <= i < n_ports and 0 <= j < n_ports:
                    param_list.append((p.upper(), i, j))
            except (ValueError, IndexError):
                pass

        n_params = len(param_list)
        n_cols = min(2, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        if show_phase:
            n_rows *= 2

        if figsize is None:
            figsize = (6 * n_cols, 3 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        colors = plt.cm.tab10.colors

        for idx, (pname, i, j) in enumerate(param_list):
            if show_phase:
                ax_mag = axes[2 * (idx // n_cols), idx % n_cols]
                ax_ph = axes[2 * (idx // n_cols) + 1, idx % n_cols]
            else:
                ax_mag = axes[idx // n_cols, idx % n_cols]
                ax_ph = None

            for m_idx, method in enumerate(methods):
                S_m = results.get(f'S_{method}')
                if S_m is None:
                    continue

                s_data = S_m[:, i, j]
                color = colors[m_idx % len(colors)]
                label = method.capitalize() if len(methods) > 1 else None

                # Magnitude
                if db_scale:
                    mag = 20 * np.log10(np.abs(s_data) + 1e-12)
                    ylabel = f'|{pname}| (dB)'
                else:
                    mag = np.abs(s_data)
                    ylabel = f'|{pname}|'

                ax_mag.plot(frequencies, mag, color=color, linewidth=1.5, label=label)

                # Phase
                if ax_ph is not None:
                    phase = np.angle(s_data, deg=True)
                    ax_ph.plot(frequencies, phase, color=color, linewidth=1.5)

            ax_mag.set_ylabel(ylabel)
            ax_mag.set_title(pname)
            ax_mag.grid(True, alpha=0.3)
            if len(methods) > 1:
                ax_mag.legend(fontsize=8)

            if ax_ph is not None:
                ax_ph.set_xlabel('Frequency (GHz)')
                ax_ph.set_ylabel(f'∠{pname} (°)')
                ax_ph.grid(True, alpha=0.3)
            else:
                ax_mag.set_xlabel('Frequency (GHz)')

        # Hide unused
        total_axes = n_rows * n_cols
        used = n_params * (2 if show_phase else 1)
        for idx in range(used, total_axes):
            axes.flat[idx].set_visible(False)

        fig.tight_layout()
        return fig, axes

    def get_cascaded_results(self) -> Optional[Dict]:
        """Get cached cascade results if available."""
        if self._Z_global_cascade is None:
            return None

        return {
            'Z': self._Z_global_cascade,
            'S': self._S_global_cascade,
            'frequencies': self.frequencies,
            'method': 'cascade'
        }

    def get_coupled_results(self) -> Optional[Dict]:
        """Get cached coupled results if available."""
        if self._Z_global_coupled is None:
            return None

        return {
            'Z': self._Z_global_coupled,
            'S': self._S_global_coupled,
            'frequencies': self.frequencies,
            'method': 'coupled'
        }

    # ====Eigenmode ====
    # Add this constant at class level
    DEFAULT_MIN_EIGENVALUE = 1.0  # ω² > 1 means ω > 1 rad/s

    @staticmethod
    def _filter_eigenvalues(
        eigenvalues: np.ndarray,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> np.ndarray:
        """
        Filter and sort eigenvalues.

        Parameters
        ----------
        eigenvalues : ndarray
            Raw eigenvalues
        filter_static : bool
            If True, remove static modes (eigenvalues <= min_eigenvalue)
        min_eigenvalue : float, optional
            Threshold for static mode filtering
        n_modes : int, optional
            Return only first n_modes eigenvalues

        Returns
        -------
        filtered_eigenvalues : ndarray
            Sorted, filtered eigenvalues
        """
        if min_eigenvalue is None:
            min_eigenvalue = FrequencyDomainSolver.DEFAULT_MIN_EIGENVALUE

        # Sort eigenvalues
        eigs_sorted = np.sort(np.real(eigenvalues))

        # Filter static modes
        if filter_static:
            eigs_sorted = eigs_sorted[eigs_sorted > min_eigenvalue]

        # Limit to n_modes
        if n_modes is not None and len(eigs_sorted) > n_modes:
            eigs_sorted = eigs_sorted[:n_modes]

        return eigs_sorted

    def get_eigenvalues(
        self,
        domain: str = None,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None,
        sigma: float = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute eigenvalues from generalized eigenvalue problem K x = λ M x.

        Parameters
        ----------
        domain : str, optional
            Specific domain or 'global'. If None, returns all domains + global.
        filter_static : bool
            If True (default), remove static modes (eigenvalues <= min_eigenvalue)
        min_eigenvalue : float, optional
            Threshold for static mode filtering. Default: 1.0
        n_modes : int, optional
            Number of eigenvalues to return (after filtering)
        sigma : float, optional
            Shift for shift-invert mode. If None, uses a default based on
            expected frequency range. Set to 0 for smallest eigenvalues.

        Returns
        -------
        eigenvalues : ndarray or dict
            Eigenvalues (ω² values), sorted and filtered
        """
        # Check if matrices are available
        has_per_domain = bool(self.M)
        has_global = self.M_global is not None

        if not has_per_domain and not has_global:
            raise ValueError("Matrices not assembled. Call assemble_matrices() first.")

        def compute_eigs(M_mat, K_mat, fes, label: str) -> np.ndarray:
            """Compute eigenvalues for a single domain/global system."""
            # Get free DOFs (not constrained by Dirichlet BC)
            freedofs = fes.FreeDofs()
            free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]])

            if len(free_idx) == 0:
                print(f"Warning: No free DOFs for {label}")
                return np.array([])

            # Extract submatrices for free DOFs
            if sp.issparse(M_mat):
                M_free = M_mat[free_idx, :][:, free_idx]
                K_free = K_mat[free_idx, :][:, free_idx]
            else:
                M_free = M_mat[np.ix_(free_idx, free_idx)]
                K_free = K_mat[np.ix_(free_idx, free_idx)]

            n_free = len(free_idx)

            # Determine number of eigenvalues to compute
            k = min(n_modes or 50, n_free - 2) if n_modes else min(50, n_free - 2)
            k = max(k, 1)

            # Determine shift for shift-invert
            shift = sigma
            if shift is None:
                # Default shift: target middle of typical frequency range
                # For GHz range: ω ~ 2π * 1e9, so ω² ~ 4e19
                # But we'll use a more moderate default
                shift = 1e18  # Target around 1 GHz

            try:
                # Try sparse eigenvalue solver with shift-invert
                from scipy.sparse.linalg import eigsh

                # Ensure matrices are in CSR format for efficiency
                if sp.issparse(M_free):
                    M_csr = sp.csr_matrix(M_free)
                    K_csr = sp.csr_matrix(K_free)
                else:
                    M_csr = sp.csr_matrix(M_free)
                    K_csr = sp.csr_matrix(K_free)

                # Shift-invert mode: solve (K - σM)^{-1} M x = θ x
                # where λ = σ + 1/θ
                eigenvalues, _ = eigsh(
                    K_csr, k=k, M=M_csr,
                    sigma=shift, which='LM',
                    return_eigenvectors=True
                )
                return eigenvalues

            except Exception as e1:
                print(f"Note: Sparse eigsh failed for {label}: {e1}")
                print(f"      Falling back to dense solver...")

                try:
                    # Fall back to dense solver
                    M_dense = M_free.toarray() if sp.issparse(M_free) else np.array(M_free)
                    K_dense = K_free.toarray() if sp.issparse(K_free) else np.array(K_free)

                    # Use scipy.linalg.eigh for generalized symmetric eigenvalue problem
                    eigenvalues = sl.eigh(K_dense, M_dense, eigvals_only=True)
                    return eigenvalues

                except Exception as e2:
                    print(f"Warning: Dense eigh also failed for {label}: {e2}")
                    print(f"         Trying regularized approach...")

                    try:
                        # Regularize M if needed
                        M_dense = M_free.toarray() if sp.issparse(M_free) else np.array(M_free)
                        K_dense = K_free.toarray() if sp.issparse(K_free) else np.array(K_free)

                        # Add small regularization to M
                        eps = 1e-10 * np.max(np.abs(M_dense.diagonal()))
                        M_reg = M_dense + eps * np.eye(M_dense.shape[0])

                        eigenvalues = sl.eigh(K_dense, M_reg, eigvals_only=True)
                        return eigenvalues

                    except Exception as e3:
                        print(f"Error: All eigenvalue methods failed for {label}: {e3}")
                        return np.array([])

        def process_eigs(raw_eigs: np.ndarray) -> np.ndarray:
            """Apply filtering to raw eigenvalues."""
            if len(raw_eigs) == 0:
                return raw_eigs
            return self._filter_eigenvalues(raw_eigs, filter_static, min_eigenvalue, n_modes)

        # Handle specific domain request
        if domain is not None:
            if domain == 'global':
                if not has_global:
                    raise ValueError("Global matrices not assembled. "
                                     "Call assemble_matrices(assemble_global=True)")
                raw_eigs = compute_eigs(self.M_global, self.K_global, self._fes_global, 'global')
                return process_eigs(raw_eigs)
            elif domain in self.M:
                raw_eigs = compute_eigs(self.M[domain], self.K[domain], self._fes[domain], domain)
                return process_eigs(raw_eigs)
            else:
                raise KeyError(f"Domain '{domain}' not found. Available: {list(self.M.keys())}")

        # Return all available
        results = {}

        # Per-domain eigenvalues
        for d in self.domains:
            if d in self.M and d in self._fes:
                raw_eigs = compute_eigs(self.M[d], self.K[d], self._fes[d], d)
                results[d] = process_eigs(raw_eigs)

        # Global eigenvalues
        if has_global and self._fes_global is not None:
            raw_eigs = compute_eigs(self.M_global, self.K_global, self._fes_global, 'global')
            results['global'] = process_eigs(raw_eigs)

        return results

    def get_resonant_frequencies(
            self,
            domain: str = None,
            n_modes: int = None,
            fmin: float = None,
            filter_static: bool = True
    ) -> np.ndarray:
        """
        Get resonant frequencies from eigenvalues.

        Parameters
        ----------
        domain : str, optional
            Specific domain or 'global'. Default: 'global' if available.
        n_modes : int, optional
            Number of modes to return
        fmin : float, optional
            Minimum frequency in GHz. Modes below this are filtered out.
            Default: ~0.16 MHz (corresponds to min_eigenvalue=1.0)
        filter_static : bool
            If True (default), remove static modes (f ≈ 0).
            When fmin is specified, this is automatically True.

        Returns
        -------
        frequencies : ndarray
            Resonant frequencies in Hz, sorted ascending
        """
        # Default to global if available
        if domain is None:
            if self.M_global is not None:
                domain = 'global'
            elif self.n_domains == 1:
                domain = self.domains[0]

        # Convert fmin (GHz) to min_eigenvalue (ω²)
        # ω² = (2π * f)² where f is in Hz
        if fmin is not None:
            fmin_hz = fmin * 1e9
            min_eigenvalue = (2 * np.pi * fmin_hz) ** 2
            filter_static = True  # Implied when fmin is set
        else:
            min_eigenvalue = self.DEFAULT_MIN_EIGENVALUE if filter_static else None

        # Convert fmin to sigma for shift-invert (target slightly below fmin)
        if fmin is not None:
            # Use fmin as the shift point for eigenvalue search
            sigma = (2 * np.pi * fmin * 1e9) ** 2
        else:
            sigma = None

        eigs = self.get_eigenvalues(
            domain=domain,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=None,  # Don't limit here, do it after freq conversion
            sigma=sigma
        )

        if isinstance(eigs, dict):
            # If dict, prefer global, else combine all
            if 'global' in eigs:
                all_eigs = eigs['global']
            else:
                all_eigs = np.concatenate(list(eigs.values()))
        else:
            all_eigs = eigs

        # Convert to frequencies
        eigs_pos = all_eigs[all_eigs > 0]
        freqs = np.sqrt(eigs_pos) / (2 * np.pi)
        freqs = np.sort(freqs)

        if n_modes is not None:
            freqs = freqs[:n_modes]

        return freqs

    # === Per-domain access methods ===

    def get_domain_z_matrix(
        self,
        domain: str,
        freq_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get Z-matrix for a specific domain.

        Parameters
        ----------
        domain : str
            Domain name
        freq_idx : int, optional
            Frequency index. If None, returns all frequencies.

        Returns
        -------
        Z : ndarray
            Z-parameter matrix. Shape (n_ports, n_ports) if freq_idx given,
            otherwise (n_freqs, n_ports, n_ports).
        """
        if domain not in self._Z_per_domain:
            raise KeyError(
                f"Domain '{domain}' not found. "
                f"Available: {list(self._Z_per_domain.keys())}"
            )

        domain_ports = self.domain_port_map[domain]
        n_ports = len(domain_ports)
        n_modes = self._n_modes_per_port or 1
        n_freqs = len(self.frequencies)

        Z_full = np.zeros((n_freqs, n_ports * n_modes, n_ports * n_modes), dtype=complex)

        for i in range(n_ports):
            for j in range(n_ports):
                for mi in range(n_modes):
                    for mj in range(n_modes):
                        key = f'{i + 1}({mi + 1}){j + 1}({mj + 1})'
                        if key in self._Z_per_domain[domain]:
                            row = i * n_modes + mi
                            col = j * n_modes + mj
                            Z_full[:, row, col] = self._Z_per_domain[domain][key]

        if freq_idx is not None:
            return Z_full[freq_idx]
        return Z_full

    def get_domain_s_matrix(
        self,
        domain: str,
        freq_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get S-matrix for a specific domain.

        Parameters
        ----------
        domain : str
            Domain name
        freq_idx : int, optional
            Frequency index. If None, returns all frequencies.

        Returns
        -------
        S : ndarray
            S-parameter matrix. Shape (n_ports, n_ports) if freq_idx given,
            otherwise (n_freqs, n_ports, n_ports).
        """
        if domain not in self._S_per_domain:
            raise KeyError(
                f"Domain '{domain}' S-parameters not computed. "
                f"Call solve() with per_domain=True and compute_s_params=True."
            )

        domain_ports = self.domain_port_map[domain]
        n_ports = len(domain_ports)
        n_modes = self._n_modes_per_port or 1
        n_freqs = len(self.frequencies)

        S_full = np.zeros((n_freqs, n_ports * n_modes, n_ports * n_modes), dtype=complex)

        for i in range(n_ports):
            for j in range(n_ports):
                for mi in range(n_modes):
                    for mj in range(n_modes):
                        key = f'{i + 1}({mi + 1}){j + 1}({mj + 1})'
                        if key in self._S_per_domain[domain]:
                            row = i * n_modes + mi
                            col = j * n_modes + mj
                            S_full[:, row, col] = self._S_per_domain[domain][key]

        if freq_idx is not None:
            return S_full[freq_idx]
        return S_full

    def get_domain_results(self, domain: str) -> Dict:
        """
        Get all results for a specific domain.

        Parameters
        ----------
        domain : str
            Domain name

        Returns
        -------
        dict
            Dictionary with Z, S, frequencies, ports for the domain
        """
        if domain not in self._Z_per_domain:
            raise KeyError(
                f"Domain '{domain}' not found. "
                f"Available: {list(self._Z_per_domain.keys())}"
            )

        domain_ports = self.domain_port_map[domain]

        return {
            'frequencies': self.frequencies,
            'Z': self.get_domain_z_matrix(domain),
            'S': self.get_domain_s_matrix(domain) if domain in self._S_per_domain else None,
            'Z_dict': self._Z_per_domain[domain].copy(),
            'S_dict': self._S_per_domain.get(domain, {}).copy(),
            'ports': domain_ports,
            'n_ports': len(domain_ports),
            'snapshots': self.snapshots.get(domain)
        }

    def get_all_domain_results(self) -> Dict[str, Dict]:
        """Get results for all domains."""
        return {d: self.get_domain_results(d) for d in self.domains if d in self._Z_per_domain}

    # Add to FrequencyDomainSolver class

    def _build_sequential_connections_with_modes(
            self,
            n_modes: int
    ) -> List[Tuple[Tuple[int, str, int], Tuple[int, str, int]]]:
        """
        Build connections for sequential domains including mode indices.

        Returns list of ((domain_idx_a, port_a, mode), (domain_idx_b, port_b, mode))
        """
        connections = []
        for i in range(self.n_domains - 1):
            ports_i = self.domain_port_map[self.domains[i]]
            ports_next = self.domain_port_map[self.domains[i + 1]]

            # Connect last port of domain i to first port of domain i+1
            # For each mode
            for m in range(n_modes):
                connections.append((
                    (i, ports_i[-1], m),  # Last port of current domain
                    (i + 1, ports_next[0], m)  # First port of next domain
                ))
        return connections

    # === ROM interface ===

    def get_rom_data(self, domain: Optional[str] = None) -> Dict:
        """
        Get data needed for reduced order modeling.

        Parameters
        ----------
        domain : str, optional
            Specific domain to get data for.
            'global' for global coupled system.
            If None, returns data for all domains and global.

        Returns
        -------
        dict
            Dictionary with M, K, B, W (snapshots), fes for requested domain(s)
        """
        if domain == 'global':
            return {
                'M': self.M_global,
                'K': self.K_global,
                'B': self.B_global,
                'W': self.snapshots.get('global'),
                'fes': self._fes_global,
                'ports': self._external_ports if self.is_compound else self._ports,
                'n_ports': len(self._external_ports) if self.is_compound else self._n_ports
            }

        if domain is not None:
            if domain not in self.domains:
                raise KeyError(f"Domain '{domain}' not found. Available: {self.domains}")
            return {
                'M': self.M.get(domain),
                'K': self.K.get(domain),
                'B': self.B.get(domain),
                'W': self.snapshots.get(domain),
                'fes': self._fes.get(domain),
                'ports': self.domain_port_map.get(domain, []),
                'n_ports': len(self.domain_port_map.get(domain, []))
            }

        # Return all
        results = {}

        for d in self.domains:
            results[d] = {
                'M': self.M.get(d),
                'K': self.K.get(d),
                'B': self.B.get(d),
                'W': self.snapshots.get(d),
                'fes': self._fes.get(d),
                'ports': self.domain_port_map.get(d, [])
            }

        if self.M_global is not None:
            results['global'] = {
                'M': self.M_global,
                'K': self.K_global,
                'B': self.B_global,
                'W': self.snapshots.get('global'),
                'fes': self._fes_global,
                'ports': self._external_ports if self.is_compound else self._ports
            }

        return results

    # === Info and printing ===

    def print_info(self) -> None:
        """Print solver information."""
        self._print_structure_info()

        print("\n--- Matrix Assembly Status ---")
        print(f"Per-domain matrices assembled: {self._per_domain_matrices_assembled}")
        print(f"Global matrices assembled: {self._global_matrices_assembled}")

        if self._per_domain_matrices_assembled:
            for domain in self.domains:
                if domain in self.M:
                    print(f"\n  {domain}:")
                    print(f"    M: {self.M[domain].shape}, nnz: {self.M[domain].nnz}")
                    print(f"    K: {self.K[domain].shape}, nnz: {self.K[domain].nnz}")
                    print(f"    B: {self.B[domain].shape}")

        if self._global_matrices_assembled:
            print(f"\n  global:")
            print(f"    M: {self.M_global.shape}, nnz: {self.M_global.nnz}")
            print(f"    K: {self.K_global.shape}, nnz: {self.K_global.nnz}")
            print(f"    B: {self.B_global.shape}")

        print("\n--- Solution Status ---")
        print(f"Solution available: {self.frequencies is not None}")
        if self.frequencies is not None:
            print(f"  Frequency range: {self.frequencies[0] / 1e9:.4f} - {self.frequencies[-1] / 1e9:.4f} GHz")
            print(f"  Number of samples: {len(self.frequencies)}")
            print(f"  Current global method: {self._current_global_method}")
            print(f"  Per-domain results: {list(self._Z_per_domain.keys())}")
            print(f"  Snapshots stored: {list(self.snapshots.keys())}")
            print(f"  Coupled results cached: {self._Z_global_coupled is not None}")
            print(f"  Cascade results cached: {self._Z_global_cascade is not None}")

    def print_port_info(self) -> None:
        """Print information about detected ports."""
        if self.port_modes is None:
            print("Port modes not computed. Call assemble_matrices() first.")
            return

        print("\n" + "=" * 60)
        print("Port Information")
        print("=" * 60)

        fc_dict = self.port_solver.get_cutoff_frequencies_dict()
        pol_info = self.port_solver.get_polarization_info()

        for port in self._ports:
            domains_with_port = [
                d for d, ports in self.domain_port_map.items()
                if port in ports
            ]

            if port in self._external_ports:
                if port == self._ports[0]:
                    port_type = "EXTERNAL (input)"
                else:
                    port_type = "EXTERNAL (output)"
            else:
                port_type = "INTERNAL"

            print(f"\nPort: {port} [{port_type}]")
            print(f"  Adjacent domains: {domains_with_port}")

            if port in pol_info:
                print(f"  Normal: {pol_info[port]['normal']}")
                print(f"  Orientation factor: {pol_info[port]['orientation_factor']}")

            if port in fc_dict:
                for mode, fc in fc_dict[port].items():
                    print(f"  Mode {mode}: fc = {fc / 1e9:.4f} GHz")

        print("=" * 60)

    def print_domain_info(self) -> None:
        """Print detailed information about each domain."""
        print("\n" + "=" * 60)
        print("Domain Information")
        print("=" * 60)

        for domain in self.domains:
            print(f"\nDomain: {domain}")
            ports = self.domain_port_map.get(domain, [])

            port_info = []
            for p in ports:
                if p in self._external_ports:
                    port_info.append(f"{p} (external)")
                else:
                    port_info.append(f"{p} (internal)")
            print(f"  Ports: {port_info}")

            if domain in self._fes:
                fes = self._fes[domain]
                print(f"  FES ndof: {fes.ndof}")

            if domain in self.M:
                print(f"  M shape: {self.M[domain].shape}, nnz: {self.M[domain].nnz}")
                print(f"  K shape: {self.K[domain].shape}, nnz: {self.K[domain].nnz}")

            if domain in self.B:
                print(f"  B shape: {self.B[domain].shape}")

            if domain in self.snapshots:
                print(f"  Snapshots shape: {self.snapshots[domain].shape}")

            if domain in self._Z_per_domain:
                n_params = len([k for k in self._Z_per_domain[domain].keys()])
                print(f"  Z-parameters computed: {n_params} entries")

        if self._fes_global is not None:
            print(f"\nGlobal (coupled):")
            print(f"  FES ndof: {self._fes_global.ndof}")
            print(f"  M_global shape: {self.M_global.shape}")
            print(f"  External ports: {self._external_ports}")

        print("=" * 60)

    # === Visualization methods ===

    def plot_port_mode(
        self,
        port: str,
        mode: int = 0,
        component: Literal['real', 'imag', 'abs', 'all'] = None,
        **kwargs
    ) -> None:
        """Visualize port eigenmode pattern."""
        if self.port_modes is None:
            raise ValueError("Port modes not computed. Call assemble_matrices() first.")

        if port not in self.port_modes:
            raise ValueError(f"Port '{port}' not found. Available: {list(self.port_modes.keys())}")

        if mode not in self.port_modes[port]:
            raise ValueError(f"Mode {mode} not found for port '{port}'")

        mode_cf = self.port_modes[port][mode]

        fc_dict = self.port_solver.get_cutoff_frequencies_dict()
        fc = fc_dict.get(port, {}).get(mode, 0)

        if port in self._external_ports:
            if port == self._ports[0]:
                port_type = "external (input)"
            else:
                port_type = "external (output)"
        else:
            port_type = "internal"

        print(f"\nPort Mode: {port} [{port_type}], Mode {mode}")
        print(f"Cutoff frequency: {fc / 1e9:.4f} GHz")

        if component == 'abs':
            cf_plot = Norm(mode_cf)
        elif component == 'real':
            cf_plot = mode_cf.real
        elif component == 'imag':
            cf_plot = mode_cf.imag
        elif component == 'all':
            print("Plotting Real part:")
            Draw(mode_cf.real, self.mesh, **kwargs)
            print("\nPlotting Imaginary part:")
            Draw(mode_cf.imag, self.mesh, **kwargs)
            print("\nPlotting Magnitude:")
            Draw(Norm(mode_cf), self.mesh, **kwargs)
            return
        else:
            cf_plot = mode_cf

        Draw(cf_plot, self.mesh, **kwargs)

    def plot_field(
        self,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        excitation_mode: int = 0,
        domain: Optional[str] = None,
        component: Literal['real', 'imag', 'abs'] = None,
        field_type: Literal['E', 'H'] = 'E',
        clipping: Optional[Dict] = None,
        **kwargs
    ) -> None:
        """
        Visualize computed field at a specific frequency.

        Parameters
        ----------
        freq_idx : int
            Frequency index
        excitation_port : str, optional
            Port used for excitation. If None, uses first available port.
        excitation_mode : int
            Mode index for excitation
        domain : str, optional
            Domain to visualize. Required for per-domain snapshots in
            compound structures. Use 'global' for coupled solve snapshots.
        component : {'real', 'imag', 'abs'}
            Field component to plot
        field_type : {'E', 'H'}
            Electric or magnetic field
        clipping : dict, optional
            Clipping plane specification
        **kwargs
            Additional arguments passed to Draw()
        """
        if self.frequencies is None:
            raise ValueError("No solution available. Call solve() first.")

        if freq_idx >= len(self.frequencies):
            raise ValueError(f"freq_idx {freq_idx} out of range [0, {len(self.frequencies) - 1}]")

        freq = self.frequencies[freq_idx]
        omega = 2 * np.pi * freq

        # Determine which snapshots to use
        snapshot_key, fes, available_ports = self._get_snapshot_context(domain)

        if excitation_port is None:
            excitation_port = available_ports[0]

        if excitation_port not in available_ports:
            raise ValueError(
                f"Port '{excitation_port}' not available for {snapshot_key}. "
                f"Available: {available_ports}"
            )

        print(f"\nField visualization at f = {freq / 1e9:.4f} GHz")
        print(f"Source: {snapshot_key}")
        print(f"Excitation: {excitation_port}, mode {excitation_mode}")

        # Reconstruct field from snapshots
        E_gf = self._reconstruct_field(
            freq_idx, excitation_port, excitation_mode,
            snapshot_key, fes, available_ports
        )

        # Select field type
        if field_type == 'E':
            field_cf = E_gf
            field_label = "E"
        elif field_type == 'H':
            field_cf = (1 / (1j * omega * mu0)) * curl(E_gf)
            field_label = "H"
        else:
            raise ValueError(f"Invalid field_type: {field_type}")

        # Select component
        if component == 'abs':
            cf_plot = Norm(field_cf)
        elif component == 'real':
            cf_plot = field_cf.real
        elif component == 'imag':
            cf_plot = field_cf.imag
        else:
            cf_plot = field_cf

        print(f"Plotting: |{field_label}| ({component})")

        draw_kwargs = kwargs.copy()
        if clipping:
            draw_kwargs['clipping'] = clipping

        Draw(BoundaryFromVolumeCF(cf_plot), self.mesh, **draw_kwargs)

    def _get_snapshot_context(
        self,
        domain: Optional[str]
    ) -> Tuple[str, HCurl, List[str]]:
        """
        Determine snapshot context based on domain specification.

        Returns
        -------
        tuple
            (snapshot_key, fes, available_ports)
        """
        if domain == 'global' or 'global' in self.snapshots:
            if 'global' not in self.snapshots:
                raise ValueError("Global snapshots not available.")
            return (
                'global',
                self._fes_global,
                self._external_ports if self.is_compound else self._ports
            )

        if domain is not None:
            if domain not in self.snapshots:
                raise ValueError(
                    f"Snapshots for domain '{domain}' not available. "
                    f"Available: {list(self.snapshots.keys())}"
                )
            return (
                domain,
                self._fes[domain],
                self.domain_port_map[domain]
            )

        # Auto-detect
        if self.n_domains == 1:
            d = self.domains[0]
            if d in self.snapshots:
                return d, self._fes[d], self._ports
            elif 'global' in self.snapshots:
                return 'global', self._fes_global, self._ports

        # For compound structures, require explicit specification
        raise ValueError(
            "For compound structures, specify 'domain' parameter. "
            f"Available snapshots: {list(self.snapshots.keys())}"
        )

    def _reconstruct_field(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int,
        snapshot_key: str,
        fes: HCurl,
        available_ports: List[str]
    ) -> GridFunction:
        """Reconstruct GridFunction from stored snapshot."""
        snapshots = self.snapshots.get(snapshot_key)
        if snapshots is None:
            raise ValueError(
                f"No snapshots for '{snapshot_key}'. "
                "Use store_snapshots=True in solve()."
            )

        port_idx = available_ports.index(excitation_port)
        n_modes = self._n_modes_per_port or 1
        n_ports = len(available_ports)

        # Snapshot indexing: [freq_idx * n_ports * n_modes + port_idx * n_modes + mode]
        snapshot_idx = freq_idx * n_ports * n_modes + port_idx * n_modes + excitation_mode

        if snapshot_idx >= snapshots.shape[1]:
            raise ValueError(
                f"Snapshot index {snapshot_idx} out of range "
                f"(max {snapshots.shape[1] - 1})"
            )

        E_gf = GridFunction(fes)
        E_gf.vec.FV().NumPy()[:] = snapshots[:, snapshot_idx]

        return E_gf

    def plot_s_parameters(
        self,
        db: bool = True,
        show_phase: bool = False,
        params: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        source: Literal['global', 'coupled', 'cascade'] = 'global',
        **kwargs
    ) -> None:
        """
        Plot S-parameters.

        Parameters
        ----------
        db : bool
            Plot magnitude in dB
        show_phase : bool
            Include phase plot
        params : list, optional
            Specific parameters to plot, e.g., ['S11', 'S21'].
            If None, plots all.
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        source : {'global', 'coupled', 'cascade'}
            Which results to plot:
            - 'global': Current global results
            - 'coupled': Cached coupled results
            - 'cascade': Cached cascade results
        **kwargs
            Additional arguments passed to plot functions
        """
        import matplotlib.pyplot as plt

        # Get appropriate S-matrix
        if source == 'coupled' and self._S_global_coupled is not None:
            S = self._S_global_coupled
            method_label = "Coupled"
        elif source == 'cascade' and self._S_global_cascade is not None:
            S = self._S_global_cascade
            method_label = "Cascade"
        else:
            S = self._S_matrix
            method_label = self._current_global_method or "Global"

        if S is None:
            raise ValueError("S-parameters not available. Call solve() first.")

        freqs_ghz = self.frequencies / 1e9
        n_ports = S.shape[1]

        # Determine which parameters to plot
        if params is None:
            params = [f'S{i+1}{j+1}' for i in range(n_ports) for j in range(n_ports)]

        # Setup figure
        if show_phase:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Plot each parameter
        for param in params:
            # Parse parameter name (e.g., 'S11', 'S21')
            i = int(param[1]) - 1
            j = int(param[2]) - 1

            if i >= n_ports or j >= n_ports:
                print(f"Warning: {param} out of range, skipping")
                continue

            s_val = S[:, i, j]

            if db:
                mag = 20 * np.log10(np.abs(s_val) + 1e-12)
                ax1.plot(freqs_ghz, mag, label=param, **kwargs)
            else:
                ax1.plot(freqs_ghz, np.abs(s_val), label=param, **kwargs)

            if show_phase:
                phase = np.angle(s_val, deg=True)
                ax2.plot(freqs_ghz, phase, label=param, **kwargs)

        # Format magnitude plot
        ax1.set_ylabel('Magnitude (dB)' if db else 'Magnitude')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title(f'S-Parameters ({method_label} Method)')

        # Format phase plot
        if show_phase:
            ax2.set_xlabel('Frequency (GHz)')
            ax2.set_ylabel('Phase (degrees)')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Frequency (GHz)')

        plt.tight_layout()
        plt.show()

    def plot_z_parameters(
        self,
        params: Optional[List[str]] = None,
        show_imag: bool = True,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        source: Literal['global', 'coupled', 'cascade'] = 'global',
        **kwargs
    ) -> None:
        """
        Plot Z-parameters.

        Parameters
        ----------
        params : list, optional
            Specific parameters to plot, e.g., ['Z11', 'Z21'].
            If None, plots all.
        show_imag : bool
            Plot imaginary part (reactance)
        figsize : tuple
            Figure size
        title : str, optional
            Plot title
        source : {'global', 'coupled', 'cascade'}
            Which results to plot
        **kwargs
            Additional arguments passed to plot functions
        """
        import matplotlib.pyplot as plt

        # Get appropriate Z-matrix
        if source == 'coupled' and self._Z_global_coupled is not None:
            Z = self._Z_global_coupled
            method_label = "Coupled"
        elif source == 'cascade' and self._Z_global_cascade is not None:
            Z = self._Z_global_cascade
            method_label = "Cascade"
        else:
            Z = self._Z_matrix
            method_label = self._current_global_method or "Global"

        if Z is None:
            raise ValueError("Z-parameters not available. Call solve() first.")

        freqs_ghz = self.frequencies / 1e9
        n_ports = Z.shape[1]

        # Determine which parameters to plot
        if params is None:
            params = [f'Z{i+1}{j+1}' for i in range(n_ports) for j in range(n_ports)]

        # Setup figure
        if show_imag:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Plot each parameter
        for param in params:
            i = int(param[1]) - 1
            j = int(param[2]) - 1

            if i >= n_ports or j >= n_ports:
                print(f"Warning: {param} out of range, skipping")
                continue

            z_val = Z[:, i, j]

            ax1.plot(freqs_ghz, np.real(z_val), label=f'Re({param})', **kwargs)

            if show_imag:
                ax2.plot(freqs_ghz, np.imag(z_val), label=f'Im({param})', **kwargs)

        # Format real part plot
        ax1.set_ylabel('Resistance (Ω)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title(f'Z-Parameters ({method_label} Method)')

        # Format imaginary part plot
        if show_imag:
            ax2.set_xlabel('Frequency (GHz)')
            ax2.set_ylabel('Reactance (Ω)')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Frequency (GHz)')

        plt.tight_layout()
        plt.show()

    def plot_domain_s_parameters(
        self,
        domain: str,
        db: bool = True,
        show_phase: bool = False,
        figsize: Tuple[float, float] = (10, 6),
        **kwargs
    ) -> None:
        """
        Plot S-parameters for a specific domain.

        Parameters
        ----------
        domain : str
            Domain name
        db : bool
            Plot magnitude in dB
        show_phase : bool
            Include phase plot
        figsize : tuple
            Figure size
        **kwargs
            Additional arguments passed to plot functions
        """
        import matplotlib.pyplot as plt

        if domain not in self._S_per_domain:
            raise ValueError(
                f"S-parameters for domain '{domain}' not available. "
                "Solve with per_domain=True and compute_s_params=True."
            )

        S = self.get_domain_s_matrix(domain)
        freqs_ghz = self.frequencies / 1e9
        n_ports = S.shape[1]
        domain_ports = self.domain_port_map[domain]

        # Setup figure
        if show_phase:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Plot all parameters
        for i in range(n_ports):
            for j in range(n_ports):
                label = f'S{i+1}{j+1}'
                s_val = S[:, i, j]

                if db:
                    mag = 20 * np.log10(np.abs(s_val) + 1e-12)
                    ax1.plot(freqs_ghz, mag, label=label, **kwargs)
                else:
                    ax1.plot(freqs_ghz, np.abs(s_val), label=label, **kwargs)

                if show_phase:
                    phase = np.angle(s_val, deg=True)
                    ax2.plot(freqs_ghz, phase, label=label, **kwargs)

        ax1.set_ylabel('Magnitude (dB)' if db else 'Magnitude')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'S-Parameters: Domain "{domain}"\nPorts: {domain_ports}')

        if show_phase:
            ax2.set_xlabel('Frequency (GHz)')
            ax2.set_ylabel('Phase (degrees)')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel('Frequency (GHz)')

        plt.tight_layout()
        plt.show()

    def plot_method_comparison(
        self,
        params: Optional[List[str]] = None,
        db: bool = True,
        figsize: Tuple[float, float] = (12, 8),
        **kwargs
    ) -> None:
        """
        Plot comparison between coupled and cascade methods.

        Parameters
        ----------
        params : list, optional
            Specific parameters to compare. If None, compares S11 and S21.
        db : bool
            Plot magnitude in dB
        figsize : tuple
            Figure size
        **kwargs
            Additional arguments passed to plot functions
        """
        import matplotlib.pyplot as plt

        if self._S_global_coupled is None or self._S_global_cascade is None:
            raise ValueError(
                "Both coupled and cascade results required. "
                "Call compare_methods() first."
            )

        freqs_ghz = self.frequencies / 1e9
        n_ports = self._S_global_coupled.shape[1]

        if params is None:
            if n_ports >= 2:
                params = ['S11', 'S21']
            else:
                params = ['S11']

        n_params = len(params)
        fig, axes = plt.subplots(n_params, 2, figsize=figsize, sharex=True)
        if n_params == 1:
            axes = axes.reshape(1, -1)

        for idx, param in enumerate(params):
            i = int(param[1]) - 1
            j = int(param[2]) - 1

            s_coupled = self._S_global_coupled[:, i, j]
            s_cascade = self._S_global_cascade[:, i, j]
            s_diff = np.abs(s_coupled - s_cascade)

            # Magnitude comparison
            ax1 = axes[idx, 0]
            if db:
                ax1.plot(freqs_ghz, 20 * np.log10(np.abs(s_coupled) + 1e-12),
                         'b-', label='Coupled', **kwargs)
                ax1.plot(freqs_ghz, 20 * np.log10(np.abs(s_cascade) + 1e-12),
                         'r--', label='Cascade', **kwargs)
                ax1.set_ylabel(f'|{param}| (dB)')
            else:
                ax1.plot(freqs_ghz, np.abs(s_coupled), 'b-', label='Coupled', **kwargs)
                ax1.plot(freqs_ghz, np.abs(s_cascade), 'r--', label='Cascade', **kwargs)
                ax1.set_ylabel(f'|{param}|')

            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f'{param}: Magnitude Comparison')

            # Difference plot
            ax2 = axes[idx, 1]
            ax2.semilogy(freqs_ghz, s_diff, 'k-', **kwargs)
            ax2.set_ylabel(f'|Δ{param}|')
            ax2.grid(True, alpha=0.3)
            ax2.set_title(f'{param}: Absolute Difference')

        axes[-1, 0].set_xlabel('Frequency (GHz)')
        axes[-1, 1].set_xlabel('Frequency (GHz)')

        plt.suptitle('Coupled vs Cascade Method Comparison', fontsize=12)
        plt.tight_layout()
        plt.show()

    # === Utility methods ===

    def get_frequency_index(self, freq_ghz: float) -> int:
        """
        Get the closest frequency index for a given frequency in GHz.

        Parameters
        ----------
        freq_ghz : float
            Frequency in GHz

        Returns
        -------
        int
            Index of closest frequency sample
        """
        if self.frequencies is None:
            raise ValueError("No solution available.")

        freq_hz = freq_ghz * 1e9
        idx = np.argmin(np.abs(self.frequencies - freq_hz))
        return idx

    def get_s_at_frequency(
        self,
        freq_ghz: float,
        source: Literal['global', 'coupled', 'cascade'] = 'global'
    ) -> np.ndarray:
        """
        Get S-matrix at a specific frequency.

        Parameters
        ----------
        freq_ghz : float
            Frequency in GHz
        source : {'global', 'coupled', 'cascade'}
            Which results to use

        Returns
        -------
        S : ndarray
            S-parameter matrix at the specified frequency
        """
        idx = self.get_frequency_index(freq_ghz)

        if source == 'coupled' and self._S_global_coupled is not None:
            return self._S_global_coupled[idx]
        elif source == 'cascade' and self._S_global_cascade is not None:
            return self._S_global_cascade[idx]
        else:
            if self._S_matrix is None:
                raise ValueError("S-parameters not available.")
            return self._S_matrix[idx]

    def get_z_at_frequency(
        self,
        freq_ghz: float,
        source: Literal['global', 'coupled', 'cascade'] = 'global'
    ) -> np.ndarray:
        """
        Get Z-matrix at a specific frequency.

        Parameters
        ----------
        freq_ghz : float
            Frequency in GHz
        source : {'global', 'coupled', 'cascade'}
            Which results to use

        Returns
        -------
        Z : ndarray
            Z-parameter matrix at the specified frequency
        """
        idx = self.get_frequency_index(freq_ghz)

        if source == 'coupled' and self._Z_global_coupled is not None:
            return self._Z_global_coupled[idx]
        elif source == 'cascade' and self._Z_global_cascade is not None:
            return self._Z_global_cascade[idx]
        else:
            if self._Z_matrix is None:
                raise ValueError("Z-parameters not available.")
            return self._Z_matrix[idx]

    def export_touchstone(
        self,
        filename: str,
        source: Literal['global', 'coupled', 'cascade'] = 'global',
        format: Literal['MA', 'DB', 'RI'] = 'MA',
        z0: float = 50.0
    ) -> None:
        """
        Export S-parameters to Touchstone format.

        Parameters
        ----------
        filename : str
            Output filename (will add .sNp extension)
        source : {'global', 'coupled', 'cascade'}
            Which results to export
        format : {'MA', 'DB', 'RI'}
            Data format (Magnitude-Angle, dB-Angle, Real-Imaginary)
        z0 : float
            Reference impedance
        """
        if source == 'coupled' and self._S_global_coupled is not None:
            S = self._S_global_coupled
        elif source == 'cascade' and self._S_global_cascade is not None:
            S = self._S_global_cascade
        else:
            S = self._S_matrix

        if S is None:
            raise ValueError("S-parameters not available.")

        n_ports = S.shape[1]
        n_freqs = len(self.frequencies)

        # Construct filename with proper extension
        if not filename.endswith(f'.s{n_ports}p'):
            filename = f"{filename}.s{n_ports}p"

        with open(filename, 'w') as f:
            # Header
            f.write(f"! Touchstone file exported from FrequencyDomainSolver\n")
            f.write(f"! Method: {source}\n")
            f.write(f"! Ports: {n_ports}\n")
            f.write(f"# GHz S {format} R {z0}\n")

            # Data
            for k in range(n_freqs):
                freq_ghz = self.frequencies[k] / 1e9
                line = f"{freq_ghz:.9e}"

                for i in range(n_ports):
                    for j in range(n_ports):
                        s_val = S[k, i, j]

                        if format == 'MA':
                            mag = np.abs(s_val)
                            ang = np.angle(s_val, deg=True)
                            line += f"  {mag:.9e}  {ang:.6f}"
                        elif format == 'DB':
                            db = 20 * np.log10(np.abs(s_val) + 1e-12)
                            ang = np.angle(s_val, deg=True)
                            line += f"  {db:.9e}  {ang:.6f}"
                        elif format == 'RI':
                            line += f"  {s_val.real:.9e}  {s_val.imag:.9e}"

                f.write(line + "\n")

        print(f"Exported to {filename}")

    def reset(self) -> None:
        """Reset solver state, clearing all results but keeping geometry."""
        self._clear_results()
        self.frequencies = None
        self._invalidate_cache()
        print("Solver state reset. Matrices retained.")

    def full_reset(self) -> None:
        """Full reset including matrices."""
        self.reset()

        # Clear matrices
        self._fes = {}
        self.M = {}
        self.K = {}
        self.B = {}

        self._fes_global = None
        self.M_global = None
        self.K_global = None
        self.B_global = None

        self._global_matrices_assembled = False
        self._per_domain_matrices_assembled = False

        # Clear port modes
        self.port_modes = None
        self.port_basis = None
        self._n_modes_per_port = None

        print("Full solver reset. All data cleared.")

    def __repr__(self) -> str:
        """String representation."""
        status = []
        status.append(f"FrequencyDomainSolver(order={self.order})")
        status.append(f"  Structure: {'Compound' if self.is_compound else 'Single'}")
        status.append(f"  Domains: {self.n_domains}")
        status.append(f"  Ports: {self._n_ports} total, {len(self._external_ports)} external")

        if self._per_domain_matrices_assembled or self._global_matrices_assembled:
            status.append(f"  Matrices: per_domain={self._per_domain_matrices_assembled}, "
                          f"global={self._global_matrices_assembled}")

        if self.frequencies is not None:
            status.append(f"  Solution: {len(self.frequencies)} frequencies, "
                          f"method={self._current_global_method}")

        return "\n".join(status)
