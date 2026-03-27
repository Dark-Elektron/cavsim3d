from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple, Union, Literal
import warnings
from datetime import datetime
import numpy as np
import copy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sl
from solvers.eigen_mixin import FDSEigenMixin
from utils.io_utils import deep_diff, strip_keys, check_source_files
from ngsolve import (
HCurl, BilinearForm, LinearForm, GridFunction, BND,
Integrate, InnerProduct, TaskManager, curl, dx, ds,
BoundaryFromVolumeCF, CoefficientFunction, Norm, BaseVector, preconditioners
)
from ngsolve.webgui import Draw

# Iterative: preconditioner MUST be registered before assembly
from ngsolve import Preconditioner as NGPreconditioner
from ngsolve import InnerProduct as NGInnerProduct
from ngsolve.krylovspace import GMRes, CG

from core.constants import mu0, eps0, c0, Z0
from solvers.base import BaseEMSolver, ParameterConverter
from solvers.ports import PortEigenmodeSolver
import cavsim3d.utils.printing as pr
from core.persistence import *

class FrequencyDomainSolver(BaseEMSolver, FDSEigenMixin):
    """
    Frequency-domain solver for electromagnetic problems.

    text
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
        'precond': 'local',
        'maxsteps': 500,
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
        self._mesh: Optional[Mesh] = None
        self.order = order
        self.bc = bc if bc is not None else getattr(geometry, 'bc', None)
        self.use_wave_impedance = use_wave_impedance
        
        # Per-domain storage (MUST be initialized before setting mesh)
        self._fes: Dict[str, HCurl] = {}
        self.M: Dict[str, sp.csr_matrix] = {}
        self.K: Dict[str, sp.csr_matrix] = {}
        self.B: Dict[str, np.ndarray] = {}

        # Global (coupled) storage
        self._fes_global: Optional[HCurl] = None
        self.M_global: Optional[sp.csr_matrix] = None
        self.K_global: Optional[sp.csr_matrix] = None
        self.B_global: Optional[np.ndarray] = None

        # Snapshots storage
        self.snapshots: Dict[str, np.ndarray] = {}

        self._project_path: Optional[str] = None
        
        # Port modes (shared across domains)
        self.port_solver: Optional[PortEigenmodeSolver] = None
        self.port_modes: Dict[str, Dict[int, CoefficientFunction]] = None
        self.port_basis: Dict[str, Dict[int, np.ndarray]] = None
        self._n_modes_per_port: int = None

        # Trigger structural detection and FES reconstruction via property setter
        if geometry is not None:
            self.mesh = geometry.mesh
        else:
            self.mesh = None

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

        # Assembly state flags
        self._global_matrices_assembled: bool = False
        self._per_domain_matrices_assembled: bool = False

        # Result-object caches (built lazily via .fom / .foms)
        self._fom_cache = None
        self._foms_cache = None

        # Project link (for automatic persistence)
        self._project_path: Optional[Path] = None
        self._project_name: Optional[str] = None
        self._project_base_dir: Optional[Union[str, Path]] = None
        self._project_ref = None
        self._loaded_config = None

        # Solver history (CST-style operation log)
        self._solver_history: List[dict] = []

        # Convergence info (iterative solver residuals)
        self._residuals: Dict[str, dict] = {}

        # Reset resonant mode cache
        self._resonant_mode_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Validate boundary conditions
        self._validate_boundary_conditions()

    @property
    def mesh(self) -> Optional[Mesh]:
        """NGSolve Mesh object."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Optional[Mesh]):
        """
        Set mesh and automatically (re)detect domains, ports, and (re)construct FES.
        """
        self._mesh = value
        
        if value is not None:
            # Update detection
            self.domains = self._detect_domains()
            self._ports = self._detect_ports()
            self.domain_port_map = self._build_domain_port_map()
            self.n_domains = len(self.domains)
            self._n_ports = len(self._ports)
            self.is_compound = self.n_domains > 1
    
            # External ports (for compound structures, first and last only)
            self._external_ports: List[str] = self._identify_external_ports()
            self._internal_ports: List[str] = self._identify_internal_ports()
            
            # Reconstruct FES for all domains (essential for field plotting after load)
            self._reconstruct_fes()

            # Reconstruct Port Solver for the new mesh
            if value is not self._mesh or self.port_solver is None:
                self.port_solver = PortEigenmodeSolver(value, self.order, self.bc)
                self.port_modes = None
                self.port_basis = None
            
            # Print structure info
            self._print_structure_info()
        else:
            self.domains = []
            self._ports = []
            self.domain_port_map = {}
            self.n_domains = 0
            self._n_ports = 0
            self.is_compound = False
            self._external_ports = []
            self._internal_ports = []
            self._fes = {}
            
            # Clear global matrices and flags
            self._fes_global = None
            self.M_global = None
            self.K_global = None
            self.B_global = None
            self._global_matrices_assembled = False
            self._per_domain_matrices_assembled = False
            
            # Clear port modes to force re-assembly on new mesh
            self.port_solver = None
            self.port_modes = None
            self.port_basis = None
            self._n_modes_per_port = None

    def _reconstruct_fes(self) -> None:
        """
        Reconstruct the Finite Element Space (FES) for each domain.
        
        This is called automatically when the mesh is set, ensuring that
        per-domain snapshots can be visualized even after a project reload.
        """
        if self._mesh is None:
            self._fes = {}
            return

        for domain in self.domains:
            try:
                # Create FES for this domain
                fes = HCurl(
                    self._mesh,
                    order=self.order,
                    dirichlet=self.bc,
                    definedon=self._mesh.Materials(domain)
                )
                self._fes[domain] = fes
            except Exception as e:
                warnings.warn(f"Could not reconstruct FES for domain '{domain}': {e}")

    def _validate_boundary_conditions(self) -> None:
        """
        Validate that boundary conditions are properly set.

        Issues warnings (not errors) for common misconfigurations so that
        the user can fix them before calling :meth:`solve`.
        """
        if self.mesh is None:
            return

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
                f"correctly, which also sets bc='default'.",
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
    def fes(self):
        """Convenience access to the global Finite Element Space."""
        return self._fes_global

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
        """Detect domains from mesh materials with fallback."""
        if self.mesh is None:
            return []
        materials = list(self.mesh.GetMaterials())
        if not materials:
            return ['default']

        # Try to find domains matching the 'cell' naming convention (sorted numerically)
        cell_domains = sorted(
            [m for m in materials if 'cell' in m.lower()],
            key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
        )

        if cell_domains:
            return cell_domains

        # If no 'cell' naming, include all unique non-default materials
        other_domains = sorted([m for m in materials if m.lower() != 'default'])
        
        if other_domains:
            return other_domains

        # Fallback to the first available material (likely 'default' or a single custom name)
        return [materials[0]]

    def _detect_ports(self) -> List[str]:
        """Detect ports from mesh boundaries with fallback."""
        if self.mesh is None:
            return []
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
        pr.info("\n" + "=" * 60)
        pr.info("Structure Topology")
        pr.info("=" * 60)
        pr.info(f"Type: {'Compound' if self.is_compound else 'Single'} structure")
        pr.info(f"Domains ({self.n_domains}): {self.domains}")
        pr.info(f"Total Ports ({self._n_ports}): {self._ports}")

        if self.is_compound:
            pr.info(f"External Ports ({len(self._external_ports)}): {self._external_ports}")
            pr.info(f"Internal Ports ({len(self._internal_ports)}): {self._internal_ports}")

        pr.info("\nDomain-Port Mapping:")
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
        pr.running("\n" + "=" * 60)
        pr.running("Assembling Matrices...")
        pr.running("=" * 60)

        # For single-domain, per-domain and global are the same
        if not self.is_compound:
            assemble_global = True
            assemble_per_domain = True

        # Solve port eigenmodes if missing or if the solver is blank
        needs_port_solve = (self.port_modes is None)
        if not needs_port_solve and self.port_solver is not None:
             # Check if the solver actually has the data for the modes we need
             if not self.port_solver.port_cutoff_kc:
                 needs_port_solve = True

        if needs_port_solve:
            if self.port_solver is None:
                # Emergency re-initialization
                from solvers.ports import PortEigenmodeSolver
                self.port_solver = PortEigenmodeSolver(self.mesh, self.order, self.bc)
                
            pr.running("Solving port eigenmodes...")
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

        if self._project_path is not None:
            self.save()
        return self._get_matrix_summary()

    def _assemble_per_domain_matrices(self) -> None:
        """Assemble matrices for each domain independently."""
        pr.debug("\n--- Assembling Per-Domain Matrices ---")

        for domain in self.domains:
            pr.debug(f"\nDomain: {domain}")

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

            pr.debug(f"  FES ndof: {fes.ndof}")
            pr.debug(f"  K shape: {self.K[domain].shape}, nnz: {self.K[domain].nnz}")
            pr.debug(f"  M shape: {self.M[domain].shape}, nnz: {self.M[domain].nnz}")
            pr.debug(f"  B shape: {self.B[domain].shape}")

    def _assemble_global_matrices(self) -> None:
        """Assemble matrices for the full (coupled) structure."""
        pr.debug("\n--- Assembling Global Matrices (Coupled System) ---")

        # Create FES for entire mesh
        print("boundary condition: ", self.bc)
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

        pr.debug(f"  Global FES ndof: {fes.ndof}")
        pr.debug(f"  K_global shape: {self.K_global.shape}, nnz: {self.K_global.nnz}")
        pr.debug(f"  M_global shape: {self.M_global.shape}, nnz: {self.M_global.nnz}")
        pr.debug(f"  B_global shape: {self.B_global.shape}")

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
            f"tol={opts.get('tol', 1e-6)}, maxsteps={opts.get('maxsteps', 100)}")
        print(f"  FES ndof: {fes.ndof}")
        return fes

    # === Frequency domain solve ===

    def solve(
            self,
            fmin: float = None,
            fmax: float = None,
            nsamples: int = None,
            config: Optional[Dict] = None,
            **kwargs
    ) -> Dict:
        """
        Solve frequency sweep.

        Supports passing arguments directly or via a 'config' dictionary.
        Individual keyword arguments override the config dictionary.

        Parameters
        ----------
        fmin : float, optional
            Minimum frequency in GHz
        fmax : float, optional
            Maximum frequency in GHz
        nsamples : int, optional
            Number of frequency samples
        config : dict, optional
            Dictionary containing solve parameters
        **kwargs :
            Individual solve parameters (order, nportmodes, store_snapshots, etc.)
        """
        # 1. Merge config and kwargs
        cfg = (config or {}).copy()
        cfg.update(kwargs)

        # 2. Extract core parameters with defaults
        fmin = fmin if fmin is not None else cfg.get('fmin')
        fmax = fmax if fmax is not None else cfg.get('fmax')
        nsamples = nsamples if nsamples is not None else cfg.get('nsamples', 100)

        # Validate mandatory frequency range
        if fmin is None or fmax is None:
            raise ValueError("fmin and fmax must be provided (either directly or via config).")

        # 3. Extract other options from merged cfg
        order = cfg.get('order')
        nportmodes = cfg.get('nportmodes')
        store_snapshots = cfg.get('store_snapshots', True)
        compute_s_params = cfg.get('compute_s_params', True)
        per_domain = cfg.get('per_domain', True)
        global_method = cfg.get('global_method', 'coupled')
        solver_type = cfg.get('solver_type', 'iterative')
        iterative_opts = cfg.get('iterative_opts')
        rerun = cfg.get('rerun', False)
        verbose = cfg.get('verbose', False)

        # Set verbosity level
        pr.set_verbosity(verbose)

        # --- Config comparison ---
        diffs = []  # Initialize outside the block so it's always defined
        if self._loaded_config and not rerun:
            # Check frequencies if they were previously solved
            if self._loaded_config.get('fmin') is not None:
                if not np.isclose(fmin, self._loaded_config['fmin']):
                    diffs.append(f"fmin: {self._loaded_config['fmin']} -> {fmin}")
                if not np.isclose(fmax, self._loaded_config['fmax']):
                    diffs.append(f"fmax: {self._loaded_config['fmax']} -> {fmax}")
                if nsamples != self._loaded_config['nsamples']:
                    diffs.append(f"nsamples: {self._loaded_config['nsamples']} -> {nsamples}")
            
            # Check solver settings
            if order is not None and order != self._loaded_config.get('order'):
                diffs.append(f"order: {self._loaded_config.get('order')} -> {order}")
            if nportmodes is not None and nportmodes != self._loaded_config.get('n_modes_per_port'):
                diffs.append(f"nportmodes: {self._loaded_config.get('n_modes_per_port')} -> {nportmodes}")
            
            current_history = getattr(self.geometry, '_history', [])
            loaded_history = self._loaded_config.get('geometry_history', [])
            component_sources = self._loaded_config.get('component_sources', {})

            # 1. Strip timestamps AND filepaths — filepaths are checked separately via hash
            keys_to_ignore = {'timestamp', 'filepath'}
            current_clean = strip_keys(current_history, keys_to_ignore)
            loaded_clean = strip_keys(loaded_history, keys_to_ignore)

            # 2. Compare the rest of the history (parameters, ops, mesh settings, etc.)
            if current_clean != loaded_clean:
                history_diffs = deep_diff(loaded_clean, current_clean, path="geometry_history")
                diffs.append("geometry/history has changed:")
                for d in history_diffs:
                    diffs.append(f"  {d}")

            # 3. Separately check if actual source files have changed via content hash
            source_diffs = check_source_files(component_sources, geometry_dir="geometry")
            if source_diffs:
                diffs.append("geometry source file(s) have changed:")
                for d in source_diffs:
                    diffs.append(f"  {d}")

            # 4. Report
            if diffs:
                msg = "\n  [WARNING] Simulation configuration has changed since last save/load:\n"
                for d in diffs:
                    msg += f"    - {d}\n"
                msg += "  Existing results may be invalid. Use rerun=True to recompute."
                pr.warning(msg)

        # --- Rerun protection ---
        has_results = (
            (self._S_matrix is not None and self._Z_matrix is not None)
            or bool(self._S_per_domain)
            or self._Z_global_coupled is not None
            or self._foms_cache is not None
            or self._fom_cache is not None
        )
        # Result exists only if frequencies are also present
        if has_results and (self.frequencies is None or len(self.frequencies) == 0):
            has_results = False

        if has_results and not rerun:
            if diffs:
                pr.warning("  Valid results found, but configuration changed since last save/load! Returning previous results anyway because rerun=False.")
            else:
                pr.info("  Valid simulation results found matching current config. Returning previous results. (Use rerun=True to force recompute)")
            return self._build_results_dict(compute_s_params, per_domain, global_method)
        elif not has_results and not rerun:
            pr.info(
                f"  No valid results found. Forcing compute. "
                f"(Z_coupled={self._Z_global_coupled is not None}, "
                f"foms={self._foms_cache is not None}, "
                f"fmin={self._loaded_config.get('fmin') if self._loaded_config else None})"
            )


        # --- Mesh synchronization and validation ---
        if self.mesh is None and self.geometry and self.geometry.mesh:
            # Sync solver mesh with geometry if available
            self.mesh = self.geometry.mesh
            # Also sync back to project to ensure consistency and auto-save
            if hasattr(self, '_project_ref') and self._project_ref:
                self._project_ref.mesh = self.mesh
            
        if self.mesh is None:
            raise RuntimeError(
                "FrequencyDomainSolver has no mesh. "
                "Please generate a mesh using geometry.generate_mesh() before solving."
            )
            
        # Ensure port_solver is initialized if mesh exists but solver was cleared
        if self.port_solver is None and self.mesh is not None:
             from solvers.ports import PortEigenmodeSolver
             self.port_solver = PortEigenmodeSolver(self.mesh, self.order, self.bc)

        if rerun:
            # Explicitly clear ROM/Concat children of existing FOM caches
            if self._fom_cache:
                self._fom_cache.clear_rom()
            if self._foms_cache:
                self._foms_cache.clear_roms()
            self._clear_results()

        if order is not None and order != self.order:
            self.order = order
            # If mesh exists, we must recreate the Port solver and clear old modes
            if self.mesh is not None:
                self.port_solver = PortEigenmodeSolver(self.mesh, self.order, self.bc)
                self.port_modes = None
                self.port_basis = None
                # Invalidate assembled matrices if order changed
                self._per_domain_matrices_assembled = False
                self._global_matrices_assembled = False

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
        self._ensure_matrices_assembled(per_domain, global_method, nportmodes=nportmodes)

        # Clear previous results
        self._clear_results()

        # Print solve configuration
        self._print_solve_config(per_domain, global_method)

        # Solve per-domain if requested
        if per_domain:
            pr.running("Per-Domain Solve")
            self._solve_per_domain(store_snapshots,
                                solver_type=solver_type,
                                iter_opts=iter_opts)
        else:
            # Compute global results
            # if global_method == 'coupled':
            pr.running("Global Coupled Solve")
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

        # Record in solver history
        self._solver_history.append({
            'op': 'solve',
            'fmin': fmin,
            'fmax': fmax,
            'nsamples': nsamples,
            'solver_type': solver_type,
            'global_method': global_method,
            'per_domain': per_domain,
            'timestamp': datetime.now().isoformat(),
        })

        if self._project_path is not None:
            self.save()

        return self._build_results_dict(compute_s_params, per_domain, global_method)

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
            pr.info(f"\nSolving domain: {domain}")

            domain_ports = self.domain_port_map[domain]
            fes = self._fes[domain]

            # Resolve solver type for this domain
            st = self._resolve_solver_type(solver_type, fes)
            pr.debug(f"  Solver type: {st}")

            # Prepare multigrid if iterative
            if st == 'iterative':
                fes = self._prepare_iterative(fes, iter_opts)
                self._fes[domain] = fes

            # Get pre-assembled B matrix for fast Z extraction
            B = self.B[domain]

            # Build excitation ordering
            excitation_keys = []
            for pm, port_m in enumerate(domain_ports):
                if port_m not in self.port_modes:
                    continue
                for mode_m in sorted(self.port_modes[port_m].keys()):
                    excitation_keys.append((pm, port_m, mode_m))

            n_excitations = len(excitation_keys)
            n_freqs = len(self.frequencies)

            u, v = fes.TnT()

            # ============================================================
            # PRE-ASSEMBLE: Frequency-independent RHS vectors (NGSolve native)
            # We store as list of BaseVectors for efficient NGSolve operations
            # ============================================================
            pr.debug(f"  Pre-assembling {n_excitations} RHS vectors...")
            
            # Create a template vector for cloning
            template_vec = LinearForm(fes)
            with TaskManager():
                template_vec.Assemble()
            template_vec = template_vec.vec
            
            # Store base RHS vectors (without omega scaling)
            rhs_base_vectors = []
            for col, (pm, port_m, mode_m) in enumerate(excitation_keys):
                f = LinearForm(fes)
                f += InnerProduct(
                    self.port_modes[port_m][mode_m], v.Trace()
                ) * ds(port_m)
                with TaskManager():
                    f.Assemble()
                
                # Clone the vector (not just reference)
                vec = template_vec.CreateVector()
                vec.data = f.vec
                rhs_base_vectors.append(vec)

            # ============================================================
            # PRE-COMPUTE: Free DOF indices (once per domain)
            # ============================================================
            freedofs = fes.FreeDofs()
            free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]], dtype=np.int64)
            n_free = len(free_idx)
            pr.debug(f"  DOFs: {fes.ndof} total, {n_free} free")

            # Pre-allocate results
            Z_matrix = np.zeros((n_freqs, n_excitations, n_excitations), dtype=complex)
            snapshots_list = [] if store_snapshots else None

            total_iter_steps = 0
            freq_iters = []
            freq_residuals = []

            # Pre-allocate reusable vectors for the solve loop
            rhs_scaled = template_vec.CreateVector()
            sol_vec = template_vec.CreateVector()

            # ============================================================
            # FREQUENCY LOOP
            # ============================================================
            for kk, freq in enumerate(self.frequencies):
                omega = 2 * np.pi * freq

                # Build system matrix: A(ω) = K - ω²M
                a_form = BilinearForm(fes)
                a_form += (1 / mu0) * curl(u) * curl(v) * dx(domain)
                a_form += -omega ** 2 * eps0 * u * v * dx(domain)

                # Prepare solver
                if st == 'direct':
                    a_form.Assemble()
                    # Use NGSolve's direct solver with PARDISO/UMFPACK
                    # Options: "sparsecholesky", "pardiso", "pardisospd", "umfpack", "mumps"
                    with TaskManager():
                        inv_a = a_form.mat.Inverse(
                            freedofs=freedofs,
                            inverse="pardiso"  # or "umfpack", "sparsecholesky"
                        )
                else:
                    if iter_opts['precond'] == 'local':
                        precond = preconditioners.Local(a_form)
                    elif iter_opts['precond'] == 'multigrid':
                        precond = preconditioners.MultiGrid(a_form)
                        # precond = preconditioners.MultiGrid(a_form,
                        #                                 smoother='point',
                        #                                 smoothingsteps=5,
                        #                                 cycle='W',
                        #                                 coarsetype='direct',
                        #                                 )
                    elif iter_opts['bddc']:
                        precond = preconditioners.BDDC(a_form)
                    else:
                        print('Preconditioner not found, defaulting to local.')
                        precond = preconditioners.Local(a_form)
                
                with TaskManager():
                    a_form.Assemble() # only assemble a_form after attaching a preconditioner
                    if iter_opts['precond'].lower() == 'direct':
                        precond = a_form.mat.Inverse(fes.FreeDofs(), inverse="pardiso")

                # ============================================================
                # SOLVE: All excitations at this frequency
                # Factorization is done once, each solve reuses it
                # ============================================================
                x_all = np.zeros((fes.ndof, n_excitations))

                for col in range(n_excitations):
                    # Scale pre-assembled RHS by omega
                    rhs_scaled.data = omega * rhs_base_vectors[col]

                    if st == 'direct':
                        # Direct solve (forward/backward substitution only - factorization reused)
                        sol_vec.data = inv_a * rhs_scaled
                        freq_iters.append(0)
                        freq_residuals.append(0.0)
                    else:
                        # Iterative solve with initial guess from previous excitation
                        if col > 0:
                            # Use previous solution as initial guess
                            sol_vec.FV().NumPy()[:] = x_all[:, col - 1].real
                        else:
                            sol_vec[:] = 0

                        sol_vec, iters, res = self._solve_system(
                            fes, a_form, rhs_scaled, precond, iter_opts, sol_vec
                        )
                        total_iter_steps += iters
                        freq_iters.append(iters)
                        freq_residuals.append(res)

                    # Store solution
                    x_all[:, col] = sol_vec.FV().NumPy()

                # ============================================================
                # Fast Z extraction: Z = 1j * B^H @ X
                # ============================================================
                Z_matrix[kk, :, :] = 1j * (B.T.conj() @ x_all)

                # Store snapshots if requested
                if store_snapshots:
                    for col in range(n_excitations):
                        snapshots_list.append(x_all[:, col].copy())

                # Progress reporting
                if (kk + 1) % max(1, n_freqs // 5) == 0 or kk == n_freqs - 1:
                    elapsed = time.time() - t_domain_start
                    pr.debug(f"    [{kk + 1}/{n_freqs}] {elapsed:.1f}s elapsed")

            # ============================================================
            # POST-PROCESS: Convert Z_matrix to dict format
            # ============================================================
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
            pr.done(f"  {msg}")

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
        B = self.B_global

        u, v = fes.TnT()

        # ============================================================
        # PRE-ASSEMBLE: Frequency-independent RHS vectors
        # ============================================================
        pr.debug(f"  Pre-assembling {n_excitations} RHS vectors...")
        
        template_vec = LinearForm(fes)
        with TaskManager():
            template_vec.Assemble()
        template_vec = template_vec.vec
        rhs_base_vectors = []
        
        for col, (pm, port_m, mode_m) in enumerate(excitation_keys):
            f = LinearForm(fes)
            f += InnerProduct(
                self.port_modes[port_m][mode_m], v.Trace()
            ) * ds(port_m)
            
            with TaskManager():
                f.Assemble()
            
            vec = template_vec.CreateVector()
            vec.data = f.vec
            rhs_base_vectors.append(vec)

        # ============================================================
        # PRE-COMPUTE: Free DOF indices
        # ============================================================
        freedofs = fes.FreeDofs()
        free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]], dtype=np.int64)
        n_free = len(free_idx)
        pr.debug(f"  DOFs: {fes.ndof} total, {n_free} free")

        # Pre-allocate results
        self._Z_matrix = np.zeros((n_freqs, n_excitations, n_excitations), dtype=complex)
        snapshots_list = [] if store_snapshots else None

        total_iter_steps = 0
        freq_iters = []
        freq_residuals = []

        # Pre-allocate reusable vectors
        rhs_scaled = template_vec.CreateVector()
        sol_vec = template_vec.CreateVector()

        # ============================================================
        # FREQUENCY LOOP
        # ============================================================
        for kk, freq in enumerate(self.frequencies):
            if kk % max(1, n_freqs // 10) == 0:
                pr.debug(f"  Frequency {kk + 1}/{n_freqs}: {freq / 1e9:.4f} GHz")

            omega = 2 * np.pi * freq

            # Build system matrix
            a_form = BilinearForm(fes)
            a_form += (1 / mu0) * curl(u) * curl(v) * dx
            a_form += -omega ** 2 * eps0 * u * v * dx

            # Prepare solver
            if st == 'direct':
                with TaskManager():
                    a_form.Assemble()
                    inv_a = a_form.mat.Inverse(
                        freedofs=freedofs,
                        inverse="pardiso"
                    )
            else:
                if iter_opts['precond'].lower() == 'local':
                    precond = preconditioners.Local(a_form)
                elif iter_opts['precond'].lower() == 'multigrid':
                    precond = preconditioners.MultiGrid(a_form)
                elif iter_opts['precond'].lower() == 'bddc':
                    precond = preconditioners.BDDC(a_form)
                elif iter_opts['precond'].lower() == 'hcurlamg':
                    precond = preconditioners.HCurlAMG(a_form)
                else:
                    print('Preconditioner not found, defaulting to local.')
                    precond = preconditioners.Local(a_form)

                with TaskManager():
                    a_form.Assemble() # only assemble a_form after attaching a preconditioner
                    if iter_opts['precond'].lower() == 'direct':
                        precond = a_form.mat.Inverse(fes.FreeDofs(), inverse="pardiso")

            # Solve for all excitations
            x_all = np.zeros((fes.ndof, n_excitations))

            for col in range(n_excitations):
                rhs_scaled.data = omega * rhs_base_vectors[col]

                if st == 'direct':
                    sol_vec.data = inv_a * rhs_scaled
                    freq_iters.append(0)
                    freq_residuals.append(0.0)
                else:
                    if col > 0:
                        sol_vec.FV().NumPy()[:] = x_all[:, col - 1].real
                    else:
                        sol_vec[:] = 0

                    sol_vec, iters, res = self._solve_system(
                        fes, a_form, rhs_scaled, precond, iter_opts, sol_vec
                    )
                    total_iter_steps += iters
                    freq_iters.append(iters)
                    freq_residuals.append(res)

                x_all[:, col] = sol_vec.FV().NumPy()

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
        pr.done(f"  {msg}")

        self._store_residuals('global', n_freqs, freq_iters, freq_residuals, st)

    def _solve_system(self, fes, a_form, f_vec, precond, opts: Dict, x0: Optional[np.ndarray] = None):
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
        # def _count_iter(sol_vec):
        def _count_iter(sol_vec, it=3):
            iter_count[0] += 1

        # GMRes is a function: GMRes(A, b, pre=...) → solution vector
        x = GridFunction(fes)

        # initialise solution with previous solution
        sol = f_vec.CreateVector()
        if x0 is not None:
            # print('it is in here:: ', x0)
            sol.data = x0 # might be confusing but solution modifies the initial guess sol internally and returns it

        with TaskManager():
            sol = GMRes(
                A=a_form.mat,
                b=f_vec,
                x=sol,
                pre=precond.mat,
                # freedofs=fes.FreeDofs(),  # only necessary f no preconditioner
                maxsteps=opts['maxsteps'],
                tol=opts['tol'],
                printrates=opts['printrates'],
                callback=_count_iter,
            )
            # sol = CG(
            #     mat=a_form.mat,
            #     rhs=f_vec,
            #     # sol=sol,
            #     pre=precond.mat,
            #     # initialize=False,
            #     # freedofs=fes.FreeDofs(),  # only necessary f no preconditioner
            #     maxsteps=opts['maxsteps'],
            #     tol=opts['tol'],
            #     printrates=opts['printrates'],
            #     callback=_count_iter,
            # )
            
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

    def _ensure_matrices_assembled(
        self,
        per_domain: bool,
        global_method: Optional[str],
        nportmodes: Optional[int] = None
    ) -> None:
        """Ensure required matrices are assembled."""
        needs_global = (global_method == 'coupled')
        needs_per_domain = per_domain

        # Check if port modes exist or if we need a different number of modes
        needs_recompute = False
        if nportmodes is not None and nportmodes != self._n_modes_per_port:
            needs_recompute = True
            self.port_modes = None # Force recompute

        if self.port_modes is None:
            self.assemble_matrices(
                nportmodes=nportmodes or self._n_modes_per_port or 1,
                assemble_global=needs_global,
                assemble_per_domain=needs_per_domain
            )
            return

        # Assemble missing matrices
        if needs_global and (not self._global_matrices_assembled or needs_recompute):
            self._assemble_global_matrices()
            self._global_matrices_assembled = True

        if needs_per_domain and (not self._per_domain_matrices_assembled or needs_recompute):
            self._assemble_per_domain_matrices()
            self._per_domain_matrices_assembled = True

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Optional[Union[str, Path]] = None, project_name: Optional[str] = None,
             base_dir: Optional[Union[str, Path]] = None):
        """
        Save the solver state to disk.

        Parameters
        ----------
        path : Path, optional
            Specific directory to save to. If provided, overrides project_name/base_dir.
            Usually this is managed by EMProject.
        project_name : str, optional
            Name of the project.
        base_dir : str or Path, optional
            Base directory for simulations.
        """
        import h5py
        from core.persistence import ProjectManager, H5Serializer
        import pickle as _pkl
        from datetime import datetime
        import json

        if path:
            fds_path = Path(path)
            fds_path.mkdir(parents=True, exist_ok=True)
            # Try to infer project_root as parent if it looks like a subfolder
            if fds_path.name == 'fds':
                project_path = fds_path.parent
            else:
                project_path = fds_path
            project_name = project_name or getattr(self, '_project_name', project_path.name)
        elif getattr(self, '_project_path', None):
            # If we are part of a project, save to 'fds' subfolder of the project root
            project_path = Path(self._project_path)
            fds_path = project_path / 'fds'
            fds_path.mkdir(parents=True, exist_ok=True)
            project_name = project_name or getattr(self, '_project_name', project_path.name)
        else:
            project_name = project_name or getattr(self, '_project_name', None) or "untitled"
            pm = ProjectManager(base_dir or "simulations")
            project_path = pm.prepare_project(project_name)
            fds_path = project_path / 'fds'
            fds_path.mkdir(parents=True, exist_ok=True)

        self._project_name = project_name
        self._project_path = project_path

        # 1. Config / metadata
        config = {
            "project_name": self._project_name,
            "fmin": self.frequencies[0] / 1e9 if self.frequencies is not None else None,
            "fmax": self.frequencies[-1] / 1e9 if self.frequencies is not None else None,
            "nsamples": len(self.frequencies) if self.frequencies is not None else None,
            "order": self.order,
            "bc": self.bc,
            "use_wave_impedance": self.use_wave_impedance,
            "is_compound": self.is_compound,
            "n_domains": self.n_domains,
            "domains": self.domains,
            "n_ports": len(self._ports),
            "ports": self._ports,
            "external_ports": self._external_ports,
            "internal_ports": self._internal_ports,
            "n_modes_per_port": self._n_modes_per_port,
            "global_matrices_assembled": self._global_matrices_assembled,
            "per_domain_matrices_assembled": self._per_domain_matrices_assembled,
            "current_global_method": self._current_global_method,
            "has_results": bool(
                self._Z_global_coupled is not None
                or self._Z_per_domain
                or self._fom_cache is not None
            ),
            "solver_history": self._solver_history,
            "geometry_history": getattr(self.geometry, '_history', []),
            "timestamp": datetime.now().isoformat(),
        }

        ProjectManager.save_json(fds_path, config, filename="config.json")

        # 2. Port modes - save via PortEigenmodeSolver (includes all port data)
        if hasattr(self, 'port_solver') and self.port_solver is not None:
            port_dir = fds_path / "port_modes"
            port_dir.mkdir(parents=True, exist_ok=True)

            # Use the new save method that extracts raw numpy data
            self.port_solver.save_to_file(port_dir / "port_modes.pkl")

        # 3. FOMs & ROMs (Hierarchical Persistence)
        if self._fom_cache is not None or self._Z_global_coupled is not None:
            fom_path = fds_path / "fom"
            self.fom.save(fom_path)

            # Nested ROM for global FOM
            if hasattr(self._fom_cache, '_rom_cache') and self._fom_cache._rom_cache is not None:
                self._fom_cache._rom_cache.save(fom_path / "rom")

        if self.is_compound and (self._foms_cache is not None or self._Z_per_domain):
            foms_path = fds_path / "foms"
            self.foms.save(foms_path)

            # Nested ROMs for per-domain FOMs
            if hasattr(self._foms_cache, '_roms_cache') and self._foms_cache._roms_cache is not None:
                self._foms_cache._roms_cache.save(foms_path / "roms")

        # 3. Save eigenmodes
        self.save_eigenmodes()

        pr.info(f"FrequencyDomainSolver saved to {fds_path}")
        return fds_path

    @classmethod
    def load_from_path(cls,
                       path: Union[str, Path],
                       geometry: Optional[BaseGeometry] = None,
                       mesh=None,
                       order: int = 3,
                       bc: str = 'default') -> 'FrequencyDomainSolver':
        """
        Load a solver state from a specific directory.
        
        Parameters
        ----------
        path : Path
            The directory containing the fds/ folder or solver results.
        geometry : BaseGeometry, optional
            The geometry associated with this solver. If None, we expect 
            the mesh to be available in the parent directory or provided via geometry.
        """
        import pickle as _pkl
        import h5py
        from core.persistence import H5Serializer
        import json
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Solver path {path} does not exist.")
            
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # If geometry is not provided, we might have limited functionality 
        # but we can still load results and matrices.
        fds = cls(
            geometry=geometry,
            order=config.get("order", 3),
            bc=config.get("bc"),
            use_wave_impedance=config.get("use_wave_impedance", True)
        )
        
        # Load matrices, snapshots, etc. from 'path'
        fds.mesh = mesh  # Set mesh before loading internals
        fds._load_internal(path, config)
        
        return fds

    def _load_internal(self, path: Path, config: dict):
        """Internal helper to load solver state from a directory."""
        import pickle as _pkl
        import h5py
        from core.persistence import H5Serializer, ProjectManager
        self._loaded_config = config

        # Restore state flags and topology
        self.is_compound = config.get("is_compound", self.is_compound)
        self.domains = config.get("domains", self.domains)
        self.n_domains = config.get("n_domains", len(self.domains))
        self._ports = config.get("ports", self._ports)
        self._external_ports = config.get("external_ports", self._external_ports)
        self._internal_ports = config.get("internal_ports", self._internal_ports)
        
        self._n_modes_per_port = config.get("n_modes_per_port", self._n_modes_per_port)
        self._global_matrices_assembled = config.get("global_matrices_assembled", False)
        self._per_domain_matrices_assembled = config.get("per_domain_matrices_assembled", False)
        self._current_global_method = config.get("current_global_method", None)
        
        # Robustly reconstitute frequencies from metadata right away
        f_min, f_max, n_samp = config.get("fmin"), config.get("fmax"), config.get("nsamples")
        if f_min is not None and f_max is not None and n_samp is not None:
            self.frequencies = np.linspace(f_min, f_max, n_samp) * 1e9
            
        # Determine the target result folder (fom or foms) INSIDE the fds directory
        fom_root_single = path / "fom"
        fom_root_compound = path / "foms"

        # 1. Result caches (Mirroring the hierarchy)
        # Load per-domain results for compound systems
        if fom_root_compound.exists() and self.is_compound:
            from solvers.results import FOMCollection
            try:
                # FOMCollection.load handles matrix loading into self (the _fds_ref)
                self._foms_cache = FOMCollection.load(fom_root_compound, _fds_ref=self)
                if self._foms_cache:
                    self.frequencies = self._foms_cache.frequencies
                
                # Nested ROMs inside foms/roms
                roms_path = fom_root_compound / "roms"
                if roms_path.exists():
                    from rom.reduction import ModelOrderReduction
                    from solvers.results import ROMCollection
                    mor_loaded = ModelOrderReduction.load(roms_path, solver=self)
                    self._foms_cache._roms_cache = ROMCollection(_fds_ref=self, _mor_ref=mor_loaded)
            except Exception as e:
                warnings.warn(f"Could not load FOM collection from {fom_root_compound}: {e}")
                
        # Load global 'fom' results (exists for single structures AND compound structures with global solve)
        if fom_root_single.exists():
            from solvers.results import FOMResult
            try:
                # FOMResult.load handles matrix loading into self (the _solver_ref)
                self._fom_cache = FOMResult.load(fom_root_single, _solver_ref=self)
                if self._fom_cache:
                    if self.frequencies is None:
                        self.frequencies = self._fom_cache.frequencies
                self._Z_global_coupled = self._fom_cache._Z_matrix
                self._S_global_coupled = self._fom_cache._S_matrix

                # Nested ROM inside fom/rom
                rom_path = fom_root_single / "rom"
                if rom_path.exists():
                    from rom.reduction import ModelOrderReduction
                    self._fom_cache._rom_cache = ModelOrderReduction.load(rom_path, solver=self)
            except Exception as e:
                print(f"Warning: Could not load FOM result: {e}")

        # ------------------------------------------------------------------
        # 2. Port modes
        # ------------------------------------------------------------------
        port_dir = path / "port_modes"
        if port_dir.exists() and self.mesh is not None:
            pm_file = port_dir / "port_modes.pkl"

            try:
                if pm_file.exists():
                    # Import the PortEigenmodeSolver class
                    from solvers.ports import PortEigenmodeSolver

                    # Ensure we have fes_global
                    if self._fes_global is None:
                        from ngsolve import HCurl
                        self._fes_global = HCurl(self.mesh, order=self.order, dirichlet=self.bc)

                    # Load using the new method that reconstructs NGSolve objects
                    self.port_solver = PortEigenmodeSolver.load_from_file(
                        pm_file,
                        mesh=self.mesh,
                        fes_full=self._fes_global
                    )

                    # Set convenience references
                    self.port_modes = self.port_solver.port_modes
                    self.port_basis = self.port_solver.port_basis
            except Exception as e:
                # NgException (vector size mismatch) or other pickle errors
                warnings.warn(
                    f"Could not load port modes: {e}. "
                    f"Port modes will need to be recomputed.",
                    UserWarning,
                    stacklevel=2
                )
                self.port_solver = None
                self.port_modes = None
                self.port_basis = None
        
        elif port_dir.exists() and self.mesh is None:
            warnings.warn(
                "Port modes found but no mesh available to load them into. "
                "Skipping port mode loading.",
                UserWarning,
                stacklevel=2
            )

        # 3. Load eigenmodes
        self.load_eigenmodes()

        # ------------------------------------------------------------------
        # 3. Snapshots (Legacy and Per-Domain)
        # ------------------------------------------------------------------
        # Legacy aggregated snapshots
        snap_path = path / "snapshots.h5"
        if snap_path.exists():
            with h5py.File(snap_path, "r") as fh:
                if "snapshots" in fh:
                    for key in fh["snapshots"]:
                        self.snapshots[key] = H5Serializer.load_dataset(fh[f"snapshots/{key}"])
        
        # Per-domain snapshots (loaded via FOMCollection.load/FOMResult.load usually,
        # but we ensure they are covered here if missed or for direct access)
        for domain in self.domains:
            d_snap_path = path / f"snapshots_{domain}.h5"
            if d_snap_path.exists():
                with h5py.File(d_snap_path, "r") as fh:
                    if "field_snapshots" in fh:
                        self.snapshots[domain] = H5Serializer.load_dataset(fh["field_snapshots"])


        print(f"FrequencyDomainSolver state loaded from {path}")

    def _clear_results(self) -> None:
        """Clear previous solve results."""
        self._Z_per_domain = {}
        self._S_per_domain = {}
        self._Z_global_coupled = None
        self._S_global_coupled = None
        self._Z_global_cascade = None
        self._S_global_cascade = None
        self._current_global_method = None
        self.snapshots = {}
        self._residuals = {}
        # Invalidate result-object caches
        self._fom_cache = None
        self._foms_cache = None
        # Clear resonant modes
        self._resonant_mode_cache = {}

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
            nsamples: int = 20,
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
        eigenvectors: Optional[np.ndarray] = None,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Filter and sort eigenvalues (and optionally eigenvectors).

        Parameters
        ----------
        eigenvalues : ndarray
            Raw eigenvalues
        eigenvectors : ndarray, optional
            Raw eigenvectors. Shape (ndof, k)
        filter_static : bool
            If True, remove static modes (eigenvalues <= min_eigenvalue)
        min_eigenvalue : float, optional
            Threshold for static mode filtering
        n_modes : int, optional
            Return only first n_modes eigenvalues

        Returns
        -------
        filtered : ndarray or tuple
            Filtered eigenvalues, or (eigenvalues, eigenvectors) if eigenvectors provided
        """
        if min_eigenvalue is None:
            min_eigenvalue = FrequencyDomainSolver.DEFAULT_MIN_EIGENVALUE

        # Sort eigenvalues
        idx_sorted = np.argsort(np.real(eigenvalues))
        eigs_sorted = np.real(eigenvalues[idx_sorted])

        # Filter static modes
        if filter_static:
            mask = eigs_sorted > min_eigenvalue
            idx_sorted = idx_sorted[mask]
            eigs_sorted = eigs_sorted[mask]

        # Limit to n_modes
        if n_modes is not None and len(eigs_sorted) > n_modes:
            idx_sorted = idx_sorted[:n_modes]
            eigs_sorted = eigs_sorted[:n_modes]

        if eigenvectors is not None:
            vecs_filtered = eigenvectors[:, idx_sorted]
            return eigs_sorted, vecs_filtered
        
        return eigs_sorted

    def calculate_resonant_modes(
        self,
        domain: str = None,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None,
        sigma: float = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute eigenvalues and eigenvectors from generalized eigenvalue problem K x = λ M x.

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
        modes : tuple or dict
            Tuple of (eigenvalues, eigenvectors) if domain is specified.
            Dict mapping domain names to (eigenvalues, eigenvectors) if domain is None.
        """
        # Single-domain optimization: if only one domain, 'global' and domain are identical
        if domain is None and self.n_domains == 1:
            res = self.calculate_resonant_modes(domain=self.domains[0], filter_static=filter_static,
                                               min_eigenvalue=min_eigenvalue, n_modes=n_modes, sigma=sigma)
            return {self.domains[0]: res, 'global': res}
        
        if domain == 'global' and self.n_domains == 1:
            domain = self.domains[0]

        # Check cache first
        cache_key = f"{domain}_{filter_static}_{min_eigenvalue}_{n_modes}_{sigma}"
        if domain is not None and cache_key in self._resonant_mode_cache:
            return self._resonant_mode_cache[cache_key]

        # Check if matrices are available
        has_per_domain = bool(self.M)
        has_global = self.M_global is not None

        if not has_per_domain and not has_global:
            raise ValueError("Matrices not assembled. Call assemble_matrices() first.")

        def compute_eigs(M_mat, K_mat, fes, label: str) -> Tuple[np.ndarray, np.ndarray]:
            """Compute eigenvalues and eigenvectors for a single domain/global system."""
            # Get free DOFs (not constrained by Dirichlet BC)
            freedofs = fes.FreeDofs()
            free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]])

            if len(free_idx) == 0:
                print(f"Warning: No free DOFs for {label}")
                return np.array([]), np.array([])

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
                # Default shift: target around 1 GHz
                shift = 1e18

            try:
                # Try sparse eigenvalue solver with shift-invert
                from scipy.sparse.linalg import eigsh

                # Ensure matrices are in CSR format for efficiency
                M_csr = sp.csr_matrix(M_free) if sp.issparse(M_free) else sp.csr_matrix(M_free)
                K_csr = sp.csr_matrix(K_free) if sp.issparse(K_free) else sp.csr_matrix(K_free)

                # Shift-invert mode: solve (K - σM)^{-1} M x = θ x
                eigenvalues, eigenvectors_free = eigsh(
                    K_csr, k=k, M=M_csr,
                    sigma=shift, which='LM',
                    return_eigenvectors=True
                )
                
                # Expand eigenvectors to full DOF count
                eigenvectors = np.zeros((fes.ndof, len(eigenvalues)))
                eigenvectors[free_idx, :] = eigenvectors_free
                
                return eigenvalues, eigenvectors

            except Exception as e1:
                print(f"Note: Sparse eigsh failed for {label}: {e1}")
                print(f"      Falling back to dense solver...")

                try:
                    # Fall back to dense solver
                    M_dense = M_free.toarray() if sp.issparse(M_free) else np.array(M_free)
                    K_dense = K_free.toarray() if sp.issparse(K_free) else np.array(K_free)

                    # Use scipy.linalg.eigh for generalized symmetric eigenvalue problem
                    eigenvalues, eigenvectors_free = sl.eigh(K_dense, M_dense)
                    
                    # Expand eigenvectors
                    eigenvectors = np.zeros((fes.ndof, len(eigenvalues)))
                    eigenvectors[free_idx, :] = eigenvectors_free
                    
                    return eigenvalues, eigenvectors

                except Exception as e2:
                    print(f"Warning: Dense eigh also failed for {label}: {e2}")
                    return np.array([]), np.array([])

        def process_modes(raw_eigs: np.ndarray, raw_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Apply filtering to raw eigenvalues and eigenvectors."""
            if len(raw_eigs) == 0:
                return raw_eigs, raw_vecs
            return self._filter_eigenvalues(raw_eigs, raw_vecs, filter_static, min_eigenvalue, n_modes)

        # Handle specific domain request
        if domain is not None:
            if domain == 'global':
                if not has_global:
                    raise ValueError("Global matrices not assembled. "
                                    "Call assemble_matrices(assemble_global=True)")
                raw_eigs, raw_vecs = compute_eigs(self.M_global, self.K_global, self._fes_global, 'global')
            elif domain in self.M:
                raw_eigs, raw_vecs = compute_eigs(self.M[domain], self.K[domain], self._fes[domain], domain)
            else:
                raise KeyError(f"Domain '{domain}' not found. Available: {list(self.M.keys())}")
            
            res = process_modes(raw_eigs, raw_vecs)
            self._resonant_mode_cache[cache_key] = res
            
            # Sync with standard eigen cache so save_eigenmodes works
            if len(res) == 2:
                self._init_eigen_cache()
                self._eigenvalues_cache[domain] = res[0]
                self._eigenvectors_cache[domain] = res[1]
                try:
                    self.save_eigenmodes(auto_compute=False)
                except Exception:
                    pass

            return res

        # Return all available
        results = {}

        # Per-domain resonant modes
        for d in self.domains:
            if d in self.M and d in self._fes:
                raw_eigs, raw_vecs = compute_eigs(self.M[d], self.K[d], self._fes[d], d)
                results[d] = process_modes(raw_eigs, raw_vecs)

        # Global resonant modes
        if has_global and self._fes_global is not None:
            raw_eigs, raw_vecs = compute_eigs(self.M_global, self.K_global, self._fes_global, 'global')
            results['global'] = process_modes(raw_eigs, raw_vecs)

        # Sync all with standard eigen cache and save
        self._init_eigen_cache()
        for d, res in results.items():
            if len(res) == 2:
                self._eigenvalues_cache[d] = res[0]
                self._eigenvectors_cache[d] = res[1]
        try:
            self.save_eigenmodes(auto_compute=False)
        except Exception:
            pass

        return results

    def get_eigenvalues(self, **kwargs):
        """Deprecated alias for calculate_resonant_modes."""
        import warnings
        warnings.warn("get_eigenvalues() is deprecated. Use calculate_resonant_modes() instead.",
                      DeprecationWarning, stacklevel=2)
        res = self.calculate_resonant_modes(**kwargs)
        if isinstance(res, dict):
            return {k: v[0] for k, v in res.items()}
        return res[0]

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

        modes = self.calculate_resonant_modes(
            domain=domain,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=None,  # Don't limit here, do it after freq conversion
            sigma=sigma
        )

        if isinstance(modes, dict):
            # If dict, prefer global, else combine all
            if 'global' in modes:
                all_eigs = modes['global'][0]
            else:
                all_eigs = np.concatenate([v[0] for v in modes.values()])
        else:
            all_eigs = modes[0]

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
            if domain != 'global' and domain not in self.domains:
                raise KeyError(f"Domain '{domain}' not found. Available: {self.domains}")
            
            if domain == 'global':
                # Already handled above, but for completeness
                return {
                    'M': self.M_global,
                    'K': self.K_global,
                    'B': self.B_global,
                    'W': self.snapshots.get('global'),
                    'fes': self._fes_global,
                    'ports': self._external_ports if self.is_compound else self._ports,
                    'n_ports': len(self._external_ports) if self.is_compound else self._n_ports
                }

            # For specific domain, check if we have per-domain matrices
            M = self.M.get(domain)
            K = self.K.get(domain)
            B = self.B.get(domain)
            W = self.snapshots.get(domain)
            fes = self._fes.get(domain)
            
            # Fallback for single-domain projects
            if not self.is_compound:
                if M is None: M = self.M_global
                if K is None: K = self.K_global
                if B is None: B = self.B_global
                if W is None: W = self.snapshots.get('global')
                if fes is None: fes = self._fes_global

            return {
                'M': M,
                'K': K,
                'B': B,
                'W': W,
                'fes': fes,
                'ports': self.domain_port_map.get(domain, []),
                'n_ports': len(self.domain_port_map.get(domain, []))
            }

        # Return all
        results = {}

        for d in self.domains:
            M = self.M.get(d)
            K = self.K.get(d)
            B = self.B.get(d)
            W = self.snapshots.get(d)
            fes = self._fes.get(d)
            
            if not self.is_compound:
                if M is None: M = self.M_global
                if K is None: K = self.K_global
                if B is None: B = self.B_global
                if W is None: W = self.snapshots.get('global')
                if fes is None: fes = self._fes_global

            results[d] = {
                'M': M,
                'K': K,
                'B': B,
                'W': W,
                'fes': fes,
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
        euler_angles: Optional[List] = [45, -45, 0],
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
            :param euler_angles:
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

        if euler_angles:
            draw_kwargs['euler_angles'] = euler_angles

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
            print(list(self._fes.keys()))
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