from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Optional, Tuple, Union, Literal
    import matplotlib.pyplot as plt
    from ngsolve import Mesh
    from cavsim3d.geometry.base import BaseGeometry
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
import platform
import numpy as np
import scipy.sparse as sp
from cavsim3d.solvers.eigen_mixin import FDSEigenMixin
from cavsim3d.utils.io_utils import deep_diff, strip_keys, check_source_files
from ngsolve import (
    HCurl, BilinearForm, LinearForm, GridFunction, InnerProduct, TaskManager,
    curl, dx, ds, BoundaryFromVolumeCF, CoefficientFunction, Norm, preconditioners,
)
from ngsolve.webgui import Draw
from ngsolve.krylovspace import GMResSolver, CGSolver
from cavsim3d.solvers.results import build_fom_collection
from cavsim3d.solvers.nedelec import check_kind, hcurl_flags
from cavsim3d.core.constants import mu0, eps0, c0, Z0, MIN_EIGENVALUE
from cavsim3d.solvers.base import BaseEMSolver, ParameterConverter
from cavsim3d.solvers.ports import (
    PortEigenmodeSolver, group_port_faces, sorted_logical_ports, logical_port_name
)
from cavsim3d.utils.names import is_port_name, region_pattern
from cavsim3d.solvers.options import (FOM_SOLVE_OPTIONS, check_solve_options,
                                      validate_sweep)
import cavsim3d.utils.printing as pr
from cavsim3d.geometry.base import _display_webgui_fallback

# PARDISO ships with MKL, which the macOS ngsolve wheels do not link, so a
# fallback is needed there. UMFPACK is NOT it: it aborts with "Numeric
# factorization failed" on the near-singular A(w) of an open structure (the
# microstrip qTEM port), where PARDISO survives by perturbing tiny pivots.
# sparsecholesky is NGSolve's built-in LDL^T -- always compiled into the wheel,
# no external dependency -- and factors those systems fine.
_DIRECT_SOLVER = "sparsecholesky" if platform.system() == "Darwin" else "pardiso"


def _available_memory() -> Optional[float]:
    """Free memory in bytes, or None if it cannot be read."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except Exception:
        return None


def direct_memory_estimate(n_free: int, nnz: int, complex_: bool = True,
                           symmetric: bool = False, inverse: str = "pardiso") -> float:
    """Bytes a sparse direct factorisation of A(w) is expected to take at its peak.

    Fitted to the peak memory of 23 PARDISO factorisations of H(curl) systems
    on tetrahedral meshes (orders 1-3, both Nedelec kinds, real and complex,
    2.6k to 714k free unknowns), which it matches within 12 %:
    0.61 GB (n_free / 1e5)^1.41 (nnz / n_free / 50)^0.10, times 1.70 for a
    complex system and 0.66 for symmetric storage.  Structured and hybrid
    meshes fill more (up to 30 % in earlier runs), hence ``_MEMORY_SAFETY``.
    NGSolve's sparse Cholesky peaks at about 0.55 times PARDISO.
    """
    n = max(int(n_free), 1)
    density = max(float(nnz) / n, 1.0)
    gb = 0.609 * (n / 1e5) ** 1.413 * (density / 50.0) ** 0.100
    if complex_:
        gb *= 1.70
    if symmetric:
        gb *= 0.66
    return gb * 1e9 * _INVERSE_MEMORY_FACTOR.get(inverse, 1.0) * _MEMORY_SAFETY


_INVERSE_MEMORY_FACTOR = {"pardiso": 1.0, "sparsecholesky": 0.55}
_MEMORY_SAFETY = 1.25


def _nnz(matrix) -> Optional[int]:
    """Stored entries of a scipy sparse matrix, None without one."""
    return int(matrix.nnz) if matrix is not None and hasattr(matrix, 'nnz') else None


def _real_csr(ngmat) -> sp.csr_matrix:
    """scipy CSR copy of an assembled NGSolve matrix.

    K, M, C and D all have real integrands; on a complex (lossy) space NGSolve
    still stores them as complex, with zero imaginary part.
    """
    m = sp.csr_matrix(ngmat.CSR()).copy()
    if np.iscomplexobj(m.data):
        m = sp.csr_matrix(m.real)
    return m


class FrequencyDomainSolver(BaseEMSolver, FDSEigenMixin):
    """
    Frequency-domain solver for electromagnetic problems.

    Handles both single-domain and compound (multi-domain) structures:

    * ``per_domain=True`` (default for compound structures) solves every
      domain on its own -- the snapshots feed ``fds.foms.reduce()``.
    * ``per_domain=False`` solves the whole mesh as one coupled system
      (``fds.fom``).

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
    >>> fds.solve(1, 10, 100)
    >>> fds.fom.plot_s()

    >>> # Compound structure: whole mesh as one coupled system
    >>> fds.solve(1, 10, 100, per_domain=False)

    >>> # Compound structure: per-domain results (for ROM training)
    >>> fds.solve(1, 10, 100, per_domain=True)
    """

    # --- Solver defaults ---
    # solver_type="auto" factorises when the factorisation fits in this share
    # of the free memory, else it solves iteratively.  Without a memory reading
    # it falls back to the number of unknowns.
    AUTO_MEMORY_SHARE = 0.6
    AUTO_DOF_THRESHOLD = 400_000
    # 'tol' is relative to the right-hand side: the preconditioned residual of
    # the start vector x = 0, whatever start vector the sweep passes.
    DEFAULT_ITERATIVE_OPTS = {
        'method': 'cocg',
        'precond': 'bddc',
        'maxsteps': 500,
        'tol': 1e-8,
        'printrates': False,
    }
    ITERATIVE_METHODS = ('cocg', 'gmres')

    def __init__(
        self,
        geometry,
        order: int = 3,
        bc: Optional[str] = None,
        use_wave_impedance: bool = True,
        nedelec: str = 'first',
    ):
        super().__init__()

        self.geometry = geometry
        self._mesh: Optional[Mesh] = None
        self.order = order
        # Nedelec element kind of every H(curl) space ('first' | 'second'; see
        # solvers.nedelec).  Set before the mesh: the mesh builds the spaces.
        self.nedelec = check_kind(nedelec)
        self.bc = bc if bc is not None else getattr(geometry, 'bc', None)
        self.use_wave_impedance = use_wave_impedance
        
        # Per-domain storage (MUST be initialized before setting mesh)
        self._fes: Dict[str, HCurl] = {}
        self.M: Dict[str, sp.csr_matrix] = {}
        self.K: Dict[str, sp.csr_matrix] = {}
        self.B: Dict[str, np.ndarray] = {}
        # Loss matrices (only for lossy domains):  A(w) = K + jwC - w^2 (M - jD)
        # C = int sigma u.v,  D = int eps0 eps_r tan(delta) u.v
        self.C: Dict[str, sp.csr_matrix] = {}
        self.D: Dict[str, sp.csr_matrix] = {}

        # Global (coupled) storage
        self._fes_global: Optional[HCurl] = None
        self.M_global: Optional[sp.csr_matrix] = None
        self.K_global: Optional[sp.csr_matrix] = None
        self.B_global: Optional[np.ndarray] = None
        self.C_global: Optional[sp.csr_matrix] = None
        self.D_global: Optional[sp.csr_matrix] = None

        # Snapshots storage
        self.snapshots: Dict[str, np.ndarray] = {}

        self._project_path: Optional[str] = None
        
        # Port modes (shared across domains).  mode_source: 'analytic' (closed
        # form for rectangular / circular / coaxial cross-sections) or
        # 'numeric' (2D eigenproblem on the port face) -- external and
        # internal ports separately.  Set through solve(mode_source=...).
        self.port_mode_source: str = 'analytic'
        self.port_mode_source_internal: str = 'analytic'
        self.port_solver: Optional[PortEigenmodeSolver] = None
        self.port_modes: Dict[str, Dict[int, CoefficientFunction]] = None
        self.port_basis: Dict[str, Dict[int, np.ndarray]] = None
        self._n_modes_per_port: int = None
        # Per-port mode specification (int or {port: count}) and the ordered
        # (port_name, mode_idx) list defining the global Z/S matrix columns
        # when mode counts differ between ports.
        self._nportmodes_spec: Union[int, Dict[str, int]] = None
        self._port_mode_order: Optional[List[Tuple[str, int]]] = None

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

        # Track which method was used for current results
        self._current_global_method: Optional[str] = None

        # Assembly state flags
        self._global_matrices_assembled: bool = False
        self._per_domain_matrices_assembled: bool = False

        # Result-object caches (built lazily via .fom / .foms)
        self._fom_cache = None
        self._foms_cache = None
        self._netlist_foms = None   # set by solve() for netlist assemblies

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
        # Sweep checkpoint folder override (see _checkpoint_root)
        self._checkpoint_dir: Optional[Path] = None

        # Reset resonant mode cache
        self._resonant_mode_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Beam excitation (solvers/beam.py): the beams of a solver without a
        # project, the beam outputs of the last solve per system ('global' or
        # a domain), their generalised matrices, and the fingerprint of the
        # beam definition they belong to
        self._beam_setup = None
        self._beam_raw: Dict[str, Dict] = {}
        self._beam_systems: Dict[str, object] = {}
        self._beam_tilde: Dict[str, Dict] = {}
        self._beam_fingerprint: Optional[str] = None

        # Validate boundary conditions
        self._validate_boundary_conditions()

    @property
    def project_sub_path(self) -> Path:
        """Relative path from project root for this solver's data."""
        if self.is_compound:
            return Path("fds") / "foms"
        return Path("fds") / "fom"

    @property
    def mesh(self) -> Optional[Mesh]:
        """NGSolve Mesh object."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Optional[Mesh]):
        """
        Set mesh and automatically (re)detect domains, ports, and (re)construct FES.
        """
        old_mesh = getattr(self, '_mesh', None)
        self._mesh = value

        if value is not None:
            if old_mesh is not None and value is not old_mesh:
                # A different mesh invalidates everything built on the old one:
                # port-mode GridFunctions and K/M/B are indexed by its DOFs.
                self._reset_discretisation()
            # Update detection
            self.domains = self._detect_domains()
            self._ports = self._detect_ports()
            self.n_domains = len(self.domains)
            self._n_ports = len(self._ports)

            # Port↔domain adjacency from the mesh (which domains each port
            # boundary touches).  This is the basis for a general,
            # non-cascade structure classification.
            self._port_domain_adjacency: Dict[str, set] = \
                self._compute_port_domain_adjacency()

            self.domain_port_map = self._build_domain_port_map()

            # Internal ports = port faces shared by two or more domains (or
            # declared internal by the geometry).  A structure is compound
            # when it has multiple domains coupled through such internal
            # ports — this now covers assemblies AND split single geometries,
            # not just linear cascades.
            self._internal_ports: List[str] = self._identify_internal_ports()
            self._external_ports: List[str] = self._identify_external_ports()
            self.is_compound = self.n_domains > 1 and len(self._internal_ports) > 0
            
            # Reconstruct FES for all domains (essential for field plotting after load)
            self._reconstruct_fes()

            # Port solver for this mesh (reused if it is already bound to it)
            if self.port_solver is None or self.port_solver.mesh is not value:
                self.port_solver = self._new_port_solver()
                self.port_modes = None
                self.port_basis = None
            else:
                self._attach_port_media(self.port_solver)

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
            self._reset_discretisation()

    def _reset_discretisation(self) -> None:
        """Drop FE spaces, system matrices and port modes (mesh/order changed)."""
        self._fes = {}
        self.M, self.K, self.B = {}, {}, {}
        self.C, self.D = {}, {}
        self._fes_global = None
        self.M_global = None
        self.K_global = None
        self.B_global = None
        self.C_global = None
        self.D_global = None
        self._global_matrices_assembled = False
        self._per_domain_matrices_assembled = False
        self.port_solver = None
        self.port_modes = None
        self.port_basis = None
        self._n_modes_per_port = None
        self._nportmodes_spec = None

    def _new_port_solver(self) -> PortEigenmodeSolver:
        """Port solver for the current mesh/order/bc, with the port media set.

        Every construction must go through here: a port solver without
        ``port_media_eps`` treats dielectric-filled ports as vacuum.
        """
        ps = PortEigenmodeSolver(self._mesh, self.order, self.bc,
                                 nedelec=self.nedelec,
                                 mode_source=self.port_mode_source,
                                 mode_source_internal=self.port_mode_source_internal)
        self._attach_port_media(ps)
        return ps

    def _attach_port_media(self, ps: PortEigenmodeSolver) -> None:
        """Give the port solver the eps_r / mu_r of the medium at each port face."""
        ps.port_media_eps = self._compute_port_media_eps('eps_r')
        ps.port_media_mu = self._compute_port_media_eps('mu_r')

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
                # A domain may map to several mesh materials (e.g. a split
                # sub-domain 'subdomain1' covers 'subdomain1/vacuum' and
                # 'subdomain1/ceramic').  Build the region from the actual
                # material list so the FES is non-empty for prefixed names.
                mesh_mats = self._get_domain_mesh_materials(domain) or [domain]
                region = self._mesh.Materials(region_pattern(mesh_mats))
                fes = HCurl(
                    self._mesh,
                    order=self.order,
                    **hcurl_flags(self.nedelec),
                    dirichlet=self.bc,
                    definedon=region,
                    complex=self._is_lossy(),
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
        ports = [b for b in boundaries if is_port_name(b)]
        unnamed = [b for b in boundaries if b in ('', None)]
        unique_boundaries = sorted(set(b for b in boundaries if b))
        not_ports = sorted({b for b in boundaries
                            if b and 'port' in b.lower() and not is_port_name(b)})
        if not_ports:
            warnings.warn(
                f"\n  Boundaries {not_ports} contain 'port' but are not ports: a "
                f"port's name starts with 'port' (e.g. 'port1', or 'port1_air' for "
                f"one face of a composite port).",
                UserWarning,
                stacklevel=2,
            )

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
            from cavsim3d.solvers.results import build_fom_result
            self._fom_cache = build_fom_result(self, domain='global')
        return self._fom_cache

    @property
    def foms(self):
        """
        Per-domain FOM results as a :class:`~solvers.results.FOMCollection`.

        For **multi-solid** (compound) structures this is the per-solid FOM
        collection (requires ``solve(per_domain=True)``).  For an assembly
        NETLIST (components with repeat counts and/or imported projects) it is
        the per-component FOM stage — same fluent chain:
        ``fds.foms.reduce(tol).concatenate()``.

        Example
        -------
        >>> fds.foms[0].plot_s()            # first domain
        >>> fds.foms.concat.plot_s()        # concatenated FOM
        >>> fds.foms.roms.concat.rom.plot_s()  # full chain
        """
        if self._netlist_foms is not None:
            return self._netlist_foms
        asm = self._netlist_assembly()
        if asm is not None:
            # a reopened project: the sections it solved are on disk
            if self._restore_netlist_foms(asm) is not None:
                return self._netlist_foms
            raise RuntimeError(
                "This geometry is an assembly netlist: call fds.solve(config=...) "
                "first (it runs/loads each unique component's FOM), then "
                "fds.foms.reduce(tol).concatenate().")
        if self._foms_cache is None:
            self._foms_cache = build_fom_collection(self)
        return self._foms_cache

    def _restore_netlist_foms(self, asm):
        """Rebuild the netlist FOM stage from the project's files, or None.

        None when a part has no saved results (the netlist must be solved).
        """
        if not self._project_path:
            return None
        from cavsim3d.solvers import netlist_persistence as npz
        from cavsim3d.solvers.results import NetlistFOMs
        root = Path(self._project_path)
        saved = npz.read_sections(root)
        imports = npz.read_imports(root)
        components: Dict[str, Dict] = {}
        for key in asm._component_order:
            base = asm._components[key].base_name
            if base in components:
                continue
            if base in saved["sections"] and npz.has_staged_fom(root, base):
                components[base] = dict(saved["sections"][base], kind="live")
            elif base in imports:
                rec = dict(imports[base], kind="imported")
                if rec.get("mode") == "copy" and npz.has_local_copy(root, base):
                    rec["local"] = True
                components[base] = rec
            else:
                return None
        self._netlist_foms = NetlistFOMs(asm, components, self, dict(saved["config"]))
        return self._netlist_foms

    # ------------------------------------------------------------------
    # Assembly netlists (repeat-N sections, imported projects)
    # ------------------------------------------------------------------

    def import_model(self, project_path):
        """Import an ALREADY-RUN project's saved results as a component.

        Returns an :class:`~cavsim3d.core.reuse.ImportedModel` handle that can
        be added to an :class:`Assembly` netlist like any geometry — its saved
        FOM/ROM is loaded (never recomputed) when the netlist is concatenated.

        (Named ``import_model`` because ``import`` is a reserved Python
        keyword.)

        Example
        -------
        >>> hom = proj.fds.import_model("path/to/hom_coupler_project")
        >>> asm.add("hom", hom, after="cavity")

        Deprecated: use ``proj.import_project(path, name=..., mode=...)``, which
        adds the project as a part (``mode='copy'`` matches this method).
        """
        import warnings
        warnings.warn(
            "fds.import_model() is deprecated: use proj.import_project(path, name=..., "
            "mode='copy' | 'reference'), which adds the project as a part.",
            DeprecationWarning, stacklevel=2)
        from cavsim3d.core.reuse import ImportedModel
        return ImportedModel(project_path)

    def _netlist_assembly(self):
        """Return the geometry if its parts are COUPLED (a netlist), else None.

        With the default strategy an assembly is coupled when a part is an
        imported project or is repeated (n > 1); plain geometry parts are glued
        into one multi-solid mesh.  ``Assembly.set_mesh_strategy`` overrides.
        """
        from cavsim3d.geometry.assembly import Assembly
        g = self.geometry
        if not isinstance(g, Assembly):
            return None
        strategy, _ = g.resolved_mesh_strategy()
        return g if strategy == 'coupled' else None

    def _solve_netlist(self, asm, cfg: Dict) -> Dict:
        """FOM stage for a netlist assembly — SINGLE fds, flat per-domain layout.

        Each UNIQUE section becomes a domain inside this project's one
        ``fds/foms`` tree (``matrices/K_<domain>.h5``, ``s/s_<domain>.h5`` …),
        its mesh files in the project's single ``mesh/`` folder
        (``mesh_<domain>.pkl``) and its geometry in ``geometry/components/``
        — exactly like a multi-solid project.  A live section is computed ONCE
        (scratch project, then staged in); an imported one is staged straight
        from its already-run project.  On disk the two are indistinguishable;
        nothing is nested as a sub-project and nothing is recomputed: a live
        section staged by an earlier solve with the same settings and the same
        geometry is reused (``fds/sections.json`` records both).
        """
        if not self._project_path:
            raise RuntimeError(
                "Assembly netlists must be driven from an EMProject "
                "(proj.fds.solve()), so the flat fds/foms tree has a home.")
        project_root = Path(self._project_path)
        _strategy, why = asm.resolved_mesh_strategy()
        pr.milestone(f"Mesh strategy: coupled ({why}): each unique part is solved on "
                f"its own and the parts are joined through their port modes.")

        from cavsim3d.solvers import netlist_persistence as npz
        plan = self._netlist_plan(asm, cfg)
        previous = npz.read_imports(project_root)
        saved = npz.read_sections(project_root)["sections"]
        components: Dict[str, Dict] = {}
        imports: Dict[str, Dict] = {}
        live: Dict[str, Dict] = {}          # sections computed in this project
        for key in asm._component_order:
            entry = asm._components[key]
            base = entry.base_name
            if base in components:
                continue
            comp = entry.geometry
            action = plan[base][1]
            is_import = isinstance(comp, (str, Path)) or hasattr(comp, 'project_path')
            if (action == 'reuse' and base in saved
                    and (not is_import or saved[base].get('derived_from'))):
                # staged by an earlier solve with the same settings: reuse it
                components[base] = live[base] = dict(saved[base], kind="live")
                continue
            if action == 'recompute':
                # The imported part does not fit (or has no results): solve it
                # here from its geometry.  Written into THIS project only.
                from cavsim3d.geometry.base import BaseGeometry
                src = Path(getattr(comp, 'project_path', comp))
                geo = BaseGeometry.load_geometry(src, check_source=False)
                if getattr(geo, 'mesh', None) is None:
                    geo.generate_mesh()
                rec = self._run_section_fom(base, geo, cfg, project_root)
                rec.update(derived_from=str(src), signature=npz.project_signature(src),
                           config=npz.section_config(cfg))
                components[base] = live[base] = rec
                continue
            if action == 'reduce':
                src = Path(getattr(comp, 'project_path', comp))
                mode = getattr(comp, 'mode', 'copy')
                if mode == 'reference':
                    components[base] = {"kind": "imported", "mode": "reference",
                                        "source": str(src), "reduce": True,
                                        "fingerprint": comp.fingerprint()}
                else:
                    components[base] = self._stage_imported_section(base, comp, project_root)
                    components[base]["reduce"] = True
                imports[base] = components[base]
                continue
            if getattr(comp, 'mode', None) == 'reference':
                components[base] = self._reference_imported_section(base, comp)
                imports[base] = components[base]
            elif isinstance(comp, (str, Path)) or hasattr(comp, 'project_path'):
                src = Path(getattr(comp, 'project_path', comp))
                prev = previous.get(base, {})
                same_source = (prev.get("source")
                               and Path(prev["source"]).resolve() == src.resolve())
                if (same_source and npz.has_local_copy(project_root, base)
                        and not (cfg.get('rerun') is True and src.exists())):
                    # A copy is a snapshot: keep using it (the source may be gone).
                    pr.info(f"  netlist section '{base}': using the local copy")
                    components[base] = {"kind": "imported", "mode": "copy",
                                        "source": str(src), "local": True,
                                        "fingerprint": prev.get("fingerprint")}
                else:
                    components[base] = self._stage_imported_section(
                        base, comp, project_root)
                    if hasattr(comp, 'fingerprint'):
                        components[base]["fingerprint"] = comp.fingerprint()
                imports[base] = components[base]
            else:
                rec = self._run_section_fom(base, comp, cfg, project_root)
                rec.update(signature=npz.geometry_signature(comp),
                           config=npz.section_config(cfg))
                components[base] = live[base] = rec
        npz.write_imports(project_root, imports)
        npz.write_sections(project_root, cfg, live)

        from cavsim3d.solvers.results import NetlistFOMs
        self._netlist_foms = NetlistFOMs(asm, components, self, dict(cfg))
        self._persist_netlist_project(cfg, project_root)
        # every section is staged: their sweep checkpoints are spent
        from cavsim3d.solvers.sweep_checkpoint import clear_checkpoints
        clear_checkpoints(self._checkpoint_root())

        pr.milestone(f"Netlist FOM stage complete: {len(components)} unique "
                     f"section(s) for {sum(int(e.metadata.get('n', 1)) for e in asm._components.values())} instance(s)")
        return {"netlist_sections": list(components.keys())}

    def _netlist_plan(self, asm, cfg: Dict) -> Dict[str, Tuple[str, str, str]]:
        """Decide, print and (if needed) gate what each unique part needs.

        ``{base: (kind, action, reason)}`` with action one of ``compute``
        (a geometry part: full-order solve), ``reuse`` (an imported reduced
        model that fits), ``reduce`` (imported full-order results without a
        reduced model) or ``recompute`` (an imported part that does not fit
        the request or has no results: full-order solve from its geometry).
        Imported parts are never written to; what they lack is computed here.
        """
        from cavsim3d.utils.io_utils import is_interactive
        from cavsim3d.solvers import netlist_persistence as npz
        root = Path(self._project_path)
        previous = npz.read_imports(root)
        saved = npz.read_sections(root)["sections"]
        plan: Dict[str, Tuple[str, str, str]] = {}
        for key in asm._component_order:
            entry = asm._components[key]
            base = entry.base_name
            if base in plan:
                continue
            comp = entry.geometry
            if isinstance(comp, (str, Path)) or hasattr(comp, 'project_path'):
                kind = f"imported ({getattr(comp, 'mode', 'copy')})"
                src = Path(getattr(comp, 'project_path', comp))
                rec = saved.get(base) or {}
                same_src = (rec.get('derived_from')
                            and Path(rec['derived_from']).resolve() == src.resolve())
                # computed here earlier from the source's geometry (the source
                # did not fit): reuse it while the source's geometry is the same
                if same_src and self._staged_section_fits(
                        base, rec, cfg, npz.project_signature(src)
                        if src.exists() else rec.get('signature')):
                    plan[base] = (kind, "reuse", "computed here from its geometry earlier")
                    continue
                action, reason = self._plan_imported_section(base, comp, cfg, previous)
                plan[base] = (kind, action, reason)
            elif self._staged_section_fits(base, saved.get(base), cfg,
                                           npz.geometry_signature(comp)):
                plan[base] = ("geometry", "reuse",
                              "its full-order results from an earlier solve")
            else:
                plan[base] = ("geometry", "compute",
                              f"full-order solve, {cfg.get('nsamples')} samples")
        width = max(len(b) for b in plan)
        lines = ["Solve plan:"] + [
            f"  {b:<{width}}  {kind:<20} {action:<9} {reason}"
            for b, (kind, action, reason) in plan.items()]
        pr.milestone("\n".join(lines))
        redo = [b for b, (_k, a, _r) in plan.items() if a == 'recompute']
        if redo and cfg.get('rerun') is not True and not is_interactive():
            raise RuntimeError(
                "\n".join(lines) + f"\nImported part(s) {redo} need a full-order "
                "solve in this project. Pass rerun=True to run it (non-interactive "
                "session).")
        return plan

    def _staged_section_fits(self, base: str, record: Optional[Dict], cfg: Dict,
                             signature: Optional[str]) -> bool:
        """True if a live section staged earlier can be reused for ``cfg``.

        Same geometry (``signature``), same full-order solve settings, its
        files still in the project, and no ``rerun=True``.
        """
        from cavsim3d.solvers import netlist_persistence as npz
        return bool(
            record and cfg.get('rerun') is not True
            and record.get('rom_template') is not None
            and signature is not None and record.get('signature') == signature
            and record.get('config') == npz.section_config(cfg)
            and npz.has_staged_fom(Path(self._project_path), base))

    @staticmethod
    def _fit_problems(band, modes, cfg: Dict) -> List[str]:
        """Why a part trained on ``band`` (GHz) with ``modes`` per port does not
        fit the request in ``cfg`` (empty list: it fits)."""
        from cavsim3d.solvers.ports import resolve_port_mode_counts
        problems = []
        fmin, fmax = cfg.get('fmin'), cfg.get('fmax')
        if band and fmin is not None and fmax is not None:
            tol = 1e-9 * max(1.0, float(fmax))
            if float(fmin) < band[0] - tol or float(fmax) > band[1] + tol:
                problems.append(f"band {float(fmin):g}-{float(fmax):g} GHz is not "
                                f"covered by its {band[0]:g}-{band[1]:g} GHz")
        if cfg.get('nportmodes') is not None and modes:
            try:
                want = resolve_port_mode_counts(cfg['nportmodes'], list(modes))
            except ValueError:
                want = None                      # request names other ports
            short = [f"{p} has {modes[p]} < {want[p]}" for p in modes
                     if want and modes[p] < want[p]]
            if short:
                problems.append("too few port modes (" + ", ".join(short) + ")")
        return problems

    def _plan_imported_section(self, base: str, comp, cfg: Dict,
                               previous: Optional[Dict] = None) -> Tuple[str, str]:
        """``(action, reason)`` for an imported part: reuse / reduce / recompute."""
        import json as _json
        from cavsim3d.core.reuse import ImportedModel
        from cavsim3d.solvers.ports import resolve_port_mode_counts
        from cavsim3d.solvers import netlist_persistence as npz
        src = Path(getattr(comp, 'project_path', comp))

        # A copy is a snapshot: if this project already holds one of the same
        # source, judge the copy (the source may be gone).
        prev = (previous or {}).get(base, {})
        root = Path(self._project_path)
        if (getattr(comp, 'mode', 'copy') == 'copy' and prev.get('source')
                and Path(prev['source']).resolve() == src.resolve()
                and npz.has_local_copy(root, base) and cfg.get('rerun') is not True):
            flat = root / "fds" / "foms" / "roms" / "structures.json"
            entry = next((e for e in (_json.loads(flat.read_text()).get("structures", [])
                                      if flat.exists() else [])
                          if e.get("domain") == base), None)
            if entry is None:
                return 'reuse', "its local copy (reduced at the next reduce)"
            b = entry.get("band")
            problems = self._fit_problems(
                (b['fmin_GHz'], b['fmax_GHz']) if b else None,
                {p: len(m) for p, m in entry.get("port_modes", {}).items()}, cfg)
            if not problems:
                return 'reuse', "its local copy fits the request"

        if not src.exists():
            raise FileNotFoundError(
                f"Imported part '{base}': project not found at {src}. Restore it, or "
                "point the part at its new location "
                "(proj.import_project(new_path, name=...) replaces the part).")
        info = comp if isinstance(comp, ImportedModel) else ImportedModel(src, mode='copy')

        band, modes = None, None
        if info.rom_dir is not None:
            have = 'rom'
            tb = info.training_band
            band = (tb['fmin_GHz'], tb['fmax_GHz']) if tb else None
            modes = {p: len(m) for p, m in info.port_modes.items()}
        else:
            snaps = info.fom_dir / "snapshots" if info.fom_dir is not None else None
            have = ('fom' if snaps is not None and snaps.exists()
                    and any(snaps.iterdir()) else 'geometry')
            conf = src / "fds" / "config.json"
            if have == 'fom' and conf.exists():
                c = _json.loads(conf.read_text())
                if c.get('fmin') is not None and c.get('fmax') is not None:
                    band = (float(c['fmin']), float(c['fmax']))
                spec = c.get('nportmodes_spec', c.get('n_modes_per_port'))
                if c.get('ports') and spec is not None:
                    modes = resolve_port_mode_counts(spec, c['ports'])
        if have == 'geometry':
            if not info.has_geometry:
                raise ValueError(f"Imported part '{base}': {src} has neither results "
                                 "nor a geometry to compute them from.")
            return 'recompute', "no results yet: full-order solve from its geometry"

        problems = self._fit_problems(band, modes, cfg)
        if problems:
            if not info.has_geometry:
                raise ValueError(
                    f"Imported part '{base}' does not fit this request "
                    f"({'; '.join(problems)}) and {src} has no geometry to "
                    "recompute it from.")
            return 'recompute', "; ".join(problems)
        if have == 'rom':
            return 'reuse', "its reduced model fits the request"
        return 'reduce', "full-order results, no reduced model yet"

    @staticmethod
    def _reference_imported_section(base: str, comp) -> Dict:
        """Resolve a REFERENCED imported section: record where it lives and a
        fingerprint of its results; nothing is copied or recomputed."""
        src = Path(comp.project_path)
        if not src.exists():
            raise FileNotFoundError(
                f"Referenced part '{base}': project not found at {src}. Restore it, "
                "or point the part at its new location "
                "(proj.import_project(new_path, name=...) replaces the part).")
        pr.info(f"  netlist section '{base}': referenced from {src}")
        return {"kind": "imported", "mode": "reference", "source": str(src),
                "fingerprint": comp.fingerprint()}

    @staticmethod
    def _stage_imported_section(base: str, comp, project_root: Path) -> Dict:
        """Resolve an IMPORTED section: copy its artifacts into this project's
        flat tree, renamed to the section's index.  Never recomputes."""
        from cavsim3d.solvers import netlist_persistence as npz
        src = Path(getattr(comp, 'project_path', comp))
        if not src.exists():
            raise FileNotFoundError(
                f"Imported section '{base}': project folder not found: {src}. "
                "Restore it or re-import.")
        npz.stage_fom(src, base, project_root)
        pr.info(f"  netlist section '{base}': imported (copied) from {src}")
        return {"kind": "imported", "mode": "copy", "source": str(src)}

    @staticmethod
    def _run_section_fom(base: str, comp, cfg: Dict, project_root: Path) -> Dict:
        """Compute a LIVE section's FOM in a throwaway scratch project, stage its
        files into this project and delete the scratch (also when the solve
        fails).  Returns the section's record: what reducing it later needs
        (see ``netlist_persistence.section_record``)."""
        import tempfile as _tf
        from cavsim3d.core.em_project import EMProject
        from cavsim3d.solvers import netlist_persistence as npz
        work = Path(_tf.mkdtemp(prefix="cavsim3d_section_"))
        try:
            sub = EMProject(name=base, base_dir=str(work), overwrite=True,
                            _announce=False)
            sub.geometry = comp
            # the section's samples live in THIS project, so an interrupted
            # netlist solve resumes the section instead of starting it over
            sub.fds._checkpoint_dir = (Path(project_root) / "fds" / "checkpoint"
                                       / "sections" / base)
            pr.milestone(f"Section '{base}': full-order solve")
            sub.fds.solve(config=dict(cfg))
            sub.save()
            npz.stage_fom(work / base, base, project_root)
            record = npz.section_record(sub.fds)
        finally:
            npz.remove_scratch(work)
        record["kind"] = "live"
        return record

    def _persist_netlist_project(self, cfg: Dict, project_root: Path) -> None:
        """Persist the module project like any other: fds/config.json,
        geometry/ (assembly netlist), project.json, timing.json."""
        import json as _json
        from cavsim3d.solvers import netlist_persistence as npz
        with open(project_root / "fds" / "config.json", "w") as fh:
            _json.dump(npz._jsonable(cfg), fh, indent=2)
        if self._project_ref is not None:
            try:
                self._project_ref.save()
            except Exception as e:
                pr.warning(f"Could not fully save module project: {e}")

    @property
    def fes(self):
        """Convenience access to the global Finite Element Space."""
        return self._fes_global

    @property
    def beam_setup(self):
        """The beams of the next solve, a :class:`~cavsim3d.solvers.beam.BeamSetup`
        with at least one beam, or None.

        A project's beams (``proj.add_beam``) are read from the project; a
        solver without a project uses the setup assigned here.
        """
        ref = getattr(self, '_project_ref', None)
        setup = (getattr(ref, 'beam_setup', None) if ref is not None
                 else getattr(self, '_beam_setup', None))
        return setup if setup else None

    @beam_setup.setter
    def beam_setup(self, setup) -> None:
        self._beam_setup = setup

    #: curve order below which a beam on curved walls is warned about
    BEAM_MIN_CURVE_ORDER = 4

    def _warn_beam_curving(self) -> None:
        """Warn (once per mesh) if a beam is solved on curved walls that the mesh
        follows only to curve order < 4.

        E_free is large and nearly normal to the walls of a pipe around the
        beam; the facets of a curved mesh tilt it into a tangential datum of
        their own, which the beam impedance is sensitive to (a few per cent
        at curve order 2-3, below 0.1 % at 4 in the convergence study).
        """
        if getattr(self, '_beam_curving_checked', None) is self._mesh:
            return
        self._beam_curving_checked = self._mesh
        order = getattr(self.geometry, 'curve_order', None)
        if order is None or order >= self.BEAM_MIN_CURVE_ORDER:
            return
        try:
            curved = bool(np.any(self._mesh.ngmesh.Elements2D().NumPy()['curved']))
        except Exception:
            return
        if curved:
            pr.warning(
                f"The beam runs past curved walls that the mesh follows to curve order "
                f"{order}: the beam impedance is sensitive to the wall's facets (errors of "
                f"a few per cent). Mesh with generate_mesh(curve_order="
                f"{self.BEAM_MIN_CURVE_ORDER}) for beam results.")

    def _make_beam_system(self, key: str, fes, B: np.ndarray, excitation_keys,
                          region_materials: Optional[List[str]] = None):
        """Beam data of one system (None without a beam).  ``excitation_keys``
        is the system's ``(index, port, mode)`` list, the columns of ``B``."""
        setup = self.beam_setup
        if setup is None:
            return None
        from cavsim3d.solvers.beam import BeamSystem
        columns: Dict[str, List[int]] = {}
        for col, (_pm, port, _m) in enumerate(excitation_keys):
            columns.setdefault(port, []).append(col)
        materials = {m: self._material_props(m) for m in set(self.mesh.GetMaterials())}
        self._warn_beam_curving()
        t0 = time.time()
        system = BeamSystem(self, key, fes, B, columns, setup, materials, region_materials)
        crossed = [f.port for f in system.faces if any(f.crossed)]
        pr.info(f"  Beam data ({key}): {system.n_sources} beam(s), {system.n_paths} path(s), "
                f"crossing {crossed or 'no port face'} ({time.time() - t0:.1f} s)")
        self._beam_systems[key] = system
        return system

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

    def _geometry_internal_ports(self) -> List[str]:
        """Internal/interface ports declared by the geometry, if any.

        ``OCCImporter`` exposes ``internal_ports`` (split-plane ports) and
        ``Assembly`` exposes ``get_interface_ports()``.  Only names that are
        actually present as mesh ports are returned.
        """
        geo = self.geometry
        names: List[str] = []
        if hasattr(geo, 'internal_ports'):
            try:
                names = list(geo.internal_ports)
            except Exception:
                names = []
        if not names and hasattr(geo, 'get_interface_ports'):
            try:
                names = list(geo.get_interface_ports())
            except Exception:
                names = []
        return [p for p in names if p in self._ports]

    def _identify_internal_ports(self) -> List[str]:
        """Internal ports: shared by ≥2 domains, or declared internal by geometry.

        Uses mesh-derived port↔domain adjacency, so it works for arbitrary
        multiport topologies (not just linear cascades).
        """
        adj = getattr(self, '_port_domain_adjacency', {}) or {}
        declared = set(self._geometry_internal_ports())
        internal = [
            p for p in self._ports
            if len(adj.get(p, set())) >= 2 or p in declared
        ]
        return internal

    def _identify_external_ports(self) -> List[str]:
        """External ports: every port that is not internal."""
        internal = set(self._identify_internal_ports())
        return [p for p in self._ports if p not in internal]

    def _material_to_domain(self) -> Dict[str, str]:
        """Map each mesh material to its owning domain.

        Uses the geometry's ``_domain_materials`` mapping (assemblies and
        split single geometries declare it); otherwise falls back to a
        ``"domain/material"`` prefix convention, and finally to treating the
        material itself as the domain.
        """
        dm = getattr(self.geometry, '_domain_materials', None)
        if dm:
            return {m: d for d, mats in dm.items() for m in mats}
        return {}

    def _compute_port_media_eps(self, key: str = 'eps_r') -> Dict[str, float]:
        """Relative permittivity of the medium filling each port.

        For every port boundary, finds the adjacent volume material (via the
        netgen face descriptors) and queries the geometry for its ``eps_r``,
        so a dielectric-filled coupler gets the correct medium wave impedance.
        Defaults to vacuum (1.0) when material data is unavailable.
        """
        eps: Dict[str, float] = {}
        if self._mesh is None:
            return eps
        get_material = getattr(self.geometry, 'get_material', None)
        if get_material is None:
            return eps
        try:
            ng = self._mesh.ngmesh
            boundaries = list(self._mesh.GetBoundaries())
            materials = list(self._mesh.GetMaterials())
            nfd = ng.GetNFaceDescriptors()
        except Exception:
            return eps

        def eps_of(vol_idx: int) -> Optional[float]:
            if vol_idx < 1 or vol_idx > len(materials):
                return None
            try:
                return float(get_material(materials[vol_idx - 1]).get(key, 1.0))
            except Exception:
                return None

        for fdi in range(1, nfd + 1):
            fd = ng.FaceDescriptor(fdi)
            bc_idx = fd.bc - 1
            if bc_idx < 0 or bc_idx >= len(boundaries):
                continue
            name = boundaries[bc_idx]
            if not is_port_name(name):
                continue
            # An external port has one adjacent volume (the other side is 0);
            # an internal one has two -- take the larger permittivity.
            for vol_idx in (fd.domin, fd.domout):
                er = eps_of(vol_idx)
                if er is not None:
                    eps[name] = max(eps.get(name, 1.0), er) if name in eps else er
        return eps

    def _has_qtem_ports(self) -> bool:
        """True if any solved port mode is quasi-TEM (frequency-dependent)."""
        ps = self.port_solver
        if ps is None:
            return False
        return any(t == 'qTEM'
                   for d in getattr(ps, 'port_mode_types', {}).values()
                   for t in d.values())

    def _build_qtem_solve_kwargs(self) -> Dict:
        """Assemble the quasi-TEM keyword arguments for the port solver.

        A logical port is treated as quasi-TEM when the user lists it in the
        ``qtem_ports`` solve option OR when its faces span more than one
        permittivity (an inhomogeneous cross-section such as microstrip
        substrate + air).  For each such port this builds:

        - ``port_eps_bnd``: a boundary ``CoefficientFunction`` giving each
          subface's ``eps_r`` (from the adjacent volume material),
        - ``port_conductor_bbnd``: the PEC conductor edge-region string used as
          the port solver's ``dirichlet_bbnd`` (from the solve config or the
          geometry's ``qtem_conductor_bbnd``),
        - ``port_voltage_path``: the ground->strip integration path for the
          power-voltage line impedance (from config or ``geometry.qtem_voltage_path``),
        - ``k0_ref``: the reference wavenumber at the top of the sweep band.
        """
        if self.mesh is None:
            return {}
        face_eps = self._compute_port_media_eps()   # {face_name: eps_r}
        region_map = getattr(self, '_port_face_region', {}) or {}

        requested = set(getattr(self, '_qtem_ports', None) or [])
        cond_cfg = dict(getattr(self, '_qtem_conductor_bbnd', None) or {}) \
            if isinstance(getattr(self, '_qtem_conductor_bbnd', None), dict) else {}
        cond_default = (getattr(self, '_qtem_conductor_bbnd', None)
                        if isinstance(getattr(self, '_qtem_conductor_bbnd', None), str) else None)
        if cond_default is None:
            cond_default = getattr(self.geometry, 'qtem_conductor_bbnd', None)
        vpath_cfg = dict(getattr(self, '_qtem_voltage_path', None) or {})

        qtem_ports: List[str] = []
        port_eps_bnd: Dict[str, object] = {}
        port_conductor_bbnd: Dict[str, str] = {}
        port_voltage_path: Dict[str, Tuple] = {}
        port_eps_max: Dict[str, float] = {}

        for lp, region in region_map.items():
            faces = region.split('|')
            eps_here = {f: float(face_eps.get(f, 1.0)) for f in faces}
            inhomogeneous = len(set(round(v, 9) for v in eps_here.values())) > 1
            if lp not in requested and not inhomogeneous:
                continue
            qtem_ports.append(lp)
            port_eps_bnd[lp] = self.mesh.BoundaryCF(eps_here, default=1.0)
            port_eps_max[lp] = max(list(eps_here.values()) + [1.0])
            bbnd = cond_cfg.get(lp, cond_default)
            if bbnd:
                port_conductor_bbnd[lp] = bbnd
            vpath = vpath_cfg.get(lp)
            if vpath is None and hasattr(self.geometry, 'qtem_voltage_path'):
                try:
                    vpath = self.geometry.qtem_voltage_path(lp)
                except Exception:
                    vpath = None
            if vpath is not None:
                port_voltage_path[lp] = vpath

        if not qtem_ports:
            return {}

        freqs = self.frequencies if self.frequencies is not None else None
        fmax = float(np.max(freqs)) if freqs is not None and len(freqs) else None
        k0_ref = (2 * np.pi * fmax / c0) if fmax else None

        return {
            'qtem_ports': qtem_ports,
            'port_eps_bnd': port_eps_bnd,
            'port_conductor_bbnd': port_conductor_bbnd,
            'k0_ref': k0_ref,
            'port_voltage_path': port_voltage_path,
            'port_eps_max': port_eps_max,
        }

    def _compute_port_domain_adjacency(self) -> Dict[str, set]:
        """Compute, for each port boundary, the set of domains it touches.

        Reads the netgen face descriptors (``domin``/``domout`` give the
        volume domains on either side of every surface) and maps the adjacent
        volume materials to their domains.  A port touching ≥2 domains is an
        internal/interface port; one touching a single domain is external.
        """
        from collections import defaultdict

        adj: Dict[str, set] = defaultdict(set)
        if self._mesh is None:
            return {}
        try:
            ng = self._mesh.ngmesh
            boundaries = list(self._mesh.GetBoundaries())
            materials = list(self._mesh.GetMaterials())
            nfd = ng.GetNFaceDescriptors()
        except Exception:
            return {}

        m2d = self._material_to_domain()

        def domain_of(vol_idx: int) -> Optional[str]:
            if vol_idx < 1 or vol_idx > len(materials):
                return None
            mat = materials[vol_idx - 1]
            if mat in m2d:
                return m2d[mat]
            return mat.split('/', 1)[0] if '/' in mat else mat

        for fdi in range(1, nfd + 1):
            fd = ng.FaceDescriptor(fdi)
            bc_idx = fd.bc - 1
            if bc_idx < 0 or bc_idx >= len(boundaries):
                continue
            name = boundaries[bc_idx]
            if not is_port_name(name):
                continue
            lp = logical_port_name(name)  # collapse composite subfaces
            for vol_idx in (fd.domin, fd.domout):
                d = domain_of(vol_idx)
                if d is not None:
                    adj[lp].add(d)

        return dict(adj)

    def _port_wave_impedance(self, port, mode: int, freq: float):
        """The wave impedance the FOM normalises its port modes to.

        Used only to rescale Z when the *reported* reference differs (TEM ports
        report the line impedance, matching CST).
        """
        if not (self.use_wave_impedance and self.port_modes is not None):
            return None
        ps = self.port_solver
        saved = getattr(ps, 'impedance_reference', 'line')
        try:
            ps.impedance_reference = 'wave'
            return ps.get_port_wave_impedance(port, mode, freq)
        except Exception:
            return None
        finally:
            ps.impedance_reference = saved

    #: Reference-impedance convention of saved Z/S. 2: only the TEM mode of a
    #: coaxial port is referred to the line impedance, its TE/TM modes to
    #: their own wave impedance (1 referred every mode of a coax port to the
    #: line impedance).
    Z_REFERENCE_VERSION = 2

    def _legacy_line_referred_modes(self) -> Dict[Tuple[str, int], complex]:
        """TE/TM port modes that version-1 results referred to a coax line
        impedance: ``{(port, mode): line impedance}``."""
        ps = self.port_solver
        if (ps is None or not self.use_wave_impedance
                or getattr(self, 'impedance_reference', 'line') != 'line'):
            return {}
        out = {}
        for port, types in (getattr(ps, 'port_mode_types', {}) or {}).items():
            tem = [m for m, t in types.items() if t == 'TEM']
            zl = ps.get_port_line_impedance(port, tem[0]) if tem else None
            if zl is None:
                continue
            out.update({(str(port), int(m)): zl for m, t in types.items() if t != 'TEM'})
        return out

    def _upgrade_saved_reference(self) -> None:
        """Re-refer version-1 results to the current convention.

        Version 1 scaled the Z of a coax port's TE/TM modes by
        |Z_line| / |Z_wave(f0)| and computed S against Z_line.  Undoing that
        factor recovers the wave-normalised Z exactly, so the single-part
        result is corrected here without a re-solve.
        """
        legacy = self._legacy_line_referred_modes()
        fom = self._fom_cache
        Z = getattr(fom, '_Z_matrix', None)
        order = self._port_mode_order
        if legacy and Z is not None and order and len(order) == Z.shape[1]:
            f = np.asarray(fom.frequencies)
            r = np.array([abs(legacy[(p, m)]) / abs(self._port_wave_impedance(p, m, f[0]))
                          if (p, m) in legacy else 1.0 for p, m in order])
            Z = Z / np.sqrt(np.outer(r, r))[None, :, :]
            S = np.array([ParameterConverter.z_to_s(
                Z[k], np.diag([self._get_port_impedance(p, m, fk) for p, m in order]))
                for k, fk in enumerate(f)])
            fom._Z_matrix, fom._S_matrix = Z, S
            fom._Z_dict = fom._S_dict = None
            self._Z_global_coupled = self._Z_matrix = Z
            self._S_global_coupled = self._S_matrix = S
            self._invalidate_cache()
            pr.info(f"Saved S/Z re-referred: the TE/TM modes {sorted(legacy)} of coaxial "
                    f"ports now use their own wave impedance, not the TEM line impedance.")
        if legacy and (self._Z_per_domain or self._foms_cache is not None):
            warnings.warn(
                f"The saved per-domain S/Z refer the TE/TM modes {sorted(legacy)} of "
                f"coaxial ports to the TEM line impedance. Solve again with rerun=True "
                f"to refer them to their own wave impedance.", UserWarning, stacklevel=3)
            return
        self._z_reference = self.Z_REFERENCE_VERSION

    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        """Get port wave impedance."""
        if self.use_wave_impedance and self.port_modes is not None:
            # propagate the reference choice to the port solver
            self.port_solver.impedance_reference = getattr(
                self, 'impedance_reference', 'line')
            return self.port_solver.get_port_reference_impedance(port, mode, freq)
        return Z0

    # === Structure detection ===

    def _detect_domains(self) -> List[str]:
        """Detect domains from geometry structure.

        For assemblies, returns the component names (each component is one
        domain, even if it contains multiple mesh materials).  For single
        geometries, returns the deduplicated mesh material list.
        """
        # Assemblies define their own domain structure
        from cavsim3d.geometry.assembly import Assembly
        if isinstance(self.geometry, Assembly):
            return list(self.geometry.keys)

        # Other geometries (e.g. a split OCCImporter) may declare their
        # sub-domain structure explicitly via _domain_materials.
        dm = getattr(self.geometry, '_domain_materials', None)
        if dm:
            return list(dm.keys())

        if self.mesh is None:
            return []
        materials = list(self.mesh.GetMaterials())
        if not materials:
            return ['default']

        # Deduplicate while preserving first-occurrence order
        seen = set()
        unique = []
        for m in materials:
            if m not in seen:
                seen.add(m)
                unique.append(m)

        # All named (non-default) materials are domains: STEP-label names like
        # 'ceramic' or 'beampipe' as well as a split model's 'cell_1', 'cell_2'.
        # Only when every one follows that cell_<N> convention are they put
        # in numeric order (cell_2 before cell_10); a material is never dropped.
        named = [m for m in unique if m.lower() != 'default']
        cells = [re.fullmatch(r'cell_?(\d+)', m, re.IGNORECASE) for m in named]
        if named and all(cells):
            return [c.string for c in sorted(cells, key=lambda c: int(c.group(1)))]
        if named:
            return named

        # Fallback to the first available material (likely 'default' or a single custom name)
        return [unique[0]]

    def _get_domain_mesh_materials(self, domain: str) -> List[str]:
        """Return mesh material names belonging to a domain.

        For assemblies a single domain (component) may contain several mesh
        materials (e.g. ``cell1/ceramic_1``, ``cell1/solid1``).  For single
        geometries, the domain name *is* the mesh material.
        """
        dm = getattr(self.geometry, '_domain_materials', None)
        if dm and domain in dm:
            return dm[domain]
        return [domain]

    def _detect_ports(self) -> List[str]:
        """Detect logical ports from mesh boundaries.

        Faces sharing a leading ``port<N>`` token (e.g. an inhomogeneous
        quasi-TEM microstrip port split into ``port1_substrate`` /
        ``port1_air``) collapse into a single logical port.  The face-region
        map used to resolve a logical port back to its mesh faces is stored on
        ``self._port_face_region``.
        """
        if self.mesh is None:
            self._port_face_region = {}
            return []
        self._port_face_region = group_port_faces(self.mesh.GetBoundaries())
        return sorted_logical_ports(self._port_face_region)

    def _region(self, port: str) -> str:
        """NGSolve region pattern of a logical port: its faces, escaped."""
        raw = getattr(self, '_port_face_region', {}).get(port, port)
        return region_pattern(raw.split('|'))

    def _build_domain_port_map(self) -> Dict[str, List[str]]:
        """Map each domain to the ports that touch it.

        Built from mesh-derived port↔domain adjacency, so a domain is mapped
        to *all* of its ports — external ports plus every shared interface
        port — regardless of how many it has.  This replaces the old linear
        ``n_ports == n_domains + 1`` cascade assumption and supports
        multiport domains (e.g. cavities with coaxial couplers).
        """
        if len(self.domains) == 1:
            return {self.domains[0]: list(self._ports)}

        adj = getattr(self, '_port_domain_adjacency', {}) or {}
        mapping: Dict[str, List[str]] = {d: [] for d in self.domains}
        for port in self._ports:
            for domain in adj.get(port, set()):
                if domain in mapping:
                    mapping[domain].append(port)

        # Fallback for any domain the adjacency missed (e.g. degenerate
        # meshes): assign by sequential position so the map is never empty.
        unmapped = [d for d in self.domains if not mapping[d]]
        if unmapped and not adj:
            for i, domain in enumerate(self.domains):
                ports = []
                if i < len(self._ports):
                    ports.append(self._ports[i])
                if i + 1 < len(self._ports):
                    ports.append(self._ports[i + 1])
                mapping[domain] = ports

        return mapping

    def _print_structure_info(self) -> None:
        """Print detected structure information."""
        print("\n" + "=" * 60)
        print("Structure Topology")
        print("=" * 60)
        stype = 'Compound' if self.is_compound else 'Single'
        print(f"  Type: {stype} structure")
        print(f"  Domains ({self.n_domains}): {self.domains}")
        print(f"  Ports ({self._n_ports}): {self._ports}")

        # Mesh statistics
        if self.mesh is not None:
            ne = self.mesh.ne  # number of volume elements (tetrahedra)
            print(f"  Mesh elements: {ne}")

        if self.is_compound:
            print(f"  External Ports ({len(self._external_ports)}): {self._external_ports}")
            print(f"  Internal Ports ({len(self._internal_ports)}): {self._internal_ports}")
            print("\n  Domain-Port Mapping:")
            for domain, ports in self.domain_port_map.items():
                port_types = []
                for p in ports:
                    if p in self._external_ports:
                        if p == self._ports[0]:
                            port_types.append(f"{p} (input)")
                        else:
                            port_types.append(f"{p} (output)")
                    else:
                        port_types.append(f"{p} (internal)")
                print(f"    {domain}: {port_types}")
        print("=" * 60)

    # === Matrix assembly ===

    def assemble_matrices(
        self,
        nportmodes: Union[int, List[int], Dict[str, int]] = 1,
        assemble_global: bool = True,
        assemble_per_domain: bool = True
    ) -> Dict[str, Tuple]:
        """
        Assemble frequency-independent system matrices.

        Parameters
        ----------
        nportmodes : int, list or dict
            Number of modes to compute per port, letting TEM ports use one mode
            while TE/TM ports use several (the way CST assigns them).

            * ``int``  -- the same count on every port.
            * ``list`` -- positional, in the order given by :meth:`port_map`;
              its length must equal the number of ports.
            * ``dict`` -- ``{'port1': 3, 'port2': 1}``; ports left out fall back
              to a ``'default'`` key, else 1. Unknown names raise.

            Call :meth:`print_port_map` first to see the port order and each
            port's geometry.
        assemble_global : bool
            Assemble global (full-structure) matrices for coupled solve.
            Required for global_method='coupled'.
        assemble_per_domain : bool
            Assemble per-domain matrices. Required for per_domain=True.

        Returns
        -------
        dict
            Dictionary summarizing assembled matrices
        """
        pr.running("\n" + "=" * 60)
        pr.running("Assembling Matrices...")
        pr.running("=" * 60)

        # For single (non-compound) structures, only global matrices are needed
        if not self.is_compound:
            assemble_global = True
            assemble_per_domain = False

        # Solve port eigenmodes if missing or if the solver is blank
        needs_port_solve = (self.port_modes is None)
        if not needs_port_solve and self.port_solver is not None:
             # Check if the solver actually has the data for the modes we need
             if not self.port_solver.port_cutoff_kc:
                 needs_port_solve = True

        if needs_port_solve:
            if self.port_solver is None:
                self.port_solver = self._new_port_solver()
            # New port modes -> every B built from the old ones is stale.
            self._per_domain_matrices_assembled = False
            self._global_matrices_assembled = False

            pr.running("Solving port eigenmodes...")
            qtem_kwargs = self._build_qtem_solve_kwargs()
            self.port_modes, self.port_basis = self.port_solver.solve(
                nmodes=nportmodes,
                internal_ports=self._internal_ports if self.is_compound else [],
                **qtem_kwargs,
            )
            # Record the per-port mode specification.  Keep the scalar
            # _n_modes_per_port for the uniform case (back-compat); for a
            # per-port dict store the spec and derive a scalar fallback.
            self._nportmodes_spec = nportmodes
            if isinstance(nportmodes, (list, tuple)):
                counts = [int(n) for n in nportmodes]
                self._n_modes_per_port = max(counts) if counts else 1
            elif isinstance(nportmodes, dict):
                counts = [len(m) for m in self.port_modes.values()] or [1]
                self._n_modes_per_port = max(counts)
            else:
                self._n_modes_per_port = nportmodes
            # Record the fmax the (qTEM) port modes were solved at, so a later
            # solve over a different band re-solves them (see _ensure_matrices_assembled).
            self._port_modes_fmax = (float(np.max(self.frequencies))
                                     if self.frequencies is not None
                                     and len(self.frequencies) else None)

        # Assemble per-domain matrices if requested
        if assemble_per_domain and not self._per_domain_matrices_assembled:
            self._assemble_per_domain_matrices()
            self._per_domain_matrices_assembled = True

        # Assemble global matrices if requested
        if assemble_global and not self._global_matrices_assembled:
            self._assemble_global_matrices()
            self._global_matrices_assembled = True

        self._persist()
        return self._get_matrix_summary()

    def _get_domain_material(self, domain: str) -> dict:
        """Get material properties for a domain from the geometry.

        Falls back to vacuum (eps_r=1, mu_r=1) if no materials are set.
        Always guarantees 'eps_r' and 'mu_r' keys in the returned dict.
        """
        defaults = {"eps_r": 1.0, "mu_r": 1.0, "sigma": 0.0, "tan_delta": 0.0}
        if hasattr(self.geometry, 'get_material'):
            mat = self.geometry.get_material(domain)
            # Ensure eps_r and mu_r always present (guard against PEC or sparse dicts)
            if "eps_r" not in mat:
                mat["eps_r"] = defaults["eps_r"]
            if "mu_r" not in mat:
                mat["mu_r"] = defaults["mu_r"]
            return mat
        return defaults

    def _material_props(self, name: str) -> Tuple[float, float, float, float]:
        """``(eps_r, mu_r, sigma [S/m], tan_delta)`` of one mesh material.

        The complex permittivity is eps0*eps_r*(1 - j tan_delta) - j sigma/w
        (e^{+jwt}); both loss terms must be >= 0 (passive material).
        """
        from cavsim3d.geometry.base import MATERIAL_DEFAULTS, validate_material_properties
        mat = self._get_domain_material(name)
        if not hasattr(mat, 'get'):
            raise ValueError(f"Material '{name}': expected a dict of properties, got {mat!r}.")
        props = {k: mat.get(k, default) for k, default in MATERIAL_DEFAULTS.items()}
        validate_material_properties(name, props)
        return (float(props['eps_r']), float(props['mu_r']),
                float(props['sigma']), float(props['tan_delta']))

    def _material_signature(self):
        """Hashable fingerprint of everything material-related in the system."""
        if self._mesh is None:
            return None
        try:
            names = sorted(set(self._mesh.GetMaterials()))
            return tuple((n, self._material_props(n)) for n in names)
        except Exception:
            return None

    def _is_lossy(self) -> bool:
        """True if any material is lossy (sigma > 0 or tan_delta > 0).

        A lossy system is complex, so its FE spaces and solutions are complex.
        """
        if self._mesh is None:
            return False
        try:
            names = set(self._mesh.GetMaterials())
        except Exception:
            return False
        for name in names:
            try:
                _e, _m, sigma, tand = self._material_props(name)
            except ValueError:
                raise
            except Exception:
                continue
            if sigma > 0 or tand > 0:
                return True
        return False

    def _build_loss_cfs(self):
        """``(sigma_cf, eps_tand_cf)`` over the mesh materials.

        ``eps_tand_cf`` is eps_r * tan_delta (the imaginary part of eps_r).
        Each is None when that loss is zero everywhere: a zero coefficient
        would drop the trial/test functions from the form (and assemble an
        empty matrix).
        """
        sig, epst = [], []
        for name in self.mesh.GetMaterials():
            eps_r, _mu, sigma, tand = self._material_props(name)
            sig.append(sigma)
            epst.append(eps_r * tand)
        return (CoefficientFunction(sig) if any(sig) else None,
                CoefficientFunction(epst) if any(epst) else None)

    def _build_material_cfs(self):
        """Build CoefficientFunctions for eps_r and mu_r from mesh materials.

        Returns a (eps_r_cf, mu_r_cf) pair that can be used directly in
        bilinear forms over the full mesh.  Each mesh material name is looked
        up via ``_get_domain_material`` so PEC-subtracted geometries (where
        only non-PEC domains remain) work transparently.
        """
        mat_names = list(self.mesh.GetMaterials())
        eps_vals = []
        mu_vals = []
        for name in mat_names:
            eps_r, mu_r, _sigma, _tand = self._material_props(name)
            eps_vals.append(eps_r)
            mu_vals.append(mu_r)
        pr.debug(f"  Material eps_r per mesh material: {eps_vals}")
        eps_r_cf = CoefficientFunction(eps_vals)
        mu_r_cf = CoefficientFunction(mu_vals)

        if any(v != 1.0 for v in eps_vals) or any(v != 1.0 for v in mu_vals):
            pr.debug(f"  Material CFs: eps_r={eps_vals}, mu_r={mu_vals}")

        return eps_r_cf, mu_r_cf

    def draw_material_cf(self, which='eps'):

        eps_r_cf, mu_r_cf = self._build_material_cfs()
        if which == 'eps':
            _display_webgui_fallback(Draw(BoundaryFromVolumeCF(eps_r_cf), self.mesh))
        elif which == 'mu':
            _display_webgui_fallback(Draw(BoundaryFromVolumeCF(mu_r_cf), self.mesh))
    def _assemble_per_domain_matrices(self) -> None:
        """Assemble matrices for each domain independently.

        A domain may span multiple mesh materials (e.g. assembly components
        with ceramics and vacuum regions).  In that case the FES and bilinear
        forms are defined on the union of those materials, and each material
        contributes its own eps_r / mu_r.
        """
        pr.debug("\n--- Assembling Per-Domain Matrices ---")

        for domain in self.domains:
            pr.debug(f"\nDomain: {domain}")

            mesh_mats = self._get_domain_mesh_materials(domain)

            # Build definedon region (union of all mesh materials in this domain)
            region = self.mesh.Materials(region_pattern(mesh_mats))

            # Create FES for this domain (complex when anything is lossy)
            lossy = self._is_lossy()
            fes = HCurl(
                self.mesh,
                order=self.order,
                **hcurl_flags(self.nedelec),
                dirichlet=self.bc,
                definedon=region,
                complex=lossy,
            )
            self._fes[domain] = fes

            u, v = fes.TnT()

            # Stiffness, mass and loss matrices with per-material properties
            k_form = BilinearForm(fes)
            m_form = BilinearForm(fes)
            c_form = BilinearForm(fes)
            d_form = BilinearForm(fes)
            has_c = has_d = False
            for mm in mesh_mats:
                eps_r, mu_r, sigma, tand = self._material_props(mm)
                if eps_r != 1.0 or mu_r != 1.0 or sigma or tand:
                    pr.debug(f"  {mm}: eps_r={eps_r}, mu_r={mu_r}, "
                             f"sigma={sigma}, tan_delta={tand}")
                k_form += (1 / (mu0 * mu_r)) * curl(u) * curl(v) * dx(region_pattern([mm]))
                m_form += eps0 * eps_r * u * v * dx(region_pattern([mm]))
                if sigma:
                    c_form += sigma * u * v * dx(region_pattern([mm]))
                    has_c = True
                if tand:
                    d_form += eps0 * eps_r * tand * u * v * dx(region_pattern([mm]))
                    has_d = True

            with TaskManager():
                k_form.Assemble()
                m_form.Assemble()
                if has_c:
                    c_form.Assemble()
                if has_d:
                    d_form.Assemble()

            self.K[domain] = _real_csr(k_form.mat)
            self.M[domain] = _real_csr(m_form.mat)
            self.C.pop(domain, None)
            self.D.pop(domain, None)
            if has_c:
                self.C[domain] = _real_csr(c_form.mat)
            if has_d:
                self.D[domain] = _real_csr(d_form.mat)
            self._assembled_signature = self._material_signature()

            # Port basis matrix for this domain
            self._construct_domain_basis_matrix(domain, fes)

            pr.debug(f"  FES ndof: {fes.ndof}")
            pr.debug(f"  K shape: {self.K[domain].shape}, nnz: {self.K[domain].nnz}")
            pr.debug(f"  M shape: {self.M[domain].shape}, nnz: {self.M[domain].nnz}")
            pr.debug(f"  B shape: {self.B[domain].shape}")

    def _assemble_global_matrices(self) -> None:
        """Assemble matrices for the full (coupled) structure.

        If per-domain material properties are set, the bilinear forms are
        assembled as a sum of domain-specific contributions so that each
        domain can have its own eps_r / mu_r.
        """
        pr.debug("\n--- Assembling Global Matrices (Coupled System) ---")

        # Create FES for entire mesh.  Losses make the system complex, so the
        # space must be complex too.
        self._fes_global = HCurl(
            self.mesh,
            order=self.order,
            **hcurl_flags(self.nedelec),
            complex=self._is_lossy(),
            dirichlet=self.bc
        )
        fes = self._fes_global
        u, v = fes.TnT()

        # Use CoefficientFunctions for material properties (handles special
        # characters in domain names and matches the frequency-loop assembly).
        eps_r_cf, mu_r_cf = self._build_material_cfs()
        k_form = BilinearForm((1 / (mu0 * mu_r_cf)) * curl(u) * curl(v) * dx)
        m_form = BilinearForm(eps0 * eps_r_cf * u * v * dx)

        sigma_cf, eps_tand_cf = self._build_loss_cfs()
        c_form = d_form = None
        if sigma_cf is not None:
            c_form = BilinearForm(sigma_cf * u * v * dx)
        if eps_tand_cf is not None:
            d_form = BilinearForm(eps0 * eps_tand_cf * u * v * dx)

        with TaskManager():
            k_form.Assemble()
            m_form.Assemble()
            for f in (c_form, d_form):
                if f is not None:
                    f.Assemble()

        self.K_global = _real_csr(k_form.mat)
        self.M_global = _real_csr(m_form.mat)
        self.C_global = _real_csr(c_form.mat) if c_form is not None else None
        self.D_global = _real_csr(d_form.mat) if d_form is not None else None
        self._assembled_signature = self._material_signature()

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
            m_bnd_form = BilinearForm(InnerProduct(u.Trace(), v.Trace()) * ds(self._region(port)),
                                      check_unused=False)
            with TaskManager():
                m_bnd_form.Assemble()

            if self.port_solver is not None and hasattr(self.port_solver, 'port_orientation_factors'):
                sigma = self.port_solver.port_orientation_factors.get(port, 1.0)
            else:
                sigma = 1.0
            for mode in sorted(self.port_basis[port].keys()):
                port_mode_cf = self.port_modes[port][mode]

                gf = GridFunction(fes)
                gf.Set(port_mode_cf, definedon=self.mesh.Boundaries(self._region(port)))

                # Use NGSolve native mat-vec (handles definedon DOF mapping correctly)
                res = gf.vec.CreateVector()
                res.data = m_bnd_form.mat * gf.vec
                # port modes are real; a complex (lossy) space only adds 0j
                basis_vectors.append(sigma * np.real(res.FV().NumPy()).copy())

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

            m_bnd_form = BilinearForm(InnerProduct(u.Trace(), v.Trace()) * ds(self._region(port)),
                                      check_unused=False)
            with TaskManager():
                m_bnd_form.Assemble()

            if self.port_solver is not None and hasattr(self.port_solver, 'port_orientation_factors'):
                sigma = self.port_solver.port_orientation_factors.get(port, 1.0)
            else:
                sigma = 1.0
            for mode in sorted(self.port_basis[port].keys()):
                port_mode_cf = self.port_modes[port][mode]

                gf = GridFunction(fes)
                gf.Set(port_mode_cf, definedon=self.mesh.Boundaries(self._region(port)))

                # Use NGSolve native mat-vec (handles DOF mapping correctly)
                res = gf.vec.CreateVector()
                res.data = m_bnd_form.mat * gf.vec
                # port modes are real; a complex (lossy) space only adds 0j
                basis_vectors.append(sigma * np.real(res.FV().NumPy()).copy())

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

    def _resolve_solver_type(self, solver_type: str, fes, nnz: Optional[int] = None) -> str:
        """Resolve 'auto': 'direct' when the factorisation fits in free memory.

        A factorisation serves every right-hand side of a sample (port modes
        and beams) and is never slowed down by a resonance, so it is preferred
        whenever it fits in ``AUTO_MEMORY_SHARE`` of the free memory; otherwise
        the sweep runs iteratively.  ``nnz`` is the number of entries of the
        system matrix (that of K).  Without a memory reading the choice falls
        back to ``AUTO_DOF_THRESHOLD`` unknowns.
        """
        if solver_type != 'auto':
            return solver_type
        ndof = fes.ndof if fes is not None else 0
        free = _available_memory()
        if free is None or nnz is None:
            chosen = 'iterative' if ndof > self.AUTO_DOF_THRESHOLD else 'direct'
            pr.echo(f"  Auto solver: {ndof} unknowns -> '{chosen}' "
                    f"(threshold: {self.AUTO_DOF_THRESHOLD})")
            return chosen
        n_free = int(sum(fes.FreeDofs()))
        need = direct_memory_estimate(
            n_free, nnz, complex_=fes.is_complex,
            symmetric=self._store_symmetric('direct'), inverse=_DIRECT_SOLVER)
        chosen = 'direct' if need <= self.AUTO_MEMORY_SHARE * free else 'iterative'
        pr.echo(f"  Auto solver: {ndof} unknowns, a factorisation needs about "
                f"{need / 1e9:.1f} GB of {free / 1e9:.1f} GB free -> '{chosen}'")
        return chosen

    def _merge_iterative_opts(self, user_opts: Optional[Dict]) -> Dict:
        """Merge user-supplied iterative options with defaults."""
        opts = dict(self.DEFAULT_ITERATIVE_OPTS)
        if user_opts:
            opts.update(user_opts)
        opts['method'] = str(opts['method']).lower()
        if opts['method'] not in self.ITERATIVE_METHODS:
            raise ValueError(f"iterative_opts['method'] must be one of "
                             f"{list(self.ITERATIVE_METHODS)}, got {opts['method']!r}")
        return opts

    def _prepare_iterative(self, fes, opts: Dict):
        """Prepare FES for iterative solve (logging only, no mesh mutation)."""
        pr.info(f"  Iterative mode: {opts['method'].upper()}, precond='{opts['precond']}', "
                f"tol={opts['tol']} (relative), maxsteps={opts['maxsteps']}")
        pr.info(f"  FES ndof: {fes.ndof}")
        return fes

    # === Frequency domain solve ===

    # ------------------------------------------------------------------
    # solve() helpers: config comparison, rerun protection, mesh/order prep
    # ------------------------------------------------------------------

    def _compare_loaded_config(self, fmin, fmax, nsamples,
                               order, nportmodes, nedelec=None) -> List[str]:
        """Compare the requested solve against the loaded (saved) config.

        Returns a list of human-readable differences (frequency range, solver
        settings, geometry history, geometry source-file hashes) and warns if
        any are found — existing results may be invalid for the new request.
        """
        loaded = self._loaded_config or {}
        diffs: List[str] = []
        # Frequencies (only if they were previously solved)
        if loaded.get('fmin') is not None:
            if not np.isclose(fmin, loaded['fmin']):
                diffs.append(f"fmin: {loaded['fmin']} -> {fmin}")
            if not np.isclose(fmax, loaded['fmax']):
                diffs.append(f"fmax: {loaded['fmax']} -> {fmax}")
            if nsamples != loaded['nsamples']:
                diffs.append(f"nsamples: {loaded['nsamples']} -> {nsamples}")

        # Solver settings
        if order is not None and order != loaded.get('order'):
            diffs.append(f"order: {loaded.get('order')} -> {order}")
        if nedelec is not None and nedelec != loaded.get('nedelec', 'second'):
            diffs.append(f"nedelec: {loaded.get('nedelec', 'second')} -> {nedelec}")
        if nportmodes is not None and nportmodes != loaded.get('n_modes_per_port'):
            diffs.append(f"nportmodes: {loaded.get('n_modes_per_port')} -> {nportmodes}")

        # Geometry history (timestamps/filepaths stripped — files are hashed
        # separately below)
        current_history = getattr(self.geometry, '_history', [])
        loaded_history = loaded.get('geometry_history', [])
        keys_to_ignore = {'timestamp', 'filepath'}
        current_clean = strip_keys(current_history, keys_to_ignore)
        loaded_clean = strip_keys(loaded_history, keys_to_ignore)
        if current_clean != loaded_clean:
            diffs.append("geometry/history has changed:")
            for d in deep_diff(loaded_clean, current_clean, path="geometry_history"):
                diffs.append(f"  {d}")

        # Geometry source files (content hash).  Resolve against the project,
        # not the current working directory.
        component_sources = loaded.get('component_sources', {})
        geometry_dir = (str(Path(self._project_path) / "geometry")
                        if self._project_path else "geometry")
        source_diffs = check_source_files(component_sources, geometry_dir=geometry_dir)
        if source_diffs:
            diffs.append("geometry source file(s) have changed:")
            for d in source_diffs:
                diffs.append(f"  {d}")

        if diffs:
            msg = "\n  Simulation configuration has changed since the results were saved:\n"
            for d in diffs:
                msg += f"    - {d}\n"
            pr.info(msg)
        return diffs

    def _compare_current_sweep(self, fmin, fmax, nsamples,
                               order=None, nportmodes=None, nedelec=None) -> List[str]:
        """Differences between a requested sweep and the one held in memory.

        Covers a second ``solve()`` in the same session (no saved config to
        compare against), which previously returned the old band silently.
        """
        f = self.frequencies
        if f is None or len(f) == 0:
            return []
        diffs = []
        if not np.isclose(fmin, f[0] / 1e9):
            diffs.append(f"fmin: {f[0] / 1e9} -> {fmin}")
        if not np.isclose(fmax, f[-1] / 1e9):
            diffs.append(f"fmax: {f[-1] / 1e9} -> {fmax}")
        if int(nsamples) != len(f):
            diffs.append(f"nsamples: {len(f)} -> {nsamples}")
        if order is not None and order != self.order:
            diffs.append(f"order: {self.order} -> {order}")
        if nedelec is not None and nedelec != self.nedelec:
            diffs.append(f"nedelec: {self.nedelec} -> {nedelec}")
        spec = (self._nportmodes_spec if self._nportmodes_spec is not None
                else self._n_modes_per_port)
        if nportmodes is not None and nportmodes != spec:
            diffs.append(f"nportmodes: {spec} -> {nportmodes}")
        solved = getattr(self, '_solved_signature', None)
        if solved is not None and solved != self._material_signature():
            diffs.append("materials changed")
        if diffs:
            pr.info("\n  The request differs from the results in memory:\n"
                    + "".join(f"    - {d}\n" for d in diffs))
        return diffs

    def _has_valid_results(self) -> bool:
        """True if this solver already holds usable sweep results."""
        has_results = (
            (self._S_matrix is not None and self._Z_matrix is not None)
            or bool(self._S_per_domain)
            or self._Z_global_coupled is not None
            or self._foms_cache is not None
            or self._fom_cache is not None
        )
        # Results are only usable if frequencies are also present.
        if has_results and (self.frequencies is None or len(self.frequencies) == 0):
            has_results = False
        return has_results

    def _cached_results_or_none(self, diffs, compute_s_params,
                                per_domain, global_method) -> Optional[Dict]:
        """Rerun protection: return existing results instead of recomputing.

        Called only when ``rerun=False``.  Returns the cached results dict if
        valid results exist (warning if the config changed since they were
        made), or ``None`` to proceed with the compute.
        """
        if self._has_valid_results():
            if diffs:
                pr.warning("  The configuration changed, but rerun=False: "
                           "returning the STORED results (they do not match "
                           "the request).")
            else:
                pr.milestone("  Returning cached results. "
                             "(Use rerun=True to force recompute)")
            return self._build_results_dict(compute_s_params, per_domain,
                                            global_method)
        pr.debug(
            f"  No valid results found. Forcing compute. "
            f"(Z_coupled={self._Z_global_coupled is not None}, "
            f"foms={self._foms_cache is not None}, "
            f"fmin={self._loaded_config.get('fmin') if self._loaded_config else None})"
        )
        return None

    def _sync_and_validate_mesh(self) -> None:
        """Adopt the geometry's mesh if the solver has none; fail if still none.

        Also re-initializes the port solver when the mesh exists but the
        solver state was cleared (e.g. after a reload).
        """
        geo = self.geometry
        if (self.mesh is None and geo is not None and geo.mesh is None
                and getattr(geo, 'mesh_on_demand', False)):
            # Geometries that are built without a mesh (the bodies of
            # revolution) are meshed here with their own settings if
            # generate_mesh() was never called.
            pr.milestone(f"No mesh yet: meshing {type(geo).__name__} with maxh={geo.maxh} m "
                         "(call generate_mesh() before solving to choose the mesh).")
            geo.generate_mesh()
        if self.mesh is None and self.geometry and self.geometry.mesh:
            self.mesh = self.geometry.mesh
            # Sync back to the project for consistency and auto-save.
            if hasattr(self, '_project_ref') and self._project_ref:
                self._project_ref.mesh = self.mesh

        if self.mesh is None:
            raise RuntimeError(
                "FrequencyDomainSolver has no mesh. "
                "Please generate a mesh using geometry.generate_mesh() before solving."
            )

        if self.port_solver is None:
            self.port_solver = self._new_port_solver()
        else:
            # materials may have been (re)assigned since the solver was built
            self._attach_port_media(self.port_solver)

    def _apply_order_change(self, order: Optional[int],
                            nedelec: Optional[str] = None) -> None:
        """Switch FE order and/or Nedelec kind: rebuild FE spaces, port solver
        and matrices."""
        new_order = order is not None and order != self.order
        new_kind = nedelec is not None and nedelec != self.nedelec
        if not (new_order or new_kind):
            return
        if new_order:
            self.order = order
        if new_kind:
            self.nedelec = nedelec
        if self.mesh is not None:
            # The requested mode counts survive an order change.
            spec, n_modes = self._nportmodes_spec, self._n_modes_per_port
            self._reset_discretisation()
            self._nportmodes_spec, self._n_modes_per_port = spec, n_modes
            self._reconstruct_fes()
            self.port_solver = self._new_port_solver()

    @staticmethod
    def _validate_sweep(fmin, fmax, nsamples) -> int:
        """Reject frequency sweeps the solver cannot handle; ``nsamples`` as int."""
        return validate_sweep(fmin, fmax, nsamples)

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

        Each finished frequency sample is written to ``fds/checkpoint/``; an
        interrupted sweep resumes when ``solve()`` is called again with the
        same request (only the missing samples are computed).

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
            Individual solve parameters (order, nportmodes, store_snapshots, etc.).
            ``nedelec='first'`` (default) or ``'second'`` selects the Nedelec
            element kind: ``'first'`` has the same curls as ``'second'`` with
            about a third fewer unknowns at order 2 (see
            :mod:`cavsim3d.solvers.nedelec`).  A project saved without the
            setting was computed with ``'second'`` and keeps it.
        """
        # 1. Merge config and kwargs
        cfg = (config or {}).copy()
        cfg.update(kwargs)
        check_solve_options(cfg, FOM_SOLVE_OPTIONS, where="fds.solve()")

        # 2. Extract core parameters with defaults
        fmin = fmin if fmin is not None else cfg.get('fmin')
        fmax = fmax if fmax is not None else cfg.get('fmax')
        nsamples = nsamples if nsamples is not None else cfg.get('nsamples', 100)

        # Validate mandatory frequency range
        if fmin is None or fmax is None:
            raise ValueError("fmin and fmax must be provided (either directly or via config).")
        nsamples = self._validate_sweep(fmin, fmax, nsamples)

        # Assembly NETLIST: per-component FOM stage (each unique component is
        # run once or loaded from its saved project; imported components are
        # never recomputed).  Continue with fds.foms.reduce(tol).concatenate().
        asm_netlist = self._netlist_assembly()
        if asm_netlist is not None:
            if self.beam_setup is not None:
                pr.warning("The beam is not solved for coupled parts (imported or repeated) "
                           "yet: only the port results are. Glue the parts, or solve them "
                           "as one model, for beam results.")
            cfg['fmin'], cfg['fmax'], cfg['nsamples'] = fmin, fmax, nsamples
            return self._solve_netlist(asm_netlist, cfg)

        # 3. Extract other options from merged cfg
        order = cfg.get('order')
        nedelec = cfg.get('nedelec')
        if nedelec is not None:
            check_kind(nedelec)
        nportmodes = cfg.get('nportmodes')
        store_snapshots = cfg.get('store_snapshots', True)
        compute_s_params = cfg.get('compute_s_params', True)
        per_domain = cfg.get('per_domain', True)
        global_method = cfg.get('global_method', 'coupled')
        solver_type = cfg.get('solver_type', 'auto')
        iterative_opts = cfg.get('iterative_opts')
        # rerun: None (default) = reuse results for the SAME request and
        # recompute automatically when anything that affects them changed;
        # True = always recompute; False = keep stored results regardless.
        rerun = cfg.get('rerun', None)
        # None: keep the console verbosity as set (pr.set_verbosity)
        verbose = cfg.get('verbose')

        # Quasi-TEM port options (microstrip / inhomogeneous cross-sections).
        # Consumed by _build_qtem_solve_kwargs during matrix assembly.  Ports
        # with a non-uniform permittivity cross-section auto-enable qTEM even
        # if not listed here.
        # Reference impedance for TEM ports: 'line' (CST's default, and the
        # only one that reproduces CST's Z-matrix) or 'wave' (the historical
        # behaviour). TE/TM always use the wave impedance -- they have no
        # unique voltage/current, and CST reports no line impedance for them.
        # qTEM ports already use their power-voltage line impedance.
        self.impedance_reference = cfg.get('impedance_reference', 'line')
        if self.impedance_reference not in ('line', 'wave'):
            raise ValueError(
                f"impedance_reference must be 'line' or 'wave', "
                f"got {self.impedance_reference!r}")

        # Port-mode source ('analytic' | 'numeric'); a change re-solves the modes
        ms = cfg.get('mode_source', self.port_mode_source)
        msi = cfg.get('mode_source_internal', self.port_mode_source_internal)
        for val in (ms, msi):
            if val not in ('analytic', 'numeric'):
                raise ValueError(f"mode_source must be 'analytic' or 'numeric', got {val!r}")
        port_setting_diffs = []
        if (ms, msi) != (self.port_mode_source, self.port_mode_source_internal):
            port_setting_diffs.append(
                f"mode_source: {(self.port_mode_source, self.port_mode_source_internal)}"
                f" -> {(ms, msi)}")
            self.port_mode_source, self.port_mode_source_internal = ms, msi
            if self._mesh is not None:
                self.port_solver = self._new_port_solver()
                self.port_modes = None
                self.port_basis = None

        self._qtem_ports = cfg.get('qtem_ports')
        self._qtem_conductor_bbnd = cfg.get('qtem_conductor_bbnd')
        self._qtem_voltage_path = cfg.get('qtem_voltage_path')

        # Start file logging if project path exists
        _file_handler = None
        if getattr(self, '_project_path', None):
            log_dir = Path(self._project_path) / "fds"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = str(log_dir / "solve.log")
            _file_handler = pr.start_file_log(self._log_path)

        # Verbosity for this solve only
        _prev_verbosity = pr.push_verbosity(verbose)
        try:
            # --- Config comparison + rerun policy ---
            diffs = []
            if rerun is not True:
                if self._loaded_config:
                    diffs = self._compare_loaded_config(fmin, fmax, nsamples,
                                                        order, nportmodes, nedelec)
                elif self._has_valid_results():
                    diffs = self._compare_current_sweep(fmin, fmax, nsamples,
                                                        order, nportmodes, nedelec)
                diffs += port_setting_diffs
                diffs += self._beam_diffs()

            if rerun is False or (rerun is None and not diffs):
                cached = self._cached_results_or_none(
                    diffs, compute_s_params, per_domain, global_method)
                if cached is not None:
                    return cached
            if (rerun is None and diffs and set(diffs) <= set(self.BEAM_DIFFS)
                    and self._has_valid_results()):
                # Only the beam is new: keep the port results, add its columns
                if self._port_snapshots_available():
                    return self._solve_beam_only(store_snapshots, solver_type,
                                                 self._merge_iterative_opts(iterative_opts),
                                                 compute_s_params)
                pr.info("  The port solutions were not stored (store_snapshots=False): "
                        "the beam needs them, so everything is solved again.")
            recompute = rerun is True or bool(diffs)
            if rerun is None and diffs and self._has_valid_results():
                pr.info("  The request differs from the stored results -> "
                        "recomputing (pass rerun=False to keep the stored ones).")

            # --- Mesh synchronization and validation ---
            self._sync_and_validate_mesh()

            if recompute:
                # Explicitly clear ROM/Concat children of existing FOM caches
                if self._fom_cache:
                    self._fom_cache.clear_rom()
                if self._foms_cache:
                    self._foms_cache.clear_roms()
                self._clear_results()
            if rerun is True:
                # a forced recompute also ignores samples of an interrupted sweep
                from cavsim3d.solvers.sweep_checkpoint import clear_checkpoints
                clear_checkpoints(self._checkpoint_root())

            self._apply_order_change(order, nedelec)

            self.frequencies = np.linspace(fmin, fmax, nsamples) * 1e9

            # Merge iterative options with defaults
            iter_opts = self._merge_iterative_opts(iterative_opts)

            # Normalize options for single-domain structures
            if not self.is_compound:
                per_domain = False
                global_method = 'coupled'

            # Only the coupled global solve exists (or None for per-domain only)
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
            if self._current_global_method == 'coupled':
                # Keep the coupled caches identical to the reported result
                # (_compute_s_from_z rescales TEM ports to the line reference).
                self._Z_global_coupled = self._Z_matrix
                self._S_global_coupled = self._S_matrix
            if self._beam_raw:
                self._compute_beam_tilde(compute_s_params)
                self._beam_fingerprint = self.beam_setup.fingerprint()

            self._invalidate_cache()

            # Record in solver history
            self._solved_signature = self._material_signature()
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

            self._persist()
            # the complete results are saved: the per-sample checkpoint is spent
            # (a redirected one is cleared by its owner once it has staged them)
            if getattr(self, '_checkpoint_dir', None) is None:
                from cavsim3d.solvers.sweep_checkpoint import clear_checkpoints
                clear_checkpoints(self._checkpoint_root())

            return self._build_results_dict(compute_s_params, per_domain, global_method)
        finally:
            pr.pop_verbosity(_prev_verbosity)
            if _file_handler:
                pr.stop_file_log(_file_handler)

    def _solve_per_domain(
        self,
        store_snapshots: bool,
        solver_type: str = 'auto',
        iter_opts: Optional[Dict] = None,
        beam_only: bool = False,
    ) -> None:
        """Solve each domain on its own: its faces to other domains are ports.

        ``beam_only``: the port results are stored; solve the beam columns
        only, with the stored port solutions for h_Z.
        """
        iter_opts = iter_opts or self._merge_iterative_opts(None)

        for domain in self.domains:
            t_domain_start = time.time()
            pr.info(f"\nSolving domain: {domain}")

            domain_ports = self.domain_port_map[domain]
            fes = self._fes[domain]

            # Material properties for this domain (may span multiple mesh materials)
            mesh_mats = self._get_domain_mesh_materials(domain)
            domain_materials = [(mm, *self._material_props(mm)) for mm in mesh_mats]

            st = self._resolve_solver_type(solver_type, fes, nnz=_nnz(self.K.get(domain)))
            pr.debug(f"  Solver type: {st}")
            if st == 'iterative':
                fes = self._prepare_iterative(fes, iter_opts)
                self._fes[domain] = fes

            # Excitation order: the domain's ports, each with its modes
            excitation_keys = []
            for pm, port_m in enumerate(domain_ports):
                if port_m not in self.port_modes:
                    continue
                for mode_m in sorted(self.port_modes[port_m].keys()):
                    excitation_keys.append((pm, port_m, mode_m))
            n_excitations = len(excitation_keys)
            n_freqs = len(self.frequencies)

            u, v = fes.TnT()

            def add_operator(a_form, omega):
                # A(w) = K + jwC - w^2 (M - jD), per mesh material of the domain
                for mm, eps_r, mu_r, sigma, tand in domain_materials:
                    a_form += (1 / (mu0 * mu_r)) * curl(u) * curl(v) * dx(region_pattern([mm]))
                    a_form += -omega ** 2 * (eps0 * eps_r) * u * v * dx(region_pattern([mm]))
                    if sigma:
                        a_form += 1j * omega * sigma * u * v * dx(region_pattern([mm]))
                    if tand:
                        a_form += 1j * omega ** 2 * (eps0 * eps_r * tand) * u * v * dx(region_pattern([mm]))

            beam = self._make_beam_system(domain, fes, self.B[domain], excitation_keys,
                                          region_materials=mesh_mats)
            out = self._sweep(domain, fes, self.B[domain], add_operator, st, iter_opts,
                              store_snapshots, beam=beam,
                              port_x=self.snapshots[domain] if beam_only else None)
            if beam is not None:
                self._beam_raw[domain] = out['beam']
            if beam_only:
                pr.done(f"  Beam columns of {domain}: {time.time() - t_domain_start:.2f}s")
                continue
            Z_matrix = out['Z']

            # Z as the solver's internal row-first dict
            self._Z_per_domain[domain] = {}
            for col, (pm, port_m, mode_m) in enumerate(excitation_keys):
                for row, (pn, port_n, mode_n) in enumerate(excitation_keys):
                    key = f"{pn + 1}({mode_n + 1}){pm + 1}({mode_m + 1})"
                    self._Z_per_domain[domain][key] = Z_matrix[:, row, col]

            if store_snapshots and n_excitations:
                self.snapshots[domain] = out['snapshots']

            t_elapsed = time.time() - t_domain_start
            msg = f"  Completed: {len(domain_ports)} ports, {n_freqs} frequencies in {t_elapsed:.2f}s"
            if st == 'iterative':
                msg += f" (total iteration steps: {out['total_iters']})"
            pr.done(f"  {msg}")

            from cavsim3d.utils.timing import get_timing_registry
            get_timing_registry().record(
                f"per-domain solve [{domain}]", t_elapsed, category="FOM",
                n_samples=n_freqs, n_dofs=int(fes.ndof), n_ports=len(domain_ports),
                solver_type=st,
            )

            self._store_residuals(domain, n_freqs, out['iters'], out['residuals'], st,
                                  iter_opts, out['methods'])

    def _solve_global_coupled(
        self,
        store_snapshots: bool,
        solver_type: str = 'auto',
        iter_opts: Optional[Dict] = None,
        beam_only: bool = False,
        ) -> None:
        """Solve the entire structure as one coupled system.

        ``beam_only``: the port results are stored; solve the beam columns
        only, with the stored port solutions for h_Z.
        """
        t_start = time.time()

        if iter_opts is None:
            iter_opts = self._merge_iterative_opts(None)

        if self._fes_global is None:
            self._assemble_global_matrices()
        elif bool(self._fes_global.is_complex) != self._is_lossy():
            # a space rebuilt on reopening is real; losses need a complex one
            self._fes_global = HCurl(self.mesh, order=self.order, **hcurl_flags(self.nedelec),
                                     complex=self._is_lossy(), dirichlet=self.bc)

        fes = self._fes_global
        st = self._resolve_solver_type(solver_type, fes, nnz=_nnz(self.K_global))

        # Build spatially-varying material CoefficientFunctions
        eps_r_cf, mu_r_cf = self._build_material_cfs()
        sigma_cf, eps_tand_cf = self._build_loss_cfs()

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

        # Record the (port, mode) order of the global Z/S matrix columns so the
        # dict labels and reference impedances are correct even when ports have
        # different numbers of modes.
        self._port_mode_order = [(p, m) for (_pm, p, m) in excitation_keys]

        u, v = fes.TnT()

        def add_operator(a_form, omega):
            # A(w) = K + jwC - w^2 (M - jD) -- exactly the K/M/C/D _global
            # matrices that the ROM and eigen solvers use
            a_form += (1 / (mu0 * mu_r_cf)) * curl(u) * curl(v) * dx
            a_form += -omega ** 2 * (eps0 * eps_r_cf) * u * v * dx
            if sigma_cf is not None:
                a_form += 1j * omega * sigma_cf * u * v * dx
            if eps_tand_cf is not None:
                a_form += 1j * omega ** 2 * eps0 * eps_tand_cf * u * v * dx

        beam = self._make_beam_system('global', fes, self.B_global, excitation_keys)
        out = self._sweep('global', fes, self.B_global, add_operator, st, iter_opts,
                          store_snapshots, beam=beam,
                          port_x=self.snapshots['global'] if beam_only else None)
        if beam is not None:
            self._beam_raw['global'] = out['beam']
        if beam_only:
            pr.done(f"  Beam columns: {time.time() - t_start:.2f}s")
            return
        self._Z_matrix = out['Z']

        if store_snapshots and n_excitations:
            self.snapshots["global"] = out['snapshots']

        self._Z_global_coupled = self._Z_matrix.copy()
        # Expose per-frequency solve times (freq[Hz], seconds) for reporting.
        self._freq_solve_times = list(zip([float(f) for f in self.frequencies],
                                          out['times']))

        t_elapsed = time.time() - t_start
        msg = f"\nCoupled solve complete: {len(target_ports)} external ports in {t_elapsed:.2f}s"
        if st == 'iterative':
            msg += f" (total iteration steps: {out['total_iters']})"
        pr.done(f"  {msg}")

        from cavsim3d.utils.timing import get_timing_registry
        get_timing_registry().record(
            "coupled solve", t_elapsed, category="FOM",
            n_samples=n_freqs, n_dofs=int(fes.ndof), n_ports=len(target_ports),
            solver_type=st,
        )

        self._store_residuals('global', n_freqs, out['iters'], out['residuals'], st,
                              iter_opts, out['methods'])

    def _sweep(self, key: str, fes, B: np.ndarray, add_operator, st: str,
               iter_opts: Dict, store_snapshots: bool, beam=None,
               port_x: Optional[np.ndarray] = None) -> Dict:
        """Frequency sweep of one system:  A(w) x = w b  for every column b of B.

        ``add_operator(a_form, omega)`` adds A(w) to an empty BilinearForm on
        ``fes``.  Per sample, one factorisation (``st='direct'``) or one
        preconditioner (``'iterative'``) serves every column; an iterative
        solve starts from the same column's solution at the previous sample.
        Samples of an interrupted sweep are read back from its checkpoint.

        ``beam`` (a :class:`~cavsim3d.solvers.beam.BeamSystem`, or None) adds
        one column per beam with the same factorisation / preconditioner: the
        load f - A g for the free unknowns, then e_s = x + g (two real columns,
        Re and Im, when the system is real).  ``port_x`` (the stored port
        solutions, one column per sample and excitation) skips the port
        solves: only the beam columns are solved, and ``Z`` stays zero.

        Returns a dict: ``Z`` (n_freqs, n, n) with Z = j B^T X; ``snapshots``
        (ndof, n_freqs * n) or None; per solve ``iters``, ``residuals`` and
        ``methods``; per sample ``times``; ``total_iters`` of the solves run;
        with a beam, ``beam``: ``kZ`` (n_freqs, n, S), ``hZ`` (n_freqs, L, n),
        ``zoc`` (n_freqs, L, S), ``snapshots`` (ndof, n_freqs * S) or None and
        the beam solves' ``iters`` and ``residuals``.
        """
        n_excitations = B.shape[1]
        n_freqs = len(self.frequencies)
        x_dtype = complex if fes.is_complex else float

        # Frequency-independent right-hand sides: the columns of B
        pr.debug(f"  Pre-assembling {n_excitations} RHS vectors...")
        template_vec = LinearForm(fes)
        with TaskManager():
            template_vec.Assemble()
        template_vec = template_vec.vec
        rhs_base_vectors = []
        for col in range(n_excitations):
            vec = template_vec.CreateVector()
            vec.FV().NumPy()[:] = B[:, col]   # the assembled port load, no integration
            rhs_base_vectors.append(vec)

        freedofs = fes.FreeDofs()
        free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]], dtype=np.int64)
        n_free = len(free_idx)
        pr.debug(f"  DOFs: {fes.ndof} total, {n_free} free")

        # Samples an interrupted solve of this same sweep already computed
        ckpt, restored = self._open_sweep_checkpoint(
            key if port_x is None else f"{key}_beam", fes, n_free, B,
            store_snapshots, beam=beam, port_solves=port_x is None)

        Z = np.zeros((n_freqs, n_excitations, n_excitations), dtype=complex)
        # snapshots: one column per (sample, excitation), filled in place
        snaps = (np.empty((fes.ndof, n_freqs * n_excitations), dtype=x_dtype, order='F')
                 if store_snapshots else None)

        n_src = beam.n_sources if beam is not None else 0
        if beam is not None:
            n_path = beam.n_paths
            kZ = np.zeros((n_freqs, n_excitations, n_src), dtype=complex)
            hZ = np.zeros((n_freqs, n_path, n_excitations), dtype=complex)
            zoc = np.zeros((n_freqs, n_path, n_src), dtype=complex)
            beam_snaps = (np.empty((fes.ndof, n_freqs * n_src), dtype=complex, order='F')
                          if store_snapshots else None)
            beam_iters, beam_res = [], []
            rhs_beam = template_vec.CreateVector()
            xb_prev = None     # previous sample's free parts, per beam and part

        total_iters = 0
        iters, residuals, methods, times = [], [], [], []
        rhs_scaled = template_vec.CreateVector()
        sol_vec = template_vec.CreateVector()

        batch = max(1, n_freqs // 10)
        t_batch, n_restored = time.time(), 0
        x_prev = None          # previous sample's solutions: Krylov start vectors
        for kk, freq in enumerate(self.frequencies):
            if kk % batch == 0:
                pr.debug(f"  Frequency {kk + 1}/{n_freqs}: {freq / 1e9:.4f} GHz")

            if kk in restored and port_x is not None:
                r = restored[kk]
                kZ[kk], hZ[kk], zoc[kk] = r['beam_kZ'], r['beam_hZ'], r['beam_zoc']
                if store_snapshots:
                    beam_snaps[:, kk * n_src:(kk + 1) * n_src] = r['beam_x']
                xb_prev = None
                times.append(r['time'])
                continue
            if kk in restored:
                r = restored[kk]
                Z[kk] = r['Z']
                iters.extend(r['iters'].tolist())
                residuals.extend(r['res'].tolist())
                methods.extend(['restored'] * n_excitations)
                times.append(r['time'])
                if store_snapshots:
                    snaps[:, kk * n_excitations:(kk + 1) * n_excitations] = r['x']
                x_prev = r['x']
                if beam is not None:
                    kZ[kk], hZ[kk], zoc[kk] = r['beam_kZ'], r['beam_hZ'], r['beam_zoc']
                    if store_snapshots:
                        beam_snaps[:, kk * n_src:(kk + 1) * n_src] = r['beam_x']
                    xb_prev = None
                n_restored += 1
                if (kk + 1) % batch == 0 or kk == n_freqs - 1:
                    self._report_batch(kk - kk % batch, kk, t_batch, iters, residuals,
                                       n_excitations, st, iter_opts, restored=n_restored,
                                       methods=methods)
                    t_batch, n_restored = time.time(), 0
                continue

            t_freq_start = time.time()
            omega = 2 * np.pi * freq

            sym = self._store_symmetric(st)
            a_form = BilinearForm(fes, symmetric=sym, symmetric_storage=sym)
            add_operator(a_form, omega)
            if st == 'direct':
                with TaskManager():
                    a_form.Assemble()
                    inv_a = a_form.mat.Inverse(freedofs=freedofs, inverse=_DIRECT_SOLVER)
            else:
                precond = self._assemble_with_preconditioner(
                    a_form, fes, iter_opts['precond'])

            # All excitations of this sample with the one factorisation /
            # preconditioner
            x_all = np.zeros((fes.ndof, n_excitations), dtype=x_dtype)
            if port_x is not None:
                x_all[:] = port_x[:, kk * n_excitations:(kk + 1) * n_excitations]
            for col in (range(n_excitations) if port_x is None else ()):
                rhs_scaled.data = omega * rhs_base_vectors[col]
                if st == 'direct':
                    # forward/backward substitution only: the factorisation is reused
                    sol_vec.data = inv_a * rhs_scaled
                    iters.append(0)
                    residuals.append(0.0)
                    methods.append('direct')
                else:
                    # start from the same excitation's solution at the previous
                    # sample (fewer steps than a zero start)
                    sol_vec, n_it, res, used = self._solve_system(
                        fes, a_form, rhs_scaled, precond, iter_opts,
                        None if x_prev is None else x_prev[:, col], free_idx)
                    total_iters += n_it
                    iters.append(n_it)
                    residuals.append(res)
                    methods.append(used)
                x_all[:, col] = sol_vec.FV().NumPy()

            beam_x = None
            if beam is not None:
                # the beam columns: A x = f - A g on the free unknowns, e_s = x + g
                beam_x = np.zeros((fes.ndof, n_src), dtype=complex)
                xb_now = []
                for j in range(n_src):
                    parts = []
                    for part, (g, f_vec) in enumerate(beam.lift_and_load(j, omega)):
                        rhs_beam.FV().NumPy()[:] = f_vec
                        rhs_beam.data -= a_form.mat * g.vec
                        if st == 'direct':
                            sol_vec.data = inv_a * rhs_beam
                            beam_iters.append(0)
                            beam_res.append(0.0)
                        else:
                            sol_vec, n_it, res, _used = self._solve_system(
                                fes, a_form, rhs_beam, precond, iter_opts,
                                None if xb_prev is None else xb_prev[j][part], free_idx)
                            total_iters += n_it
                            beam_iters.append(n_it)
                            beam_res.append(res)
                        x_free = sol_vec.FV().NumPy().copy()
                        parts.append(x_free)
                        e_part = x_free + g.vec.FV().NumPy()
                        beam_x[:, j] += e_part if part == 0 else 1j * e_part
                    xb_now.append(parts)
                xb_prev = xb_now
                for j in range(n_src):
                    kZ[kk, :, j] = beam.port_voltages(j, omega, beam_x[:, j])
                zoc[kk] = beam.voltages(omega, beam_x)
                hZ[kk] = 1j * beam.voltages(omega, x_all)
                if store_snapshots:
                    beam_snaps[:, kk * n_src:(kk + 1) * n_src] = beam_x

            # release this sample's factorization / preconditioner before the
            # next one is built (otherwise both are held at once)
            inv_a = precond = None

            if port_x is not None:
                times.append(time.time() - t_freq_start)
                ckpt.write(kk, Z[kk], None, [], [], times[-1],
                           extra=dict(beam_kZ=kZ[kk], beam_hZ=hZ[kk], beam_zoc=zoc[kk],
                                      **({'beam_x': beam_x} if store_snapshots else {})))
                pr.debug(f"  \tsample {kk + 1}: beam columns {times[-1]:.1f} s")
                continue

            # Z = j B^T X
            Z[kk, :, :] = 1j * (B.T @ x_all)

            if store_snapshots:
                snaps[:, kk * n_excitations:(kk + 1) * n_excitations] = x_all
            x_prev = x_all

            times.append(time.time() - t_freq_start)
            extra = None
            if beam is not None:
                extra = dict(beam_kZ=kZ[kk], beam_hZ=hZ[kk], beam_zoc=zoc[kk])
                if store_snapshots:
                    extra['beam_x'] = beam_x
            ckpt.write(kk, Z[kk], x_all if store_snapshots else None,
                       iters[-n_excitations:], residuals[-n_excitations:], times[-1],
                       extra=extra)
            if (kk + 1) % batch == 0 or kk == n_freqs - 1:
                self._report_batch(kk - kk % batch, kk, t_batch, iters, residuals,
                                   n_excitations, st, iter_opts, restored=n_restored,
                                   methods=methods)
                t_batch, n_restored = time.time(), 0

        out = dict(Z=Z, snapshots=snaps, iters=iters, residuals=residuals,
                   methods=methods, times=times, total_iters=total_iters)
        if beam is not None:
            out['beam'] = dict(kZ=kZ, hZ=hZ, zoc=zoc, snapshots=beam_snaps,
                               iters=beam_iters, residuals=beam_res)
        return out

    _PRECONDITIONERS = {
        'local': lambda a: preconditioners.Local(a),
        'multigrid': lambda a: preconditioners.MultiGrid(a),
        'bddc': lambda a: preconditioners.BDDC(a),
        'hcurlamg': lambda a: preconditioners.HCurlAMG(a),
    }

    def _assemble_with_preconditioner(self, a_form, fes, name: str):
        """Register the named preconditioner, then assemble ``a_form``.

        NGSolve preconditioners hook into element assembly, so they must be
        created BEFORE ``Assemble()``.  ``'direct'`` factorises the assembled
        matrix instead.  Unknown names fall back to ``'local'`` with a warning.
        """
        name = str(name).lower()
        precond = None
        if name != 'direct':
            if name not in self._PRECONDITIONERS:
                pr.warning(f"Unknown preconditioner {name!r}; using 'local'. "
                           f"Options: {sorted(self._PRECONDITIONERS) + ['direct']}")
                name = 'local'
            precond = self._PRECONDITIONERS[name](a_form)
        with TaskManager():
            a_form.Assemble()
            if name == 'direct':
                precond = a_form.mat.Inverse(fes.FreeDofs(), inverse=_DIRECT_SOLVER)
        return precond

    def _store_symmetric(self, solver_type: str) -> bool:
        """Store and factorise the (complex) symmetric A(w) as symmetric?

        Measured with PARDISO on a 1.1M-unknown cavity: for second-kind
        elements the symmetric factorisation takes about the same time with
        half the memory; for first-kind elements it also halves the memory but
        takes ~50 % longer, so those keep the full matrix.
        """
        return solver_type == 'direct' and self.nedelec == 'second'

    @staticmethod
    def _report_batch(first, last, t_start, freq_iters, freq_residuals,
                      n_excitations, solver_type, opts, restored=0, methods=None):
        """Print the wall time of samples ``first``..``last`` (0-based).

        Indented one tab deeper than the ``Frequency k/N`` line that opens the
        batch.  For the iterative solver it adds the Krylov method and its
        steps per solve, the largest relative residual, how many solves
        reached ``maxsteps`` without converging and how many COCG solves were
        finished by GMRES (``methods``, one entry per solve).  ``restored``
        samples came from the checkpoint of an interrupted solve and are not
        timed.
        """
        n = last - first + 1 - restored
        span = f"  \tsamples {first + 1}-{last + 1}: "
        if n == 0:
            pr.debug(span + "read from the checkpoint")
            return
        dt = time.time() - t_start
        msg = span + f"{dt:.1f} s ({dt / n:.1f} s/sample)"
        if restored:
            msg += f", {restored} read from the checkpoint"
        its = np.asarray(freq_iters[first * n_excitations:(last + 1) * n_excitations])
        res = np.asarray(freq_residuals[first * n_excitations:(last + 1) * n_excitations])
        if solver_type == 'iterative' and its.size:
            name = str(opts.get('method', 'gmres')).upper()
            msg += (f", {name} {its.mean():.0f} steps/solve (max {its.max()}), "
                    f"residual <= {res.max():.1e}")
            stalled = int(np.sum(its >= opts['maxsteps']))
            if stalled:
                msg += f", {stalled} solve(s) stopped at maxsteps={opts['maxsteps']}"
            used = (methods or [])[first * n_excitations:(last + 1) * n_excitations]
            fallbacks = sum(m == 'cocg+gmres' for m in used)
            if fallbacks:
                msg += f", {fallbacks} solve(s) finished by GMRES"
        pr.debug(msg)

    def _checkpoint_root(self) -> Optional[Path]:
        """Folder of this solver's sweep checkpoints (None: no project).

        ``<project>/fds/checkpoint`` -- unless ``_checkpoint_dir`` redirects it:
        a netlist section solved in a scratch project keeps its samples in the
        importing project, so an interrupted netlist solve resumes too.
        """
        if getattr(self, '_checkpoint_dir', None) is not None:
            return Path(self._checkpoint_dir)
        root = getattr(self, '_project_path', None)
        return Path(root) / "fds" / "checkpoint" if root else None

    def _open_sweep_checkpoint(self, key, fes, n_free, rhs, store_snapshots, beam=None,
                               port_solves: bool = True):
        """The checkpoint of this sweep and the samples it already holds.

        Samples without stored solutions do not count when snapshots are
        wanted (they were written by a solve with ``store_snapshots=False``).
        With a beam the sweep's identity includes the beam definition, and a
        sample counts only with its beam outputs.
        """
        from cavsim3d.solvers.sweep_checkpoint import (SweepCheckpoint, rhs_signature,
                                                       sweep_fingerprint)
        folder = self._checkpoint_root()
        if folder is None:
            return SweepCheckpoint(None, key), {}
        ckpt = SweepCheckpoint(
            folder, key,
            sweep_fingerprint(self.frequencies, fes.ndof, n_free, np.shape(rhs)[1],
                              self._material_signature(),
                              beam=beam.setup.fingerprint() if beam is not None else None),
            rhs_signature(rhs))
        need = ('beam_kZ', 'beam_hZ', 'beam_zoc') if beam is not None else ()
        restored = {k: r for k, r in ckpt.load().items()
                    if 0 <= k < len(self.frequencies)
                    and (not port_solves or r['x'] is not None or not store_snapshots)
                    and all(r.get(n) is not None for n in need)
                    and (beam is None or not store_snapshots
                         or r.get('beam_x') is not None)}
        if restored:
            pr.milestone(f"  Resuming an interrupted sweep ({key}): {len(restored)} of "
                         f"{len(self.frequencies)} samples read from {ckpt.folder}")
        return ckpt, restored

    def _solve_system(self, fes, a_form, f_vec, precond, opts: Dict, x0=None,
                      free_idx=None):
        """Solve ``a_form.mat * x = f_vec`` with a preconditioned Krylov method.

        ``opts['method']`` is ``'cocg'`` (conjugate gradients with the
        unconjugated product, for complex symmetric matrices -- the default)
        or ``'gmres'``.  ``opts['tol']`` is relative to the right-hand side:
        the solve stops when the preconditioned residual is ``tol`` times that
        of the zero start vector, so a warm start ``x0`` (a numpy vector; zero
        if None) saves steps without tightening the tolerance.  A COCG solve
        that breaks down or stops at ``opts['maxsteps']`` without reaching the
        tolerance is continued by GMRES from its iterate.

        Returns
        -------
        x : BaseVector
            Solution.
        iters : int
            Krylov steps, of both methods when GMRES finished a COCG solve.
        residual : float
            True relative residual ||Ax - b|| / ||b|| on the free DOFs.
        method : str
            ``'cocg'``, ``'gmres'`` or ``'cocg+gmres'``.
        """
        mat = a_form.mat
        # a Preconditioner exposes .mat; the 'direct' option already is one
        pre = getattr(precond, 'mat', precond)
        sol = f_vec.CreateVector()
        if x0 is None:
            sol[:] = 0
        else:
            sol.FV().NumPy()[:] = x0
        pf = f_vec.CreateVector()
        r = f_vec.CreateVector()
        with TaskManager():
            pf.data = pre * f_vec
        tol = float(opts['tol'])
        iters, method = 0, opts.get('method', 'cocg')
        # The solver objects count their own steps: an iteration callback would
        # make NGSolve rebuild the solution at every step.  initialize=False
        # keeps the start vector; the first residual check is the start vector,
        # not a step.
        if method == 'cocg':
            # COCG's residual measure is sqrt|r^T P r|; relative to the zero start
            ref = np.sqrt(abs(pf.InnerProduct(f_vec, conjugate=False)))
            solver = CGSolver(mat=mat, pre=pre, conjugate=False, maxiter=opts['maxsteps'],
                              atol=max(tol * ref, 1e-300), printrates=opts['printrates'])
            with TaskManager():
                solver.Solve(rhs=f_vec, sol=sol, initialize=False)
            iters = max(solver.iterations - 1, 0)
            last = solver.residuals[-1] if solver.residuals else np.inf
            if not (np.isfinite(last) and last <= solver._final_residual):
                # broke down or stopped at maxsteps: GMRES continues from the
                # iterate (from the start vector if the iterate is not finite)
                if not np.all(np.isfinite(sol.FV().NumPy())):
                    if x0 is None:
                        sol[:] = 0
                    else:
                        sol.FV().NumPy()[:] = x0
                method = 'cocg+gmres'
        if method != 'cocg':
            solver = GMResSolver(mat=mat, pre=pre, maxiter=opts['maxsteps'],
                                 atol=max(tol * Norm(pf), 1e-300),
                                 printrates=opts['printrates'])
            with TaskManager():
                solver.Solve(rhs=f_vec, sol=sol, initialize=False)
            iters += max(solver.iterations - 1, 0)
        if opts['printrates']:
            print('=' * 50)
        with TaskManager():
            r.data = mat * sol - f_vec

        # Norms on the free DOFs only (Dirichlet rows are not solved for)
        if free_idx is None:
            fd = fes.FreeDofs()
            free_idx = np.array([i for i in range(fes.ndof) if fd[i]], dtype=np.int64)
        r_norm = np.linalg.norm(r.FV().NumPy()[free_idx])
        b_norm = np.linalg.norm(f_vec.FV().NumPy()[free_idx])
        rel_res = r_norm / b_norm if b_norm > 0 else r_norm

        return sol, iters, rel_res, method

    def _ensure_matrices_assembled(
        self,
        per_domain: bool,
        global_method: Optional[str],
        nportmodes: Optional[int] = None
    ) -> None:
        """Ensure required matrices are assembled."""
        needs_global = (global_method == 'coupled')
        needs_per_domain = per_domain

        # Materials changed since the matrices were assembled -> re-assemble
        sig = self._material_signature()
        prev = getattr(self, '_assembled_signature', None)
        if prev is not None and sig is not None and sig != prev:
            self._global_matrices_assembled = False
            self._per_domain_matrices_assembled = False

        # A flag restored from config.json whose matrices were never saved (a
        # solve interrupted before its results were written) -> assemble again
        if self._global_matrices_assembled and self.B_global is None:
            self._global_matrices_assembled = False
        if self._per_domain_matrices_assembled and not self.B:
            self._per_domain_matrices_assembled = False

        # Check if port modes exist or if we need a different number of modes.
        # Compare like with like: a dict/list spec against the stored spec,
        # not against its scalar summary (which never compares equal).
        needs_recompute = False
        current_spec = (self._nportmodes_spec if self._nportmodes_spec is not None
                        else self._n_modes_per_port)
        if nportmodes is not None and nportmodes != current_spec:
            needs_recompute = True
            self.port_modes = None # Force recompute

        # Quasi-TEM port modes are solved at a reference wavenumber k0 = 2*pi*fmax/c,
        # so a change in the frequency band invalidates them (unlike frequency-
        # independent TE/TM modes).  Re-solve when fmax moved.
        if (self.port_modes is not None and self.frequencies is not None
                and self._has_qtem_ports()):
            cur_fmax = float(np.max(self.frequencies))
            prev = getattr(self, '_port_modes_fmax', None)
            if prev is not None and not np.isclose(cur_fmax, prev, rtol=1e-6):
                needs_recompute = True
                self.port_modes = None

        # no modes yet, or a port without modes (a port solver saved before
        # its modes were computed): solve the port modes and assemble
        if not self.port_modes or not all(self.port_modes.values()):
            self.assemble_matrices(
                nportmodes=nportmodes or current_spec or 1,
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

    def _persist(self) -> None:
        """Save after a stage, through the owning project when there is one.

        Saving only ``fds/`` left ``project.json``, the geometry and the mesh
        unwritten, so reopening the project found nothing to load.
        """
        if self._project_ref is not None:
            self._project_ref.save()
        elif self._project_path is not None:
            self.save()

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
        from cavsim3d.core.persistence import ProjectManager
        from datetime import datetime
        import json

        # Netlist mode: the module solver holds no global matrices/port modes
        # of its own — per-component state lives under fds/foms/<name>/ (or is
        # linked).  Only the solve config belongs at the module level.
        if self._netlist_foms is not None:
            fds_path = Path(path) if path else (Path(self._project_path) / "fds"
                                                if self._project_path else None)
            if fds_path is not None:
                from cavsim3d.solvers import netlist_persistence as npz
                fds_path.mkdir(parents=True, exist_ok=True)
                cfg = getattr(self._netlist_foms, '_config', {}) or {}
                with open(fds_path / "config.json", "w") as f:
                    json.dump(npz._jsonable(cfg), f, indent=2)
            return fds_path

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
            "nedelec": self.nedelec,
            "bc": self.bc,
            "use_wave_impedance": self.use_wave_impedance,
            "is_compound": self.is_compound,
            "n_domains": self.n_domains,
            "domains": self.domains,
            "n_ports": len(self._ports),
            "ports": self._ports,
            "external_ports": self._external_ports,
            "internal_ports": self._internal_ports,
            "lossy": bool(self.C_global is not None or self.D_global is not None
                          or self.C or self.D),
            "n_modes_per_port": self._n_modes_per_port,
            # full per-port spec (int | list | dict) -- the scalar above is
            # only its maximum
            "nportmodes_spec": self._nportmodes_spec,
            "port_mode_order": self._port_mode_order,
            "z_reference": getattr(self, '_z_reference', self.Z_REFERENCE_VERSION),
            "global_matrices_assembled": self._global_matrices_assembled,
            "per_domain_matrices_assembled": self._per_domain_matrices_assembled,
            "current_global_method": self._current_global_method,
            "has_results": bool(
                self._Z_global_coupled is not None
                or self._Z_per_domain
                or self._fom_cache is not None
                or self._resonant_mode_cache
            ),
            "solver_history": self._solver_history,
            "geometry_history": getattr(self.geometry, '_history', []),
            # the beam definition the stored beam results belong to
            "beam": ({"fingerprint": self._beam_fingerprint,
                      "setup": next(iter(self._beam_tilde.values())).get('setup')}
                     if self._beam_tilde else None),
            "timestamp": datetime.now().isoformat(),
        }

        ProjectManager.save_json(fds_path, config, filename="config.json")

        # 2. Port modes - save via PortEigenmodeSolver (includes all port data).
        # A port solver whose modes are not computed yet is not saved (it would
        # be read back as ports without modes); a stale file is removed.
        port_file = fds_path / "port_modes" / "port_modes.pkl"
        ps = getattr(self, 'port_solver', None)
        if ps is not None and ps.port_modes and all(ps.port_modes.values()):
            port_file.parent.mkdir(parents=True, exist_ok=True)
            # Use the new save method that extracts raw numpy data
            ps.save_to_file(port_file)
        elif port_file.exists():
            port_file.unlink()

        # Beam: the electrostatic port solutions Phi_reg (own file: the port
        # modes do not depend on the beam)
        beam_file = fds_path / "port_modes" / "beam_port_fields.pkl"
        if self._beam_systems:
            import pickle
            from cavsim3d.solvers.beam import port_fields
            beam_file.parent.mkdir(parents=True, exist_ok=True)
            with open(beam_file, "wb") as fh:
                pickle.dump({"fingerprint": self._beam_fingerprint,
                             "systems": {k: port_fields(b)
                                         for k, b in self._beam_systems.items()}}, fh)
        elif not self._beam_tilde and beam_file.exists():
            beam_file.unlink()

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
            use_wave_impedance=config.get("use_wave_impedance", True),
            nedelec=config.get("nedelec", "second"),
        )
        
        # Load matrices, snapshots, etc. from 'path'
        fds.mesh = mesh  # Set mesh before loading internals
        fds._load_internal(path, config)

        # Restore log path if log file exists on disk
        log_file = path / "solve.log"
        if log_file.exists():
            fds._log_path = str(log_file)

        return fds

    def _load_internal(self, path: Path, config: dict):
        """Internal helper to load solver state from a directory."""
        import h5py
        from cavsim3d.core.persistence import H5Serializer
        self._loaded_config = config

        # Restore state flags and topology
        self.is_compound = config.get("is_compound", self.is_compound)
        self.domains = config.get("domains", self.domains)
        self.n_domains = config.get("n_domains", len(self.domains))
        self._ports = config.get("ports", self._ports)
        self._external_ports = config.get("external_ports", self._external_ports)
        self._internal_ports = config.get("internal_ports", self._internal_ports)
        
        self._n_modes_per_port = config.get("n_modes_per_port", self._n_modes_per_port)
        self._nportmodes_spec = config.get("nportmodes_spec", self._nportmodes_spec)
        if config.get("lossy"):
            # lossy results are complex -> rebuild the per-domain spaces complex
            self._reconstruct_fes()
        _pmo = config.get("port_mode_order")
        if _pmo is not None:
            # JSON round-trips tuples as lists; restore (port_name, mode_idx).
            self._port_mode_order = [(str(p), int(m)) for p, m in _pmo]
        self._global_matrices_assembled = config.get("global_matrices_assembled", False)
        self._per_domain_matrices_assembled = config.get("per_domain_matrices_assembled", False)
        self._current_global_method = config.get("current_global_method", None)
        self._beam_fingerprint = (config.get("beam") or {}).get("fingerprint")
        
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
            from cavsim3d.solvers.results import FOMCollection
            try:
                # FOMCollection.load handles matrix loading into self (the _fds_ref)
                self._foms_cache = FOMCollection.load(fom_root_compound, _fds_ref=self)
                if self._foms_cache:
                    self.frequencies = self._foms_cache.frequencies
                
                # Nested ROMs inside foms/roms
                roms_path = fom_root_compound / "roms"
                if roms_path.exists():
                    from cavsim3d.rom.reduction import ModelOrderReduction
                    from cavsim3d.solvers.results import ROMCollection
                    mor_loaded = ModelOrderReduction.load(roms_path, solver=self)
                    self._foms_cache._roms_cache = ROMCollection(_fds_ref=self, _mor_ref=mor_loaded)
            except Exception as e:
                warnings.warn(f"Could not load FOM collection from {fom_root_compound}: {e}")
                
        # Load global 'fom' results (exists for single structures AND compound structures with global solve)
        if fom_root_single.exists():
            from cavsim3d.solvers.results import FOMResult
            try:
                # FOMResult.load handles matrix loading into self (the _solver_ref)
                self._fom_cache = FOMResult.load(fom_root_single, _solver_ref=self)
                if self._fom_cache:
                    if self.frequencies is None:
                        self.frequencies = self._fom_cache.frequencies
                self._Z_global_coupled = self._fom_cache._Z_matrix
                self._S_global_coupled = self._fom_cache._S_matrix
                # The reported global result IS the coupled one (as after a
                # solve), so a reopened project's solve() returns S and Z too.
                if self._Z_matrix is None:
                    self._Z_matrix = self._Z_global_coupled
                if self._S_matrix is None:
                    self._S_matrix = self._S_global_coupled

                # Nested ROM inside fom/rom
                rom_path = fom_root_single / "rom"
                if rom_path.exists():
                    from cavsim3d.rom.reduction import ModelOrderReduction
                    self._fom_cache._rom_cache = ModelOrderReduction.load(rom_path, solver=self)
            except Exception as e:
                pr.warning(f"Could not load FOM result: {e}")

        # ------------------------------------------------------------------
        # 2. Port modes
        # ------------------------------------------------------------------
        port_dir = path / "port_modes"
        if port_dir.exists() and self.mesh is not None:
            pm_file = port_dir / "port_modes.pkl"

            try:
                if pm_file.exists():
                    # Import the PortEigenmodeSolver class
                    from cavsim3d.solvers.ports import PortEigenmodeSolver

                    # Ensure we have fes_global
                    if self._fes_global is None:
                        from ngsolve import HCurl
                        self._fes_global = HCurl(self.mesh, order=self.order, dirichlet=self.bc,
                                                 **hcurl_flags(self.nedelec))

                    # Load using the new method that reconstructs NGSolve objects
                    self.port_solver = PortEigenmodeSolver.load_from_file(
                        pm_file,
                        mesh=self.mesh,
                        fes_full=self._fes_global
                    )

                    # Projects saved before port media were persisted
                    if not self.port_solver.port_media_eps:
                        self._attach_port_media(self.port_solver)

                    # Set convenience references
                    self.port_modes = self.port_solver.port_modes
                    self.port_basis = self.port_solver.port_basis

                    # The saved ROMs were loaded (step 1) before the port
                    # modes existed: hand them the restored modes.
                    roms = [getattr(self._fom_cache, '_rom_cache', None),
                            getattr(getattr(self._foms_cache, '_roms_cache', None),
                                    '_mor_ref', None)]
                    for rom in roms:
                        if rom is not None and getattr(rom, 'port_modes', None) is None:
                            rom.port_modes = self.port_modes
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

        # Results saved under an older reference-impedance convention
        self._z_reference = config.get("z_reference", 1)
        if self._z_reference < self.Z_REFERENCE_VERSION and self.port_solver is not None:
            self._upgrade_saved_reference()

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
        # The base-class matrices back .fom and the S computation; a stale
        # global Z would otherwise survive a per-domain-only re-solve.
        self._Z_matrix = None
        self._S_matrix = None
        self._invalidate_cache()
        self._z_reference = self.Z_REFERENCE_VERSION
        self._current_global_method = None
        self.snapshots = {}
        self._residuals = {}
        # Invalidate result-object caches
        self._fom_cache = None
        self._foms_cache = None
        # Clear resonant modes
        self._resonant_mode_cache = {}
        # Beam outputs belong to the port results they were solved with
        self._clear_beam_results()

    #: _beam_diffs() reasons that leave the port results valid
    BEAM_DIFFS = ("a beam was added", "the beam definition changed")

    def _port_snapshots_available(self) -> bool:
        """True if every solved system kept its port solutions (h_Z needs them)."""
        keys = list(self._Z_per_domain) or (['global'] if self._Z_matrix is not None else [])
        return bool(keys) and all(
            self.snapshots.get(k) is not None
            and np.shape(self.snapshots[k])[1] == len(self.frequencies) * self._n_port_modes_of(k)
            for k in keys)

    def _n_port_modes_of(self, key: str) -> int:
        if key == 'global':
            return 0 if self._Z_matrix is None else int(self._Z_matrix.shape[1])
        return len(self._domain_port_mode_order(key))

    def _solve_beam_only(self, store_snapshots: bool, solver_type: str, iter_opts: Dict,
                         compute_s_params: bool) -> Dict:
        """Add the beam columns to stored port results.

        A(w) is assembled and factorised (or preconditioned) again per
        sample for the beam's right-hand sides; h_Z comes from the stored port
        solutions.  The port results themselves are not touched.
        """
        self._sync_and_validate_mesh()
        pr.milestone("Beam added to solved port results: solving the beam columns only "
                     "(the port results are kept).")
        self._beam_raw, self._beam_systems, self._beam_tilde = {}, {}, {}
        if self._Z_per_domain:
            self._solve_per_domain(store_snapshots, solver_type=solver_type,
                                   iter_opts=iter_opts, beam_only=True)
        else:
            self._solve_global_coupled(store_snapshots, solver_type=solver_type,
                                       iter_opts=iter_opts, beam_only=True)
        self._compute_beam_tilde(compute_s_params)
        self._beam_fingerprint = self.beam_setup.fingerprint()
        # the result objects in memory get the beam (their ROMs stay)
        if self._fom_cache is not None:
            self._fom_cache._beam = self._beam_tilde.get(self._fom_cache.domain)
        if self._foms_cache is not None:
            for fom in self._foms_cache:
                fom._beam = self._beam_tilde.get(fom.domain)
        self._persist()
        if getattr(self, '_checkpoint_dir', None) is None:
            from cavsim3d.solvers.sweep_checkpoint import clear_checkpoints
            clear_checkpoints(self._checkpoint_root())
        return self._build_results_dict(compute_s_params, bool(self._Z_per_domain),
                                        None if self._Z_per_domain else 'coupled')

    def _clear_beam_results(self) -> None:
        """Drop the beam outputs (in memory; save() removes their files)."""
        self._beam_raw = {}
        self._beam_systems = {}
        self._beam_tilde = {}
        self._beam_fingerprint = None

    def _beam_diffs(self) -> List[str]:
        """Why the stored beam results do not answer the request (empty: they do).

        A beam added, changed or removed since the results were solved; the
        port results themselves are unaffected by the beam.
        """
        setup = self.beam_setup
        stored = self._beam_fingerprint
        if setup is None:
            if stored is not None or self._beam_tilde:
                pr.info("  The beam was removed: its results are dropped "
                        "(the port results stay).")
                self._clear_beam_results()
                for fom in ([self._fom_cache] if self._fom_cache is not None else []) + \
                        (list(self._foms_cache) if self._foms_cache is not None else []):
                    fom._beam = None
                joined = getattr(self._foms_cache, '_concat_cache', None)
                if joined is not None and getattr(joined, '_scattering_join', False):
                    # joined through the parts' S~: meaningless without the beam
                    self._foms_cache._concat_cache = None
                    if self._project_path:
                        import shutil
                        shutil.rmtree(Path(self._project_path) / "fds" / "foms" / "concat",
                                      ignore_errors=True)
                self._persist()          # removes the beam files
            return []
        if not self._has_valid_results():
            return []
        if stored is None:
            return ["a beam was added"]
        if stored != setup.fingerprint():
            return ["the beam definition changed"]
        return []

    def _compute_beam_tilde(self, compute_s_params: bool) -> None:
        """Generalised matrices of every system solved with a beam.

        Z~ = [[Z, k_Z], [h_Z, z_oc]] in the normalisation of the reported Z;
        S~ = [[S, k], [h, z_b]] with the same reference impedances as S.  TEM
        ports are rescaled to the line reference exactly as Z is
        (k_Z -> D k_Z, h_Z -> h_Z D); S~ does not depend on that scaling.
        """
        from cavsim3d.solvers import beam as bm
        setup = self.beam_setup
        self._beam_tilde = {}
        for key, raw in self._beam_raw.items():
            kZ, hZ, zoc = raw['kZ'], raw['hZ'], raw['zoc']
            n_f = len(self.frequencies)
            if key == 'global':
                Z = self._Z_matrix
                n = Z.shape[1]
                labels = [f"{p}({m})" for p, m in
                          self._matrix_index_labels(n, self._n_modes_per_port or 1)]
                scale = self._reference_rescale_factors() if compute_s_params else None
                if scale is not None:
                    d = np.sqrt(scale)
                    kZ = d[None, :, None] * kZ
                    hZ = hZ * d[None, None, :]
                Zt = bm.z_tilde(Z, kZ, hZ, zoc)
                St = None
                if compute_s_params:
                    Zref = np.array([self._get_impedance_matrix(f) for f in self.frequencies])
                    St = bm.s_tilde(Z, kZ, hZ, zoc, Zref)
            else:
                order = self._domain_port_mode_order(key)
                Z = self._domain_dict_to_matrix(key, self._Z_per_domain[key])
                labels = [f"{pi + 1}({m + 1})" for (pi, _pn, m) in order]
                # the per-domain Z is reported in the modes' own normalisation;
                # S~ uses the line reference of TEM ports, as the per-domain S
                Zt = bm.z_tilde(Z, kZ, hZ, zoc)
                St = None
                if compute_s_params:
                    scale = np.ones(len(order))
                    for ri, (_pi, pn, m) in enumerate(order):
                        zt = self._get_port_impedance(pn, m, self.frequencies[0])
                        zw = self._port_wave_impedance(pn, m, self.frequencies[0])
                        if zw is not None and abs(zw) > 1e-12:
                            scale[ri] = abs(zt) / abs(zw)
                    d = np.sqrt(scale)
                    Zs = Z * np.outer(d, d)[None]
                    Zref = np.array([np.diag([self._get_port_impedance(pn, m, f)
                                              for (_pi, pn, m) in order])
                                     for f in self.frequencies])
                    St = bm.s_tilde(Zs, d[None, :, None] * kZ, hZ * d[None, None, :], zoc,
                                    Zref)
            rows, cols = bm.matrix_labels(labels, setup)
            system = self._beam_systems.get(key)
            self._beam_tilde[key] = {
                'Z_tilde': Zt, 'S_tilde': St, 'rows': rows, 'cols': cols,
                'frequencies': np.asarray(self.frequencies).copy(),
                'names': {lab: line.name for lab, line in zip(setup.path_labels, setup.paths)},
                'setup': setup.to_dict(), 'fingerprint': setup.fingerprint(),
                'summary': system.summary() if system is not None else raw.get('summary'),
            }
            if n_f and not np.all(np.isfinite(Zt)):
                pr.warning(f"  Beam ({key}): the generalised matrix has non-finite entries.")

    def _print_solve_config(
        self,
        per_domain: bool,
        global_method: Optional[str]
    ) -> None:
        """Print solve configuration."""
        pr.running(f"\nFDS Solve: {self.frequencies[0]/1e9:.4f} - {self.frequencies[-1]/1e9:.4f} GHz, {len(self.frequencies)} samples")
        pr.info("=" * 60)
        pr.info("Frequency Domain Solve Configuration")
        pr.info("=" * 60)
        pr.info(f"Frequency range: {self.frequencies[0]/1e9:.4f} - {self.frequencies[-1]/1e9:.4f} GHz")
        pr.info(f"Number of samples: {len(self.frequencies)}")
        pr.info(f"Structure type: {'Compound' if self.is_compound else 'Single'}")
        pr.info(f"Per-domain solve: {per_domain}")
        pr.info(f"Global method: {global_method}")
        if self.is_compound:
            pr.info(f"External ports: {self._external_ports}")
            pr.info(f"Internal ports: {self._internal_ports}")
        pr.info("=" * 60)

    def _store_residuals(
        self,
        key: str,
        n_freqs: int,
        freq_iters: List[int],
        freq_residuals: List[float],
        solver_type: str,
        opts: Optional[Dict] = None,
        methods: Optional[List[str]] = None,
    ) -> None:
        """Store the per-sample Krylov steps and residuals of one solve.

        ``opts`` (the iterative options) adds ``method``, ``maxsteps`` and
        ``tol``, so a plot can show which solves stopped at the step limit;
        ``methods`` (one entry per solve) counts the COCG solves that GMRES
        finished.
        """
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
        if solver_type == 'iterative' and opts:
            self._residuals[key]['method'] = str(opts.get('method', 'gmres'))
            self._residuals[key]['maxsteps'] = int(opts['maxsteps'])
            self._residuals[key]['tol'] = float(opts['tol'])
            self._residuals[key]['gmres_fallbacks'] = int(
                sum(m == 'cocg+gmres' for m in (methods or [])))

    def _domain_port_mode_order(self, domain: str):
        """Ordered ``(local_port_idx, port_name, mode_idx)`` for a domain.

        Matches the column order of the per-domain Z dict (which labels by the
        1-based local port index within ``domain_port_map[domain]`` and the
        1-based mode index), and supports a different number of modes per port.
        """
        domain_ports = self.domain_port_map[domain]
        order = []
        for pidx, port in enumerate(domain_ports):
            if self.port_modes and port in self.port_modes:
                modes = sorted(self.port_modes[port].keys())
            else:
                modes = list(range(self._n_modes_per_port or 1))
            for m in modes:
                order.append((pidx, port, m))
        return order

    def _compute_per_domain_s_from_z(self) -> None:
        """Compute per-domain S-matrices from per-domain Z data."""
        n_freqs = len(self.frequencies)

        for domain in self.domains:
            if domain not in self._Z_per_domain:
                continue

            order = self._domain_port_mode_order(domain)

            # Build Z-matrix for this domain in the port-mode order.
            Z_d = self._domain_dict_to_matrix(domain, self._Z_per_domain[domain])

            self._S_per_domain[domain] = {}
            # The FOM normalises its port modes to the WAVE impedance, so the
            # raw Z is in that normalisation. When the reported reference is
            # the line impedance (CST's convention for TEM), Z has to be
            # rescaled with it: z_to_s(a*Z, a*Z0) == z_to_s(Z, Z0), so scaling
            # Z and Z0 by the same per-port factor leaves S untouched while
            # turning Z into physical line-referenced ohms.
            scale = np.ones(len(order))
            for ri, (_pi, pn, m) in enumerate(order):
                zt = self._get_port_impedance(pn, m, self.frequencies[0])
                zw = self._port_wave_impedance(pn, m, self.frequencies[0])
                if zw is not None and abs(zw) > 1e-12:
                    scale[ri] = abs(zt) / abs(zw)
            scale_mat = np.sqrt(np.outer(scale, scale))
            if not np.allclose(scale_mat, 1.0):
                Z_d = Z_d * scale_mat[None, :, :]

            for k in range(n_freqs):
                freq = self.frequencies[k]
                Z0_d = np.diag([
                    self._get_port_impedance(pn, m, freq) for (_pi, pn, m) in order
                ])
                S_d = ParameterConverter.z_to_s(Z_d[k], Z0_d)
                for ri, (pi, _pn, mi) in enumerate(order):
                    for ci, (pj, _pm, mj) in enumerate(order):
                        key = f'{pi + 1}({mi + 1}){pj + 1}({mj + 1})'
                        if key not in self._S_per_domain[domain]:
                            self._S_per_domain[domain][key] = np.zeros(n_freqs, dtype=complex)
                        self._S_per_domain[domain][key][k] = S_d[ri, ci]

    def _get_per_domain_s_matrices(self) -> Dict[str, np.ndarray]:
        """Get per-domain S-matrices as arrays.

        Returns
        -------
        dict
            {domain: S_array} where S_array is
            (n_freqs, n_port_modes_d, n_port_modes_d).
        """
        domain_S = {}

        for domain in self.domains:
            if domain not in self._S_per_domain:
                continue

            domain_S[domain] = self._domain_dict_to_matrix(
                domain, self._S_per_domain[domain])

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

        # Beam: generalised matrices (with labels) per system
        if self._beam_tilde:
            if 'global' in self._beam_tilde and global_method is not None:
                t = self._beam_tilde['global']
                results['Z_tilde'] = t['Z_tilde']
                results['S_tilde'] = t['S_tilde'] if compute_s_params else None
                results['tilde_labels'] = (t['rows'], t['cols'])
            if per_domain:
                results['Z_tilde_per_domain'] = {
                    d: t['Z_tilde'] for d, t in self._beam_tilde.items() if d != 'global'}
                results['S_tilde_per_domain'] = {
                    d: t['S_tilde'] for d, t in self._beam_tilde.items() if d != 'global'}

        # Snapshots
        results['snapshots'] = self.snapshots.copy()

        # Residuals
        results['residuals'] = self._residuals.copy()

        return results

    # === Method comparison ===

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
            {'frequencies', 'methods', 'S_<method>'} dict to plot. If None,
            uses the current solution.
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
    DEFAULT_MIN_EIGENVALUE = MIN_EIGENVALUE  # omega^2 of 1 MHz: below is static

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
            Threshold (omega^2) for static mode filtering. Default: omega^2 of 1 MHz
        n_modes : int, optional
            Number of eigenvalues to return (after filtering)
        sigma : float, optional
            Shift (in omega^2) for shift-invert mode; modes nearest to it are
            found. Default: the centre of the solved band, or ~c0/L from the
            model size when nothing has been solved yet.

        Returns
        -------
        modes : tuple or dict
            Tuple of (eigenvalues, eigenvectors) if domain is specified.
            Dict mapping domain names to (eigenvalues, eigenvectors) if domain is None.
        """
        # A single-domain structure only assembles the global system, so its
        # domain name and 'global' denote the same (K, M).
        single = self.n_domains == 1 and self.M_global is not None
        if domain is None and single:
            res = self.calculate_resonant_modes(domain='global', filter_static=filter_static,
                                               min_eigenvalue=min_eigenvalue, n_modes=n_modes, sigma=sigma)
            return {self.domains[0]: res, 'global': res}
        if single and domain == self.domains[0] and domain not in self.M:
            domain = 'global'

        # Check cache first
        cache_key = f"{domain}_{filter_static}_{min_eigenvalue}_{n_modes}_{sigma}"
        if domain is not None and cache_key in self._resonant_mode_cache:
            return self._resonant_mode_cache[cache_key]

        # Check if matrices are available
        has_global = self.M_global is not None
        if not self.M and not has_global:
            raise ValueError("Matrices not assembled. Call assemble_matrices() first.")

        def compute_eigs(M_mat, K_mat, fes, label: str) -> Tuple[np.ndarray, np.ndarray]:
            """Shift-invert eigenpairs of one domain/global system (free DOFs)."""
            freedofs = fes.FreeDofs()
            free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]], dtype=np.int64)
            if len(free_idx) == 0:
                pr.warning(f"No free DOFs for {label}")
                return np.array([]), np.zeros((fes.ndof, 0))
            return self._compute_eigenpairs_sparse(
                M_mat, K_mat, free_idx, fes.ndof, n_modes=n_modes or 50, sigma=sigma)

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
                available = list(self.M.keys()) + (['global'] if has_global else [])
                raise KeyError(f"Domain '{domain}' not found. Available: {available}")

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
            Default: 1 MHz (see core.constants.STATIC_MODE_CUTOFF_HZ)
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

        Z_full = self._domain_dict_to_matrix(domain, self._Z_per_domain[domain])
        if freq_idx is not None:
            return Z_full[freq_idx]
        return Z_full

    def _domain_dict_to_matrix(self, domain: str, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Per-domain ``{'i(m)j(n)': values}`` dict -> (n_freq, N, N) matrix.

        Rows/columns follow :meth:`_domain_port_mode_order`, so ports may carry
        different numbers of modes.  Per-domain keys are row-first
        (``'<row port>(<row mode>)<col port>(<col mode>)'``).
        """
        order = self._domain_port_mode_order(domain)
        n = len(order)
        out = np.zeros((len(self.frequencies), n, n), dtype=complex)
        for ri, (pi, _pn, mi) in enumerate(order):
            for ci, (pj, _pm, mj) in enumerate(order):
                key = f'{pi + 1}({mi + 1}){pj + 1}({mj + 1})'
                if key in data:
                    out[:, ri, ci] = data[key]
        return out

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

        S_full = self._domain_dict_to_matrix(domain, self._S_per_domain[domain])
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
                'C': self.C_global,
                'D': self.D_global,
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
            C = self.C.get(domain)
            D = self.D.get(domain)
            W = self.snapshots.get(domain)
            fes = self._fes.get(domain)
            
            # Fallback for single-domain projects
            if not self.is_compound:
                if M is None: M = self.M_global
                if K is None: K = self.K_global
                if B is None: B = self.B_global
                if C is None: C = self.C_global
                if D is None: D = self.D_global
                if W is None: W = self.snapshots.get('global')
                if fes is None: fes = self._fes_global

            return {
                'M': M,
                'K': K,
                'B': B,
                'C': C,
                'D': D,
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
                'C': self.C_global,
                'D': self.D_global,
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
            print("\n  global:")
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

    def port_map(self) -> List[Dict]:
        """Detected ports, in the order ``nportmodes`` lists expect.

        Works straight after the geometry is assigned -- it only fits each port
        face, with no eigenvalue solve -- so a per-port mode count can be
        written against real port names and types.

        Returns
        -------
        list of dict
            ``{'index', 'port', 'geometry', 'dims_mm', 'modes'}`` per port,
            where ``modes`` names the family the port will carry (``'TEM +
            TE/TM'`` for a coaxial port, ``'TE/TM'`` otherwise).
        """
        from cavsim3d.solvers.ports import (group_port_faces,
                                            sorted_logical_ports)

        # A NETLIST never meshes the assembly: each unique section is solved
        # standalone, so nportmodes applies per section and the ports that
        # matter are the section's own.
        asm = self._netlist_assembly()
        if asm is not None and self.mesh is None:
            rows = []
            seen = set()
            for key in asm._component_order:
                entry = asm._components[key]
                base = entry.base_name
                if base in seen:
                    continue
                seen.add(base)
                comp = entry.geometry
                mesh = getattr(comp, 'mesh', None)
                if mesh is None:
                    rows.append({'index': 0, 'section': base, 'port': '(unmeshed)',
                                 'geometry': 'unknown', 'dims_mm': '',
                                 'modes': '', 'role': 'section'})
                    continue
                sub = PortEigenmodeSolver(mesh, self.order, self.bc)
                sub.port_face_region = group_port_faces(mesh.GetBoundaries())
                for i, port in enumerate(sorted_logical_ports(sub.port_face_region)):
                    rows.append(dict(self._port_row(sub, i, port),
                                     section=base, role='section port'))
            return rows

        if self.mesh is None:
            raise RuntimeError(
                "No mesh yet: assign geometry (and generate_mesh) first.")
        if self.port_solver is None:
            self.port_solver = self._new_port_solver()
        ps = self.port_solver
        ps.port_face_region = group_port_faces(self.mesh.GetBoundaries())
        ports = sorted_logical_ports(ps.port_face_region)

        internal = set(getattr(self, '_internal_ports', None) or [])
        rows = []
        for i, port in enumerate(ports):
            row = self._port_row(ps, i, port)
            # A glued assembly SHARES its join, so N sections give fewer ports
            # than N x (ports per section): two 4-port cavities glue to 7, and
            # the join is internal. Its mode count still matters -- it sets how
            # well the two halves couple.
            row['role'] = 'internal (join)' if port in internal else 'external'
            rows.append(row)
        return rows

    @staticmethod
    def _port_row(ps, index: int, port: str) -> Dict:
        """One :meth:`port_map` row: geometry, size and mode family."""
        try:
            g = ps.port_geometries.get(port) or ps._detect_port_geometry(port)
            ps.port_geometries[port] = g
            kind = g.type.value
            if getattr(g, 'inner_radius', None):
                dims = (f'R_out={g.radius * 1e3:.2f} '
                        f'R_in={g.inner_radius * 1e3:.2f}')
            elif getattr(g, 'radius', None):
                dims = f'R={g.radius * 1e3:.2f}'
            else:
                dims = f'area={g.area * 1e6:.1f} mm^2'
            fam = 'TEM + TE/TM' if kind == 'coaxial' else 'TE/TM'
        except Exception as e:                        # detection is best-effort
            kind, dims, fam = 'unknown', f'({type(e).__name__})', 'TE/TM'
        return {'index': index, 'port': port, 'geometry': kind,
                'dims_mm': dims, 'modes': fam}

    def print_port_map(self) -> None:
        """Print :meth:`port_map` as a table, with usage examples."""
        rows = self.port_map()
        netlist = any('section' in r for r in rows)
        head = f'{"idx":>4s}  {"port":10s}{"geometry":12s}{"dims [mm]":26s}{"carries":14s}'
        print(('  ' + f'{"section":12s}' if netlist else '') + head
              + ('' if netlist else 'role'))
        for r in rows:
            pre = f'  {r.get("section", ""):12s}' if netlist else ''
            tail = '' if netlist else r.get('role', '')
            print(pre + f'{r["index"]:>4d}  {r["port"]:10s}{r["geometry"]:12s}'
                  f'{r["dims_mm"]:26s}{r["modes"]:14s}' + tail)

        if netlist:
            secs = {}
            for r in rows:
                secs.setdefault(r['section'], []).append(r['port'])
            print('\nThis is a netlist: each section is solved on its own, so '
                  'nportmodes applies\nPER SECTION (not to the whole chain). '
                  'The join ports are eliminated when\nthe sections are '
                  'concatenated, so the coupled system has fewer ports.')
            for s, ns in secs.items():
                print(f'  section {s!r}: {len(ns)} ports -> '
                      f'nportmodes={[1] * len(ns)} or '
                      f'{{{", ".join(repr(n) + ": 1" for n in ns[:2])}, '
                      f"'default': 1}}")
            return

        names = [r['port'] for r in rows]
        n_int = sum(1 for r in rows if r.get('role', '').startswith('internal'))
        print('\nnportmodes accepts:')
        print('  int   nportmodes=1')
        print(f'  list  nportmodes={[1] * len(names)}   '
              f'(one entry per port, this order)')
        ex = ', '.join(f"'{n}': 1" for n in names[:2])
        print(f'  dict  nportmodes={{{ex}, \'default\': 1}}')
        if n_int:
            print(f'\n  {n_int} internal (join) port(s) are included above: a '
                  f'glued assembly SHARES\n  its joins, so the port count is '
                  f'not simply (ports per section) x (sections).\n  Give them '
                  f'a count too -- the join modes set how well the halves '
                  f'couple.')

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
            print("\nGlobal (coupled):")
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
            _display_webgui_fallback(Draw(mode_cf.real, self.mesh, **kwargs))
            print("\nPlotting Imaginary part:")
            _display_webgui_fallback(Draw(mode_cf.imag, self.mesh, **kwargs))
            print("\nPlotting Magnitude:")
            _display_webgui_fallback(Draw(Norm(mode_cf), self.mesh, **kwargs))
            return
        else:
            cf_plot = mode_cf

        _display_webgui_fallback(Draw(cf_plot, self.mesh, **kwargs))
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
            # Faraday, e^{+jwt}: curl E = -j w mu H  =>  H = j curl(E) / (w mu)
            _eps_r_cf, mu_r_cf = self._build_material_cfs()
            field_cf = (1j / (omega * mu0 * mu_r_cf)) * curl(E_gf)
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

        _display_webgui_fallback(Draw(BoundaryFromVolumeCF(cf_plot), self.mesh, **draw_kwargs))
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
        # An explicit domain wins; 'global' is only the default when none is given.
        if domain == 'global' or (domain is None and 'global' in self.snapshots):
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

        # Snapshot columns are stored per frequency in excitation order: the
        # ports in order, each with its (sorted) modes -- ports may carry
        # different numbers of modes.
        excitations = [(p, m) for p in available_ports
                       if self.port_modes and p in self.port_modes
                       for m in sorted(self.port_modes[p])]
        if not excitations:  # no port-mode data (e.g. after a partial load)
            n_modes = self._n_modes_per_port or 1
            excitations = [(p, m) for p in available_ports for m in range(n_modes)]
        if (excitation_port, excitation_mode) not in excitations:
            raise ValueError(
                f"No excitation ({excitation_port!r}, mode {excitation_mode}). "
                f"Available: {excitations}")
        snapshot_idx = (freq_idx * len(excitations)
                        + excitations.index((excitation_port, excitation_mode)))

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
        source: Literal['global', 'coupled'] = 'global',
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
        source : {'global', 'coupled'}
            Which results to plot:
            - 'global': Current global results
            - 'coupled': Cached coupled results
        **kwargs
            Additional arguments passed to plot functions
        """
        import matplotlib.pyplot as plt

        # Get appropriate S-matrix
        if source == 'coupled' and self._S_global_coupled is not None:
            S = self._S_global_coupled
            method_label = "Coupled"
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
        source: Literal['global', 'coupled'] = 'global',
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
        source : {'global', 'coupled'}
            Which results to plot
        **kwargs
            Additional arguments passed to plot functions
        """
        import matplotlib.pyplot as plt

        # Get appropriate Z-matrix
        if source == 'coupled' and self._Z_global_coupled is not None:
            Z = self._Z_global_coupled
            method_label = "Coupled"
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
        source: Literal['global', 'coupled'] = 'global'
    ) -> np.ndarray:
        """
        Get S-matrix at a specific frequency.

        Parameters
        ----------
        freq_ghz : float
            Frequency in GHz
        source : {'global', 'coupled'}
            Which results to use

        Returns
        -------
        S : ndarray
            S-parameter matrix at the specified frequency
        """
        idx = self.get_frequency_index(freq_ghz)

        if source == 'coupled' and self._S_global_coupled is not None:
            return self._S_global_coupled[idx]
        else:
            if self._S_matrix is None:
                raise ValueError("S-parameters not available.")
            return self._S_matrix[idx]

    def get_z_at_frequency(
        self,
        freq_ghz: float,
        source: Literal['global', 'coupled'] = 'global'
    ) -> np.ndarray:
        """
        Get Z-matrix at a specific frequency.

        Parameters
        ----------
        freq_ghz : float
            Frequency in GHz
        source : {'global', 'coupled'}
            Which results to use

        Returns
        -------
        Z : ndarray
            Z-parameter matrix at the specified frequency
        """
        idx = self.get_frequency_index(freq_ghz)

        if source == 'coupled' and self._Z_global_coupled is not None:
            return self._Z_global_coupled[idx]
        else:
            if self._Z_matrix is None:
                raise ValueError("Z-parameters not available.")
            return self._Z_matrix[idx]

    def export_touchstone(
        self,
        filename: Union[str, Path],
        source: Literal['global', 'coupled'] = 'global',
        format: Literal['MA', 'DB', 'RI'] = 'MA',
        z0: Optional[float] = 50.0
    ) -> str:
        """
        Export S-parameters to a Touchstone v1 (``.sNp``) file.

        Parameters
        ----------
        filename : str or Path
            Output filename (``.sNp`` is appended if missing)
        source : {'global', 'coupled'}
            Which results to export
        format : {'MA', 'DB', 'RI'}
            Data format (Magnitude-Angle, dB-Angle, Real-Imaginary)
        z0 : float or None
            Reference impedance in ohm (default 50).  Every port is
            renormalised to it (via Z), so the file's ``R <z0>`` option line
            is exact and any circuit simulator reads the data correctly.
            ``None``: write S exactly as solved, each port referenced to its
            own impedance (modal wave impedance for TE/TM, line impedance for
            TEM) -- the values ``fom.plot_s`` shows.  Touchstone has no way
            to state those (frequency-dependent) references: the option line
            then says ``R 50`` only nominally, the true references are listed
            in the header comments, and a warning says so.

        Returns
        -------
        str
            The path written.
        """
        if source == 'coupled' and self._S_global_coupled is not None:
            S, Z = self._S_global_coupled, self._Z_global_coupled
        else:
            S, Z = self._S_matrix, self._Z_matrix

        if S is None:
            raise ValueError("S-parameters not available.")
        if format not in ('MA', 'DB', 'RI'):
            raise ValueError(f"format must be 'MA', 'DB' or 'RI', got {format!r}")

        if z0 is not None:
            if Z is None:
                raise ValueError("Renormalising to z0 needs the Z-parameters.")
            z0 = float(z0)
            if not (np.isfinite(z0) and z0 > 0):
                raise ValueError(f"z0 must be a positive impedance in ohm, got {z0!r}")
            S = ParameterConverter.z_to_s(Z, z0)
        else:
            warnings.warn(
                "export_touchstone(z0=None) writes S referenced to each port's own "
                "impedance, but the file's option line says R 50: a tool reading the "
                "file takes the data as 50-ohm S-parameters. Pass z0=50 (the default) "
                "for a file that is exact as written.", UserWarning, stacklevel=2)

        n_ports = S.shape[1]
        n_freqs = len(self.frequencies)

        # Construct filename with proper extension
        filename = str(filename)
        if not filename.endswith(f'.s{n_ports}p'):
            filename = f"{filename}.s{n_ports}p"

        def pair(v):
            if format == 'MA':
                return f"{np.abs(v):.9e} {np.angle(v, deg=True):.6f}"
            if format == 'DB':
                return f"{20 * np.log10(np.abs(v) + 1e-300):.9e} {np.angle(v, deg=True):.6f}"
            return f"{v.real:.9e} {v.imag:.9e}"

        with open(filename, 'w') as f:
            f.write("! Touchstone file exported from cavsim3d FrequencyDomainSolver\n")
            f.write(f"! Method: {source}\n")
            f.write(f"! Ports: {n_ports}\n")
            if z0 is None:
                f.write("! S is referenced to each port's own impedance (not to R):\n")
                try:
                    zref = np.diag(self._get_impedance_matrix(self.frequencies[0]))
                    for i, zr in enumerate(zref, 1):
                        f.write(f"!   port-mode {i}: Z_ref = {zr:.6g} ohm at "
                                f"{self.frequencies[0] / 1e9:.6g} GHz\n")
                except Exception:
                    pass
            else:
                f.write(f"! Every port-mode renormalised to {z0:g} ohm\n")
            f.write(f"# GHz S {format} R {50.0 if z0 is None else float(z0)}\n")

            for k in range(n_freqs):
                Sk = S[k]
                if n_ports == 2:
                    # Touchstone v1 2-port order is S11 S21 S12 S22
                    rows = [[Sk[0, 0], Sk[1, 0], Sk[0, 1], Sk[1, 1]]]
                else:
                    rows = [list(Sk[i, :]) for i in range(n_ports)]
                lines = []
                for row in rows:
                    # at most 4 pairs per line
                    for c in range(0, len(row), 4):
                        lines.append("  ".join(pair(v) for v in row[c:c + 4]))
                f.write(f"{self.frequencies[k] / 1e9:.9e}  {lines[0]}\n")
                for ln in lines[1:]:
                    f.write(f"  {ln}\n")

        print(f"Exported to {filename}")
        return filename

    def reset(self) -> None:
        """Reset solver state, clearing all results but keeping geometry."""
        self._clear_results()
        self._netlist_foms = None
        self.frequencies = None
        self._invalidate_cache()
        print("Solver state reset. Matrices retained.")

    def full_reset(self) -> None:
        """Full reset including matrices."""
        self.reset()

        # FE spaces, K/M/B and the loss matrices C/D, assembly flags, port
        # solver and port modes (rebuilt by the next solve)
        self._reset_discretisation()

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