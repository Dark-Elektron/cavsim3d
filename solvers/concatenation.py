from __future__ import annotations
"""
Structure concatenation for multi-cell analysis.

A ConcatenatedSystem represents a SINGLE unified structure formed by
coupling multiple reduced-order models at internal ports. Field
reconstruction produces fields over the entire unified mesh.

Key concepts:
- Multiple ROMs are coupled via Kirchhoff constraints at internal ports
- The result is ONE structure with external ports only
- Field visualization shows the entire structure, not individual pieces
"""

from typing import List, Tuple, Dict, Optional, Callable, Union, Any, Literal
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ngsolve import (
    Norm, curl, BoundaryFromVolumeCF, GridFunction, HCurl, Mesh
)
from ngsolve.webgui import Draw

from solvers.eigen_mixin import ConcatEigenMixin
from core.constants import Z0, mu0
from solvers.base import BaseEMSolver
from utils.plot_mixin import PlotMixin
from rom.structures import ReducedStructure
from core.persistence import H5Serializer, ProjectManager
import h5py
import json
from pathlib import Path
from datetime import datetime
import cavsim3d.utils.printing as pr


# Connection specification: ((struct_idx, port_name), (struct_idx, port_name))
Conn = Tuple[Tuple[int, str], Tuple[int, str]]
ConnSigns = Tuple[float, float]  # (signA, signB) e.g. (+1,-1)


class ConcatenatedSystem(BaseEMSolver, ConcatEigenMixin, PlotMixin):
    """
    Unified structure formed by coupling multiple reduced-order models.

    This class represents the concatenation of multiple ROMs as a SINGLE
    structure. After coupling, the system has only external ports - internal
    connections are eliminated via Kirchhoff constraints.

    Field reconstruction produces fields over the ENTIRE unified mesh,
    not individual sub-structures.

    Coupling Formulation
    --------------------
    Each ReducedStructure has:
        (A - ω² I) x = ω B u
        Z = j B^T x

    Concatenation couples internal ports via:
        F^T B_int^T x = 0

    where F is an incidence matrix encoding the connection topology.

    Parameters
    ----------
    structures : list of ReducedStructure
        The reduced-order models to concatenate. Each must have:
        - Ard, Brd: reduced system matrices
        - W, Q_L_inv: reconstruction matrices
        - fes: per-domain FEM space
        - domain: domain name matching mesh material region
    mesh : Mesh
        The unified mesh for the entire structure (required for field plots)
    fes : HCurl
        The unified FEM space covering entire mesh (required for field plots)
    port_impedance_func : callable, optional
        Function (port, mode, freq) -> impedance
    solver_ref : object, optional
        Reference to original solver

    Examples
    --------
    >>> # Create from ROMs
    >>> rom = ModelOrderReduction(solver)
    >>> rom.reduce(tol=1e-6)
    >>> structures = rom.get_all_structures()
    >>>
    >>> # Concatenate into unified structure
    >>> concat = ConcatenatedSystem(
    ...     structures,
    ...     mesh=solver.mesh,
    ...     fes=solver._fes_global
    ... )
    >>> concat.define_connections([((0, 'port2'), (1, 'port1'))])
    >>> concat.couple()
    >>>
    >>> # Solve and visualize unified field
    >>> concat.solve(fmin=1, fmax=10, nsamples=100)
    >>> concat.plot_field(freq_idx=50)  # Shows entire structure
    """

    DEFAULT_MIN_EIGENVALUE = 1.0
    ITERATIVE_SIZE_THRESHOLD = 10000

    def __init__(
        self,
        structures: List[ReducedStructure],
        mesh: Optional[Mesh] = None,
        fes: Optional[HCurl] = None,
        port_impedance_func: Optional[Callable[[str, int, float], complex]] = None,
        solver_ref: Any = None,
    ):
        super().__init__()

        self.structures = structures
        self.n_structures = len(structures)
        self._port_impedance_func = port_impedance_func or self._default_impedance
        self._solver_ref = solver_ref

        # Unified mesh and FEM space for the entire structure
        self.mesh = mesh
        self.fes = fes

        # Initialize with default before validation
        self._n_modes_per_port = 1

        # Try to resolve mesh/fes from available sources
        self._resolve_mesh_and_fes()

        # Validate consistent mode counts across structures (updates _n_modes_per_port)
        self._validate_mode_counts()

        # Build global port-mode indexing
        self._assign_global_port_modes()

        # Compute DOF offsets for reconstruction
        self._compute_dof_offsets()

        self.A_coupled: Optional[np.ndarray] = None
        self.B_coupled: Optional[np.ndarray] = None
        self.W_coupled: Optional[np.ndarray] = None

        # Caches
        self._resonant_mode_cache = {}

        # Connection tracking
        self.connections: Optional[List[Conn]] = None
        self.n_connections: int = 0
        self._connection_signs: Optional[List[ConnSigns]] = None

        # Port classification
        self._internal_port_modes: List[int] = []
        self._external_port_modes: List[int] = []
        self._external_port_mode_names: List[Tuple[str, int]] = []
        self._external_port_mode_map: Dict[str, Tuple[int, str, int]] = {}
        self._permutation: Optional[np.ndarray] = None
        self._n_internal: int = 0
        self._n_external: int = 0

        # Domain names for reference
        self.domains = [s.domain for s in structures]

        # Snapshot storage (populated by solve())
        self._snapshots: Optional[np.ndarray] = None

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """
        Compute or retrieve eigenmodes for the concatenated system.
        """
        res = super().get_eigenmodes(**kwargs)
        
        # Hierarchical save
        if _auto_save:
            self._auto_save_eigenmodes(res, **kwargs)
            
        return res

    def get_eigenvalues(self, **kwargs):
        """Compute or retrieve eigenvalues for the concatenated system."""
        kwargs['return_eigenvalues'] = True
        kwargs.setdefault('return_eigenvectors', False)
        # Delegate down the MRO (bypasses _auto_save to avoid double saving/looping if only requesting values)
        return super().get_eigenmodes(**kwargs)

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        if self._solver_ref is None or not hasattr(self._solver_ref, '_project_path'):
            return
        
        # Mirror: fds/foms/roms/concat -> eigenmode/foms/roms/concat
        # Note: Concatenated systems are usually under roms
        save_path = Path(self._solver_ref._project_path) / "eigenmode" / "foms" / "roms" / "concat"
        try:
            self.save_eigenmodes(save_path, **kwargs)
        except Exception as e:
            pr.warning(f"Could not auto-save eigenmodes for ConcatenatedSystem: {e}")

    def _resolve_mesh_and_fes(self) -> None:
        """Resolve mesh and fes from available sources."""
        # Try to get mesh from structures
        if self.mesh is None:
            for struct in self.structures:
                if struct.mesh is not None:
                    self.mesh = struct.mesh
                    break

        # Try to get from solver_ref
        if self._solver_ref is not None:
            if self.mesh is None:
                self.mesh = getattr(self._solver_ref, 'mesh', None)
            if self.fes is None:
                # Prefer global fes for unified structure
                self.fes = getattr(self._solver_ref, '_fes_global', None)
                if self.fes is None:
                    self.fes = getattr(self._solver_ref, 'fes', None)

    def _ensure_unified_fes(self) -> None:
        """Ensure unified FES exists, creating it if necessary."""
        if self.fes is not None:
            return

        if self.mesh is None:
            raise ValueError(
                "Cannot create unified FES: no mesh available. "
                "Provide mesh to constructor or ensure structures have mesh."
            )

        # Determine polynomial order from structures or solver_ref
        order = 3  # default
        if self._solver_ref is not None:
            order = getattr(self._solver_ref, 'order', order)
        
        # Try to get from first structure's fes
        for struct in self.structures:
            if struct.fes is not None:
                order = struct.fes.order
                break

        # Determine Dirichlet BC label
        bc = 'default'
        if self._solver_ref is not None:
            bc = getattr(self._solver_ref, 'bc', bc)

        from ngsolve import HCurl
        self.fes = HCurl(self.mesh, order=order, complex=True, dirichlet=bc)
        pr.debug(f"  Created unified FES: {self.fes.ndof} DOFs (order={order})")

    def _validate_mode_counts(self) -> None:
        """Validate consistent mode counts across all structures."""
        if not self.structures:
            self._n_modes_per_port = 1  # Default for empty structures
            return

        # Get reference mode count from first structure
        ref_n_modes = self.structures[0].n_port_modes
        
        # Handle None or invalid values
        if ref_n_modes is None or ref_n_modes < 1:
            ref_n_modes = 1

        for i, struct in enumerate(self.structures):
            n_modes = struct.n_port_modes
            if n_modes is None:
                n_modes = 1
            if n_modes != ref_n_modes:
                raise ValueError(
                    f"Inconsistent mode counts: structure 0 has {ref_n_modes} modes/port, "
                    f"but structure {i} has {n_modes} modes/port"
                )

        self._n_modes_per_port = ref_n_modes

    @staticmethod
    def _default_impedance(port: str, mode: int, freq: float) -> complex:
        return Z0

    def _assign_global_port_modes(self) -> None:
        """Assign global indices to each (structure, port, mode) combination."""
        offset = 0

        # (struct_idx, port_name, mode_idx) -> global_index
        self.port_mode_map: Dict[Tuple[int, str, int], int] = {}
        # global_index -> (struct_idx, port_name, mode_idx)
        self._global_to_local: Dict[int, Tuple[int, str, int]] = {}
        # (struct_idx, port_name) -> (start_global_idx, n_modes)
        self.port_to_mode_range: Dict[Tuple[int, str], Tuple[int, int]] = {}

        for struct_idx, struct in enumerate(self.structures):
            n_modes = struct.n_port_modes
            for port in struct.ports:
                start_idx = offset
                for mode_idx in range(n_modes):
                    self.port_mode_map[(struct_idx, port, mode_idx)] = offset
                    self._global_to_local[offset] = (struct_idx, port, mode_idx)
                    offset += 1
                self.port_to_mode_range[(struct_idx, port)] = (start_idx, n_modes)

        self.n_total_port_modes = offset

    def _compute_dof_offsets(self) -> None:
        """Compute DOF offsets for stacking structure solutions."""
        self._structure_dof_offsets: List[int] = []
        self._structure_full_dof_offsets: List[int] = []

        reduced_offset = 0
        full_offset = 0

        for struct in self.structures:
            self._structure_dof_offsets.append(reduced_offset)
            self._structure_full_dof_offsets.append(full_offset)
            reduced_offset += struct.r
            full_offset += (struct.n_full or struct.r)

        self._total_stacked_dofs = reduced_offset
        self._total_full_dofs = full_offset

    # =========================================================================
    # Connection Definition
    # =========================================================================

    def define_connections(
        self,
        connections: List[Conn],
        connection_signs: Optional[List[ConnSigns]] = None,
        validate: bool = True,
    ) -> "ConcatenatedSystem":
        """
        Define internal port connections.

        Parameters
        ----------
        connections : list
            List of ((structA, portA), (structB, portB)) tuples.
            All modes of connected ports are coupled mode-by-mode.
        connection_signs : list, optional
            Per-connection signs (signA, signB). Default: (+1, -1) for each.
        validate : bool
            Perform sanity checks if True.

        Returns
        -------
        self : ConcatenatedSystem
            For method chaining.
        """
        self.connections = list(connections)
        self.n_connections = len(self.connections)

        if connection_signs is None:
            self._connection_signs = [(+1.0, -1.0)] * self.n_connections
        else:
            if len(connection_signs) != self.n_connections:
                raise ValueError(
                    f"connection_signs length ({len(connection_signs)}) must match "
                    f"number of connections ({self.n_connections})."
                )
            self._connection_signs = [(float(a), float(b)) for a, b in connection_signs]

        # Identify internal vs external port-modes
        internal_set = set()
        for (sA_idx, pA), (sB_idx, pB) in self.connections:
            if validate:
                self._validate_connection((sA_idx, pA), (sB_idx, pB))

            for mode_idx in range(self._n_modes_per_port):
                internal_set.add(self.port_mode_map[(sA_idx, pA, mode_idx)])
                internal_set.add(self.port_mode_map[(sB_idx, pB, mode_idx)])

        self._internal_port_modes = sorted(internal_set)
        self._external_port_modes = sorted(set(range(self.n_total_port_modes)) - internal_set)
        self._n_internal = len(self._internal_port_modes)
        self._n_external = len(self._external_port_modes)

        # Build permutation matrix: reorder to [internal | external]
        perm = self._internal_port_modes + self._external_port_modes
        P = np.zeros((self.n_total_port_modes, self.n_total_port_modes))
        for new_pos, old_pos in enumerate(perm):
            P[new_pos, old_pos] = 1.0
        self._permutation = P

        # Build external port naming
        self._external_port_mode_names = []
        self._external_port_mode_map = {}
        for i, global_idx in enumerate(self._external_port_modes):
            struct_idx, orig_port, mode_idx = self._global_to_local[global_idx]
            new_name = f"port{i // self._n_modes_per_port + 1}({mode_idx + 1})"
            self._external_port_mode_names.append((orig_port, mode_idx))
            self._external_port_mode_map[new_name] = (struct_idx, orig_port, mode_idx)

        if validate:
            self._validate_connections()

        return self

    def _validate_connection(self, portA: Tuple[int, str], portB: Tuple[int, str]) -> None:
        """Validate a single connection."""
        sA_idx, pA = portA
        sB_idx, pB = portB

        if (sA_idx, pA) not in self.port_to_mode_range:
            raise KeyError(f"Unknown port: ({sA_idx}, '{pA}')")
        if (sB_idx, pB) not in self.port_to_mode_range:
            raise KeyError(f"Unknown port: ({sB_idx}, '{pB}')")

        n_modes_A = self.port_to_mode_range[(sA_idx, pA)][1]
        n_modes_B = self.port_to_mode_range[(sB_idx, pB)][1]
        if n_modes_A != n_modes_B:
            raise ValueError(
                f"Mode count mismatch: ({sA_idx}, '{pA}') has {n_modes_A} modes, "
                f"({sB_idx}, '{pB}') has {n_modes_B} modes"
            )

    def _validate_connections(self) -> None:
        """Validate all connections."""
        for j, ((sA, pA), (sB, pB)) in enumerate(self.connections):
            if sA == sB and pA == pB:
                raise ValueError(f"Connection {j} connects port '{pA}' to itself")

    def _build_incidence_matrix(self) -> np.ndarray:
        """Build incidence matrix F for Kirchhoff coupling."""
        n_int = self._n_internal
        n_mode_connections = self.n_connections * self._n_modes_per_port

        F = np.zeros((n_int, n_mode_connections))
        int_pos = {g: i for i, g in enumerate(self._internal_port_modes)}

        col = 0
        for ((sA_idx, pA), (sB_idx, pB)), (sgnA, sgnB) in zip(
            self.connections, self._connection_signs
        ):
            for mode_idx in range(self._n_modes_per_port):
                gA = self.port_mode_map[(sA_idx, pA, mode_idx)]
                gB = self.port_mode_map[(sB_idx, pB, mode_idx)]
                F[int_pos[gA], col] = sgnA
                F[int_pos[gB], col] = sgnB
                col += 1

        return F

    # =========================================================================
    # Coupling
    # =========================================================================

    def couple(self, rcond_pinv: float = 1e-12, rcond_null: float = 1e-12) -> "ConcatenatedSystem":
        """
        Perform structure coupling via null-space projection.

        This creates the unified system by eliminating internal ports
        through Kirchhoff constraints.

        Returns
        -------
        self : ConcatenatedSystem
            For method chaining.
        """
        if self.connections is None:
            raise ValueError("Must call define_connections() first")

        # Block-diagonal assembly of uncoupled structures
        A_blocks = [np.asarray(s.Ard) for s in self.structures]
        B_blocks = [np.asarray(s.Brd) for s in self.structures]

        A_blk = sl.block_diag(*A_blocks).astype(complex, copy=False)
        B_blk = sl.block_diag(*B_blocks).astype(complex, copy=False)

        # Permute to [internal | external]
        B_perm = B_blk @ self._permutation.T
        B_int = B_perm[:, :self._n_internal]
        B_ext = B_perm[:, self._n_internal:]

        # Build and apply Kirchhoff constraints
        F = self._build_incidence_matrix()
        C = B_int @ F

        CtC = C.T.conj() @ C
        CtC_inv = sl.pinvh(CtC)

        I = np.eye(A_blk.shape[0], dtype=complex)
        K = I - C @ CtC_inv @ C.T.conj()

        # Null space basis
        M = sl.null_space(C.T.conj(), rcond=rcond_null).astype(complex, copy=False)
        if M.size == 0:
            raise RuntimeError("Null space empty; constraints overconstrained.")

        KM = K @ M

        # Project system onto constraint-satisfying subspace
        self.A_coupled = KM.T.conj() @ A_blk @ KM
        self.B_coupled = KM.T.conj() @ B_ext
        self.W_coupled = KM

        # Ensure Hermitian symmetry
        self.A_coupled = 0.5 * (self.A_coupled + self.A_coupled.T.conj())

        pr.info(f"\nCoupled unified system: {A_blk.shape[0]} -> {self.A_coupled.shape[0]} DOFs")
        pr.debug(f"  External port-modes: {self._n_external}")
        pr.debug(f"  Internal port-modes (eliminated): {self._n_internal}")
        pr.debug(f"  Connections: {self.n_connections}")

        if hasattr(self, '_mor_ref') and self._mor_ref:
             self._mor_ref._A_r_global = self.A_coupled
             self._mor_ref._B_r_global = self.B_coupled
             self._mor_ref._W_r_global = self.W_coupled

        # Automatic save after state is synchronized
        if hasattr(self, '_solver_ref') and self._solver_ref and hasattr(self._solver_ref, '_project_ref'):
            if self._solver_ref._project_ref:
                self._solver_ref._project_ref.save()

        return self


    def calculate_resonant_modes(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors for the coupled system."""
        # Check cache
        cache_key = tuple(sorted(kwargs.items()))
        if cache_key in self._resonant_mode_cache:
            return self._resonant_mode_cache[cache_key]

        from solvers.frequency_domain import FrequencyDomainSolver
        eigenvalues, eigenvectors = np.linalg.eigh(self.A_coupled)
        res = FrequencyDomainSolver._filter_eigenvalues(eigenvalues, eigenvectors, **kwargs)
        
        # Update cache
        self._resonant_mode_cache[cache_key] = res
        return res

    def get_eigenvalues(self, **kwargs):
        """Deprecated alias for calculate_resonant_modes."""
        import warnings
        warnings.warn("get_eigenvalues() is deprecated. Use calculate_resonant_modes() instead.",
                      DeprecationWarning, stacklevel=2)
        res = self.calculate_resonant_modes(**kwargs)
        if isinstance(res, dict):
            return {k: v[0] for k, v in res.items()}
        return res[0]

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """Standardized API for eigenmode computation with auto-save."""
        res = self.calculate_resonant_modes(**kwargs)
        
        # Populate the eigen caches so save_eigenmodes can find them
        self._init_eigen_cache()
        if isinstance(res, dict):
            for d, (eigs, vecs) in res.items():
                self._eigenvalues_cache[d] = eigs
                self._eigenvectors_cache[d] = vecs
        else:
            eigs, vecs = res
            domain_key = 'global'
            self._eigenvalues_cache[domain_key] = eigs
            self._eigenvectors_cache[domain_key] = vecs
        
        if _auto_save:
            self._auto_save_eigenmodes(res, **kwargs)
        return res

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        if self._solver_ref is None:
            return
        
        # Resolve project path from solver_ref
        project_path = None
        if hasattr(self._solver_ref, '_project_path'):
            project_path = self._solver_ref._project_path
        elif hasattr(self._solver_ref, 'solver') and hasattr(self._solver_ref.solver, '_project_path'):
            project_path = self._solver_ref.solver._project_path
        
        if project_path is None:
            return

        # Determine path: eigenmode/foms/concat or eigenmode/fom/rom/concat etc.
        # We look at the solver_ref to see where the FOM results are stored
        # This is a bit heuristic but aligns with the hierarchy
        project_path = Path(project_path)
        
        # Default for ConcatenatedSystem (usually multi-solid)
        sub_path = "foms/concat"
        
        # Check if we are in a ROM hierarchy
        from rom.reduction import ModelOrderReduction
        if isinstance(self._solver_ref, ModelOrderReduction):
            if self._solver_ref.n_domains == 1:
                sub_path = "fom/rom/concat"
            else:
                sub_path = "foms/roms/concat"
        
        save_path = project_path / "fds" / sub_path / "eigenmodes"
        try:
            self.save_eigenmodes(save_path, **kwargs)
        except Exception as e:
            print(f"Warning: Could not auto-save eigenmodes for ConcatenatedSystem: {e}")

    # =========================================================================
    # BaseEMSolver Interface
    # =========================================================================

    @property
    def n_ports(self) -> int:
        return self._n_external

    @property
    def n_external_ports(self) -> int:
        return self._n_external // self._n_modes_per_port

    @property
    def n_modes_per_port(self) -> int:
        return self._n_modes_per_port

    @property
    def ports(self) -> List[str]:
        return list(self._external_port_mode_map.keys())

    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        if port not in self._external_port_mode_map:
            raise KeyError(f"Port '{port}' not found. Available: {self.ports}")
        _, orig_port, orig_mode = self._external_port_mode_map[port]
        return self._port_impedance_func(orig_port, orig_mode, freq)

    def _get_impedance_matrix(self, freq: float) -> np.ndarray:
        Z0_diag = []
        for global_idx in self._external_port_modes:
            struct_idx, port_name, mode_idx = self._global_to_local[global_idx]
            Zw = self._port_impedance_func(port_name, mode_idx, freq)
            Z0_diag.append(Zw)
        return np.diag(Z0_diag)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Union[str, Path]):
        """
        Save ConcatenatedSystem data to disk.
        
        Saves coupled matrices (A, B, W) to separate files in matrices/
        and S/Z parameters, snapshots, and eigenmodes to their respective folders.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Subfolders
        s_path_dir = path / "s"
        z_path_dir = path / "z"
        snap_path_dir = path / "snapshots"
        eig_path_dir = path / "eigenmodes"
        for p in [s_path_dir, z_path_dir, snap_path_dir, eig_path_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # 1. Save coupled matrices to modular files
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(mat_path / "A.h5", "a") as fa, \
             h5py.File(mat_path / "B.h5", "a") as fb, \
             h5py.File(mat_path / "W.h5", "a") as fw:
            H5Serializer.save_dataset(fa, "data", self.A_coupled)
            H5Serializer.save_dataset(fb, "data", self.B_coupled)
            if self.W_coupled is not None:
                H5Serializer.save_dataset(fw, "data", self.W_coupled)

        # 2. Save S and Z results
        if self._Z_matrix is not None:
            with h5py.File(z_path_dir / "z.h5", "a") as f:
                H5Serializer.save_dataset(f, "data", self._Z_matrix)
        if self._S_matrix is not None:
            with h5py.File(s_path_dir / "s.h5", "a") as f:
                H5Serializer.save_dataset(f, "data", self._S_matrix)

        # 3. Save frequencies and snapshots
        with h5py.File(snap_path_dir / "snapshots.h5", "a") as f:
            if hasattr(self, 'frequencies') and self.frequencies is not None:
                H5Serializer.save_dataset(f, "frequencies", self.frequencies)
            if self._snapshots is not None:
                # Concatenated snapshots are stored as a single array
                if "coupled_snapshots" in f: del f["coupled_snapshots"]
                H5Serializer.save_dataset(f, "coupled_snapshots", self._snapshots)

        # 4. Save eigenmodes
        self.save_eigenmodes()

        metadata = {
            "n_structures": self.n_structures,
            "domains": self.domains,
            "n_connections": self.n_connections,
            "n_internal": self._n_internal,
            "n_external": self._n_external,
            "n_modes_per_port": self._n_modes_per_port,
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)

    @classmethod
    def load(cls, path: Union[str, Path], solver_ref=None) -> ConcatenatedSystem:
        """Load ConcatenatedSystem from disk."""
        path = Path(path)
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create skeleton
        cs = cls.__new__(cls)
        cs._solver_ref = solver_ref
        cs.n_structures = metadata["n_structures"]
        cs.domains = metadata["domains"]
        cs.n_connections = metadata["n_connections"]
        cs._n_internal = metadata["n_internal"]
        cs._n_external = metadata["n_external"]
        cs._n_modes_per_port = metadata["n_modes_per_port"]
        
        # Initialize caches
        cs._resonant_mode_cache = {}
        
        # Initialize result dicts for PlotMixin
        cs._S_dict = None
        cs._Z_dict = None
        
        # 1. Load matrices from modular files or legacy matrices.h5
        mat_path = path / "matrices"
        if mat_path.exists():
            with h5py.File(mat_path / "A.h5", "r") as f:
                cs.A_coupled = H5Serializer.load_dataset(f["data"])
            with h5py.File(mat_path / "B.h5", "r") as f:
                cs.B_coupled = H5Serializer.load_dataset(f["data"])
            if (mat_path / "W.h5").exists():
                with h5py.File(mat_path / "W.h5", "r") as f:
                    cs.W_coupled = H5Serializer.load_dataset(f["data"])
        elif (path / "matrices.h5").exists(): # Legacy
            with h5py.File(path / "matrices.h5", "r") as f:
                cs.A_coupled = H5Serializer.load_dataset(f["A_coupled"])
                cs.B_coupled = H5Serializer.load_dataset(f["B_coupled"])
                cs.W_coupled = H5Serializer.load_dataset(f["W_coupled"])

        # 2. Load Z and S
        cs.frequencies = None
        cs._Z_matrix = None
        cs._S_matrix = None
        
        z_path = path / "z" / "z.h5"
        if not z_path.exists(): z_path = path / "z.h5"
        if z_path.exists():
            with h5py.File(z_path, "r") as f:
                cs._Z_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None
                
        s_path = path / "s" / "s.h5"
        if not s_path.exists(): s_path = path / "s.h5"
        if s_path.exists():
            with h5py.File(s_path, "r") as f:
                cs._S_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None

        # 3. Load snapshots and frequencies
        cs._snapshots = {}
        snap_path = path / "snapshots" / "snapshots.h5"
        if not snap_path.exists(): snap_path = path / "snapshots.h5"
        if snap_path.exists():
            with h5py.File(snap_path, "r") as f:
                if cs.frequencies is None:
                    cs.frequencies = H5Serializer.load_dataset(f["frequencies"]) if "frequencies" in f else None
                
                if "field_snapshots" in f:
                    group = f["field_snapshots"]
                    for domain in metadata.get("structure_domains", []):
                        if domain in group:
                            cs._snapshots[domain] = H5Serializer.load_dataset(group[domain])

        # 4. Load eigenmodes
        eig_path = path / "eigenmodes" / "eigenmodes.h5"
        if eig_path.exists():
            # We don't have a cache on CS, but we can load them if needed.
            # For now, we mainly want them saved for the user.
            pass

        return cs

    # =========================================================================
    # Frequency Domain Solution
    # =========================================================================

    def solve(
        self,
        fmin: float = None,
        fmax: float = None,
        nsamples: int = None,
        config: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Solve the unified coupled system over a frequency range.

        Supports passing arguments directly or via a 'config' dictionary.
        Individual keyword arguments override the config dictionary.

        Parameters
        ----------
        fmin, fmax : float, optional
            Frequency range [GHz]
        nsamples : int, optional
            Number of frequency samples
        config : dict, optional
            Dictionary containing solve parameters
        **kwargs :
            Individual solve parameters (compute_s_params, solver_type, rerun, etc.)
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
        compute_s_params = cfg.get('compute_s_params', True)
        solver_type = cfg.get('solver_type', 'auto')
        verbose = cfg.get('verbose', False)

        # Set verbosity level
        pr.set_verbosity(verbose)

        # 4. Initialize frequency array
        self.frequencies = np.linspace(fmin, fmax, nsamples) * 1e9

        if self.A_coupled is None:
            raise ValueError("Must call couple() first")

        # --- Rerun protection ---
        has_results = (self._Z_matrix is not None)
        rerun = cfg.get('rerun', False)

        # Check disk if in-memory is missing
        if not has_results and not rerun and self._solver_ref and hasattr(self._solver_ref, '_project_path'):
            # Concatenated results are usually in fds/foms/concat or fds/fom/rom/concat
            # We check the most likely locations
            project_path = Path(self._solver_ref._project_path)
            
            # Use same logic as auto_save_eigenmodes to find sub_path
            sub_path = "foms/concat"
            from rom.reduction import ModelOrderReduction
            if isinstance(self._solver_ref, ModelOrderReduction):
                if self._solver_ref.n_domains == 1:
                    sub_path = "fom/rom/concat"
                else:
                    sub_path = "foms/roms/concat"
            
            concat_dir = project_path / "fds" / sub_path
            z_path = concat_dir / "z" / "z.h5"
            if z_path.exists():
                try:
                    with h5py.File(z_path, "r") as f:
                        self._Z_matrix = H5Serializer.load_dataset(f["data"])
                    s_path = concat_dir / "s" / "s.h5"
                    if s_path.exists():
                        with h5py.File(s_path, "r") as f:
                            self._S_matrix = H5Serializer.load_dataset(f["data"])
                    
                    # Load frequencies
                    snap_path = concat_dir / "snapshots" / "snapshots.h5"
                    if not snap_path.exists(): snap_path = concat_dir / "snapshots.h5"
                    if snap_path.exists():
                        with h5py.File(snap_path, "r") as f:
                            self.frequencies = H5Serializer.load_dataset(f["frequencies"])

                    has_results = True
                    pr.info(f"  Loaded existing Concatenated results from {concat_dir}")
                except Exception as e:
                    pr.warning(f"  Could not load existing Concatenated results: {e}")

        if has_results and not rerun:
            import warnings
            warnings.warn(
                "Results already exist for this Concatenated system. "
                "To overwrite, call solve(..., rerun=True).",
                UserWarning,
                stacklevel=2,
            )
            return {
                "frequencies": self.frequencies,
                "Z": self._Z_matrix,
                "S": self._S_matrix if compute_s_params else None,
                "Z_dict": self.Z_dict,
                "S_dict": self.S_dict if compute_s_params else None,
            }
        n_ext = self._n_external
        r = self.A_coupled.shape[0]

        if solver_type == 'auto':
            solver_type = 'iterative' if r >= self.ITERATIVE_SIZE_THRESHOLD else 'direct'
            pr.debug(f"  Solver: {solver_type} (system size {r})")

        self._Z_matrix = np.zeros((nsamples, n_ext, n_ext), dtype=complex)
        omegas = 2 * np.pi * self.frequencies

        import time
        t0 = time.time()

        if solver_type == 'direct':
            x_all = self._solve_direct(omegas, n_ext, r)
        else:
            x_all = self._solve_iterative(omegas, n_ext, r)

        pr.done(f"  Solve: {time.time() - t0:.3f}s ({nsamples} frequencies)")

        self._snapshots = np.array(x_all)

        if compute_s_params:
            self._compute_s_from_z()
        self._invalidate_cache()

        # Automatic save after simulation
        if hasattr(self, '_solver_ref') and self._solver_ref and hasattr(self._solver_ref, '_project_ref'):
            if self._solver_ref._project_ref:
                self._solver_ref._project_ref.save()

        return {
            "frequencies": self.frequencies,
            "Z": self._Z_matrix,
            "S": self._S_matrix if compute_s_params else None,
            "Z_dict": self.Z_dict,
            "S_dict": self.S_dict if compute_s_params else None,
        }

    def _solve_direct(self, omegas: np.ndarray, n_ext: int, r: int) -> List[np.ndarray]:
        """Direct eigendecomposition-based solve."""
        eigenvalues, V = np.linalg.eigh(self.A_coupled)
        C = V.T.conj() @ self.B_coupled
        D = self.B_coupled.T.conj() @ V

        d = 1.0 / (eigenvalues[None, :] - omegas[:, None] ** 2)

        x_all = []
        for k in range(len(omegas)):
            self._Z_matrix[k] = 1j * omegas[k] * (D * d[k, :]) @ C
            x_all.append(omegas[k] * V @ (d[k, :, None] * C))

        return x_all

    def _solve_iterative(self, omegas: np.ndarray, n_ext: int, r: int) -> List[np.ndarray]:
        """GMRES-based iterative solve."""
        I_ext = np.eye(n_ext, dtype=complex)
        x_all = []
        failures = 0

        num_freq = len(omegas)
        pr.running(f"  Solving {num_freq} frequencies...")
        
        # Progress reporting logic
        report_interval = max(1, num_freq // 10) # 10% steps
        
        for k, omega in enumerate(omegas):
            if (k + 1) % report_interval == 0 or k == 0 or k == num_freq - 1:
                pr.debug(f"    - Frequency {k+1}/{num_freq} ({self.frequencies[k]/1e9:.4f} GHz)")
                
            lhs = self.A_coupled - omega ** 2 * np.eye(r, dtype=complex)
            rhs = omega * self.B_coupled @ I_ext

            lhs_sp = sp.csr_matrix(lhs)
            x = np.zeros_like(rhs)
            for col in range(n_ext):
                x[:, col], info = spla.gmres(lhs_sp, rhs[:, col])
                if info != 0:
                    failures += 1

            x_all.append(x)
            self._Z_matrix[k] = 1j * self.B_coupled.T.conj() @ x

        if failures > 0:
            pr.warning(f"{failures} GMRES solves did not converge")

        return x_all

    # =========================================================================
    # Field Reconstruction - Unified Structure
    # =========================================================================

    @property
    def has_snapshots(self) -> bool:
        return self._snapshots is not None

    def can_reconstruct(self) -> bool:
        """Check if unified field reconstruction is possible."""
        if not self.has_snapshots:
            return False
        if self.W_coupled is None:
            return False
        if self.fes is None:
            return False
        if self.mesh is None:
            return False
        return all(s.can_reconstruct() for s in self.structures)

    def _reconstruct_field(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int = 0
    ) -> GridFunction:
        """
        Reconstruct the unified field over the entire structure.
        """
        if not self.has_snapshots:
            raise ValueError("No snapshots available. Call solve() first.")

        if self.mesh is None:
            raise ValueError("No mesh available. Provide mesh to constructor.")

        # Ensure we have a unified FES (create if needed)
        self._ensure_unified_fes()

        # Validate excitation port
        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found. Available: {self.ports}")
        col_idx = self.ports.index(excitation_port)

        # Map from coupled to uncoupled stacked coordinates
        x_coupled = self._snapshots[freq_idx, :, col_idx]
        x_uncoupled = self.W_coupled @ x_coupled

        # Create unified GridFunction on global FES
        E_gf = GridFunction(self.fes, complex=True)

        for struct_idx, struct in enumerate(self.structures):
            # Check that structure can reconstruct
            if not struct.can_reconstruct():
                raise ValueError(
                    f"Structure {struct_idx} ({struct.domain}) cannot reconstruct. "
                    f"W: {'set' if struct.W is not None else 'MISSING'}, "
                    f"Q_L_inv: {'set' if struct.Q_L_inv is not None else 'MISSING'}, "
                    f"fes: {'set' if struct.fes is not None else 'MISSING'}"
                )

            # Extract this structure's reduced solution
            start_r = self._structure_dof_offsets[struct_idx]
            x_reduced = x_uncoupled[start_r:start_r + struct.r]

            # Reconstruct full-order solution for this domain
            x_full_local = struct.reconstruct(x_reduced)

            # Create per-domain GridFunction
            E_local = GridFunction(struct.fes, complex=True)
            E_local.vec.FV().NumPy()[:] = x_full_local

            # Transfer to global GridFunction using definedon
            domain_region = self.mesh.Materials(struct.domain)
            E_gf.Set(E_local, definedon=domain_region)

        return E_gf

    def plot_field(
        self,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        excitation_mode: int = 0,
        component: Literal['real', 'imag', 'abs'] = 'abs',
        field_type: Literal['E', 'H'] = 'E',
        clipping: Optional[Dict] = None,
        euler_angles: Optional[List] = [45, -45, 0],
        **kwargs
    ) -> None:
        """
        Visualize field over the entire unified structure.

        Parameters
        ----------
        freq_idx : int
            Frequency index
        excitation_port : str, optional
            Port used for excitation. If None, uses first port.
        excitation_mode : int
            Mode index for excitation
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

        if self._snapshots is None:
            raise ValueError("No snapshots available.")

        freq = self.frequencies[freq_idx]
        omega = 2 * np.pi * freq

        # Default excitation
        if excitation_port is None:
            if not self.ports:
                raise ValueError("No external ports available")
            excitation_port = self.ports[0]

        if excitation_port not in self.ports:
            raise ValueError(f"Port '{excitation_port}' not found. Available: {self.ports}")

        pr.info(f"\nField visualization at f = {freq / 1e9:.4f} GHz")
        pr.info(f"  Excitation: {excitation_port}, mode {excitation_mode}")
        pr.debug(f"  Unified structure with {self.n_structures} domains: {self.domains}")

        # Reconstruct unified field
        E_gf = self._reconstruct_field(freq_idx, excitation_port, excitation_mode)

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
            plot_name = f"|{field_label}|"
        elif component == 'real':
            cf_plot = field_cf.real
            plot_name = f"Re({field_label})"
        elif component == 'imag':
            cf_plot = field_cf.imag
            plot_name = f"Im({field_label})"
        else:
            raise ValueError(f"Invalid component: {component}")

        pr.debug(f"  Plotting: {plot_name}")

        draw_kwargs = kwargs.copy()
        if clipping:
            draw_kwargs['clipping'] = clipping

        if euler_angles:
            draw_kwargs['euler_angles'] = euler_angles

        Draw(BoundaryFromVolumeCF(cf_plot), self.mesh, plot_name, **draw_kwargs)

    def plot_field_at_frequency(self, freq: float, **kwargs) -> None:
        """
        Plot field at specific frequency (Hz).

        Parameters
        ----------
        freq : float
            Frequency in Hz
        **kwargs
            Additional arguments passed to plot_field()
        """
        if self.frequencies is None:
            raise ValueError("No solution available.")
        freq_idx = int(np.argmin(np.abs(self.frequencies - freq)))
        actual_freq = self.frequencies[freq_idx]
        if abs(actual_freq - freq) / max(freq, 1e-10) > 0.01:
            pr.debug(f"  Note: Using nearest frequency {actual_freq / 1e9:.4f} GHz")
        self.plot_field(freq_idx=freq_idx, **kwargs)

    def get_reconstruction_info(self) -> Dict:
        """Get information about field reconstruction capability."""
        info = {
            'can_reconstruct': self.can_reconstruct(),
            'has_snapshots': self.has_snapshots,
            'has_mesh': self.mesh is not None,
            'has_fes': self.fes is not None,
            'structures': []
        }
        for i, struct in enumerate(self.structures):
            info['structures'].append({
                'index': i,
                'domain': struct.domain,
                'can_reconstruct': struct.can_reconstruct(),
                'has_W': struct.W is not None,
                'has_Q_L_inv': struct.Q_L_inv is not None,
                'has_fes': struct.fes is not None,
            })
        return info

    # =========================================================================
    # Eigenvalue Analysis
    # =========================================================================

    @staticmethod
    def _filter_eigenvalues(
        eigenvalues: np.ndarray,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> np.ndarray:
        if min_eigenvalue is None:
            min_eigenvalue = ConcatenatedSystem.DEFAULT_MIN_EIGENVALUE

        eigs = np.sort(np.real(eigenvalues))
        if filter_static:
            eigs = eigs[eigs > min_eigenvalue]
        if n_modes is not None:
            eigs = eigs[:n_modes]
        return eigs

    def get_eigenvalues(
        self,
        domain: str = None,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> np.ndarray:
        """Get eigenvalues of the unified coupled system."""
        if self.A_coupled is None:
            raise ValueError("Must call couple() first")

        if domain is None or domain == 'global':
            raw = np.linalg.eigvalsh(self.A_coupled)
        else:
            for struct in self.structures:
                if struct.domain == domain:
                    raw = np.linalg.eigvalsh(struct.Ard)
                    break
            else:
                raise KeyError(f"Domain '{domain}' not found")

        return self._filter_eigenvalues(raw, filter_static, min_eigenvalue, n_modes)

    def get_resonant_frequencies(
        self,
        n_modes: int = None,
        fmin: float = None,
        filter_static: bool = True
    ) -> np.ndarray:
        """Get resonant frequencies of the unified structure."""
        if fmin is not None:
            min_eigenvalue = (2 * np.pi * fmin * 1e9) ** 2
            filter_static = True
        else:
            min_eigenvalue = self.DEFAULT_MIN_EIGENVALUE if filter_static else None

        eigs = self.get_eigenvalues(
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue
        )
        eigs_pos = eigs[eigs > 0]
        freqs = np.sqrt(eigs_pos) / (2 * np.pi)

        if n_modes is not None:
            freqs = freqs[:n_modes]
        return freqs

    # =========================================================================
    # Backward-compatible ROM accessor
    # =========================================================================

    @property
    def rom(self):
        """
        Backward-compatible accessor for further reduction.

        .. deprecated::
            Use ``concat.reduce()`` instead.
        """
        if not hasattr(self, '_rom_cache') or self._rom_cache is None:
            import warnings
            warnings.warn(
                "ConcatenatedSystem.rom is deprecated. Use concat.reduce() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._rom_cache = self.reduce()
        return self._rom_cache

    # =========================================================================
    # Further Reduction
    # =========================================================================

    def reduce(self, tol: float = 1e-6, max_rank: Optional[int] = None) -> 'ReducedConcatenatedSystem':
        """
        Further reduce this system via POD on solution snapshots.

        Parameters
        ----------
        tol : float
            SVD truncation tolerance
        max_rank : int, optional
            Maximum rank

        Returns
        -------
        ReducedConcatenatedSystem
            Further-reduced unified system
        """
        return reduce_concatenated_system(self, tol=tol, max_rank=max_rank)

    # =========================================================================
    # Diagnostics and Info
    # =========================================================================

    @property
    def coupled_dofs(self) -> int:
        return self.A_coupled.shape[0] if self.A_coupled is not None else 0

    @property
    def has_solution(self) -> bool:
        return self.frequencies is not None

    def get_coupled_dimensions(self) -> Dict[str, int]:
        return {
            'n_structures': self.n_structures,
            'total_uncoupled_dofs': sum(s.r for s in self.structures),
            'coupled_dofs': self.coupled_dofs,
            'n_internal_port_modes': self._n_internal,
            'n_external_port_modes': self._n_external,
            'n_modes_per_port': self._n_modes_per_port,
            'n_connections': self.n_connections or 0,
        }

    def verify_kirchhoff(self, x_uncoupled: np.ndarray) -> float:
        """Verify constraint satisfaction on an uncoupled stacked state vector."""
        if self.connections is None:
            raise ValueError("Must call define_connections() first")

        B_blocks = [np.asarray(s.Brd) for s in self.structures]
        B_blk = sl.block_diag(*B_blocks).astype(complex, copy=False)
        B_perm = B_blk @ self._permutation.T
        B_int = B_perm[:, :self._n_internal]
        F = self._build_incidence_matrix()

        viol = F.T @ (B_int.T.conj() @ x_uncoupled)
        return float(np.max(np.abs(viol)))

    def print_info(self) -> None:
        """Print unified system information."""
        pr.info("\n" + "=" * 60)
        pr.info("Unified Concatenated System")
        pr.info("=" * 60)

        pr.info(f"\nComponent structures ({self.n_structures}):")
        for i, s in enumerate(self.structures):
            recon = "✓" if s.can_reconstruct() else "✗"
            pr.info(f"  [{i}] {s.domain}: r={s.r}, n_full={s.n_full}, ports={s.ports} [recon:{recon}]")

        if self.connections:
            pr.info(f"\nInternal connections ({len(self.connections)}):")
            for (sA, pA), (sB, pB) in self.connections:
                pr.info(f"  structure[{sA}].{pA} <-> structure[{sB}].{pB}")

        dims = self.get_coupled_dimensions()
        pr.debug(f"\nUnified system dimensions:")
        pr.debug(f"  Total uncoupled DOFs: {dims['total_uncoupled_dofs']}")
        pr.debug(f"  Coupled DOFs: {dims['coupled_dofs']}")
        pr.debug(f"  External port-modes: {dims['n_external_port_modes']}")
        pr.debug(f"  Internal port-modes: {dims['n_internal_port_modes']}")

        pr.info(f"\nExternal ports: {self.ports}")

        recon_ready = self.can_reconstruct()
        pr.debug(f"\nField reconstruction: {'Ready' if recon_ready else 'Not available'}")
        if not recon_ready:
            info = self.get_reconstruction_info()
            if not info['has_mesh']:
                pr.debug("  - Missing: mesh")
            if not info['has_fes']:
                pr.debug("  - Missing: global fes")
            if not info['has_snapshots']:
                pr.debug("  - Missing: snapshots (call solve())")
            for s_info in info['structures']:
                if not s_info['can_reconstruct']:
                    pr.debug(f"  - Structure {s_info['index']} ({s_info['domain']}): "
                          f"W={'✓' if s_info['has_W'] else '✗'}, "
                          f"Q_L_inv={'✓' if s_info['has_Q_L_inv'] else '✗'}, "
                          f"fes={'✓' if s_info['has_fes'] else '✗'}")
        else:
            if self.mesh is not None:
                pr.debug(f"  Mesh: {self.mesh.ne} elements")
            if self.fes is not None:
                pr.debug(f"  FES: {self.fes.ndof} DOFs")

        if self.frequencies is not None:
            print(f"\nSolution:")
            print(f"  Range: {self.frequencies[0] / 1e9:.4f} - {self.frequencies[-1] / 1e9:.4f} GHz")
            print(f"  Samples: {len(self.frequencies)}")

        print("=" * 60)


class ReducedConcatenatedSystem(ConcatenatedSystem):
    """
    Further-reduced unified system via POD.

    Created by calling reduce() on a ConcatenatedSystem after solve().
    """

    def __init__(
        self,
        parent: ConcatenatedSystem,
        A_reduced: np.ndarray,
        B_reduced: np.ndarray,
        W_reduction: np.ndarray,
        singular_values: np.ndarray,
    ):
        BaseEMSolver.__init__(self)

        # Copy all parent state
        self.structures = parent.structures
        self.n_structures = parent.n_structures
        self._port_impedance_func = parent._port_impedance_func
        self._solver_ref = parent._solver_ref
        self.mesh = parent.mesh
        self.fes = parent.fes

        self.port_mode_map = parent.port_mode_map
        self.port_to_mode_range = parent.port_to_mode_range
        self._global_to_local = parent._global_to_local
        self.n_total_port_modes = parent.n_total_port_modes
        self._n_modes_per_port = parent._n_modes_per_port

        self.connections = parent.connections
        self.n_connections = parent.n_connections
        self._connection_signs = parent._connection_signs

        self._internal_port_modes = parent._internal_port_modes
        self._external_port_modes = parent._external_port_modes
        self._external_port_mode_names = parent._external_port_mode_names
        self._external_port_mode_map = parent._external_port_mode_map
        self._permutation = parent._permutation
        self._n_internal = parent._n_internal
        self._n_external = parent._n_external

        self._structure_dof_offsets = parent._structure_dof_offsets
        self._structure_full_dof_offsets = parent._structure_full_dof_offsets
        self._total_stacked_dofs = parent._total_stacked_dofs
        self._total_full_dofs = parent._total_full_dofs

        self.domains = parent.domains

        # Reduced system matrices
        self.A_coupled = np.asarray(A_reduced).astype(complex, copy=False)
        self.B_coupled = np.asarray(B_reduced).astype(complex, copy=False)

        # Projection matrices
        # W_reduction: maps from parent coupled coords to this reduced level
        self._W_this_level = np.asarray(W_reduction).astype(complex, copy=False)
        # Combined projection: uncoupled stacked -> this reduced level
        self.W_coupled = parent.W_coupled @ W_reduction
        # Store parent's W_coupled for multi-level reconstruction
        self._parent_W_coupled = parent.W_coupled

        self._singular_values = np.asarray(singular_values)
        self._parent = parent
        self._reduction_level = getattr(parent, "_reduction_level", 0) + 1
        self._parent_coupled_dofs = parent.A_coupled.shape[0] if parent.A_coupled is not None else None

        # Snapshot storage (populated by solve())
        self._snapshots = None

        # Cache for interface scaling factors
        self._interface_scale_cache: Dict[Tuple[int, str, int], np.ndarray] = {}

    def _compute_interface_scaling_factors(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int = 0
    ) -> np.ndarray:
        """
        Compute scaling factors to enforce field continuity at interfaces.
        
        Domain 0 is the reference (scale = 1). For each connection, we compute
        a scaling factor that makes the field magnitude continuous at the 
        shared interface. Scales are applied cumulatively along the chain.
        
        Parameters
        ----------
        freq_idx : int
            Frequency index in the solution snapshots
        excitation_port : str
            Name of the excited port
        excitation_mode : int
            Mode index of excitation
            
        Returns
        -------
        scales : ndarray of shape (n_structures,)
            Complex scaling factors for each structure. Structure 0 has scale=1.
        """
        from ngsolve import Integrate, BoundaryFromVolumeCF, Norm, InnerProduct, Conj
        
        if self.n_structures == 1:
            return np.array([1.0 + 0j])
        
        # Check cache
        cache_key = (freq_idx, excitation_port, excitation_mode)
        if cache_key in self._interface_scale_cache:
            return self._interface_scale_cache[cache_key]
        
        # Validate excitation port
        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found. Available: {self.ports}")
        col_idx = self.ports.index(excitation_port)
        
        # Get uncoupled solution
        x_coupled = self._snapshots[freq_idx, :, col_idx]
        x_uncoupled = self.W_coupled @ x_coupled
        
        scales = self._compute_vector_scaling_factors(x_uncoupled)
        
        # Cache the result
        self._interface_scale_cache[cache_key] = scales
        
        return scales
        
    def _compute_vector_scaling_factors(self, x_uncoupled: np.ndarray) -> np.ndarray:
        """Compute interface scaling factors for a specific uncoupled vector."""
        if self.n_structures == 1:
            return np.array([1.0 + 0j])
            
        gf_list = []
        x_full_list = []
        x_reduced_list = []
        for struct_idx, struct in enumerate(self.structures):
            start_r = self._structure_dof_offsets[struct_idx]
            x_reduced = x_uncoupled[start_r:start_r + struct.r]
            x_reduced_list.append(x_reduced)
            
            x_full = struct.reconstruct(x_reduced)
            x_full_list.append(x_full)
            
            gf = GridFunction(struct.fes, complex=True)
            gf.vec.FV().NumPy()[:] = x_full
            gf_list.append(gf)
        
        # Initialize scales: first structure is reference
        scales = np.ones(self.n_structures, dtype=complex)
        processed = {0}
        
        # Process connections to propagate scaling through the chain
        # Handle general graph traversal (not just sequential)
        remaining_connections = list(enumerate(self.connections))
        max_iterations = len(self.connections) + 1
        
        for _ in range(max_iterations):
            if not remaining_connections:
                break
                
            made_progress = False
            still_remaining = []
            
            for conn_idx, ((sA, pA), (sB, pB)) in remaining_connections:
                # Determine which structure is reference (already processed)
                if sA in processed and sB not in processed:
                    ref_idx, new_idx = sA, sB
                    ref_port, new_port = pA, pB
                elif sB in processed and sA not in processed:
                    ref_idx, new_idx = sB, sA
                    ref_port, new_port = pB, pA
                elif sA in processed and sB in processed:
                    # Both already processed - skip
                    continue
                else:
                    # Neither processed yet - defer to next iteration
                    still_remaining.append((conn_idx, ((sA, pA), (sB, pB))))
                    continue
                
                # Compute scaling factor at this interface
                alpha = self._compute_single_interface_scale(
                    x_reduced_list[ref_idx], 
                    x_reduced_list[new_idx],
                    self.structures[ref_idx], 
                    self.structures[new_idx],
                    ref_port, 
                    new_port,
                    scales[ref_idx]  # Pass current cumulative scale of reference
                )
                
                # Apply cumulative scaling
                scales[new_idx] = alpha
                processed.add(new_idx)
                made_progress = True
                
                pr.debug(f"  Interface {ref_idx}({ref_port}) <-> {new_idx}({new_port}): "
                        f"scale = {abs(alpha):.4f}")
            
            remaining_connections = still_remaining
            
            if not made_progress and remaining_connections:
                # Disconnected structures - set remaining to 1
                pr.warning("Some structures are not connected - using scale=1 for disconnected parts")
                for conn_idx, ((sA, pA), (sB, pB)) in remaining_connections:
                    if sA not in processed:
                        scales[sA] = 1.0
                        processed.add(sA)
                    if sB not in processed:
                        scales[sB] = 1.0
                        processed.add(sB)
                break
        
        return scales


    def _compute_single_interface_scale(
        self,
        x_ref: np.ndarray,
        x_new: np.ndarray,
        struct_ref: 'ReducedStructure',
        struct_new: 'ReducedStructure',
        port_ref: str,
        port_new: str,
        cumulative_scale_ref: complex = 1.0
    ) -> complex:
        """
        Compute scaling factor for a single interface to enforce continuity.
        
        Uses the port projection operator (B_rd) analytically to calculate the precise 
        amplitude ratio preserving magnitude and relative sign.
        
        Parameters
        ----------
        x_ref : ndarray
            Reduced state vector for reference domain
        x_new : ndarray
            Reduced state vector for new domain
        struct_ref, struct_new : ReducedStructure
            Structure objects containing projection info
        port_ref, port_new : str
            Port names (boundary labels) for the shared interface
        cumulative_scale_ref : complex
            Cumulative scaling already applied to reference domain
            
        Returns
        -------
        alpha : complex
            Scaling factor for the new domain
        """
        try:
            # Obtain port modes indices for fundamental mode 0
            col_ref = struct_ref.get_port_mode_column(port_ref, mode=0)
            col_new = struct_new.get_port_mode_column(port_new, mode=0)
            
            # Calculate algebraic port coupling amplitude
            # V = B_rd^T @ x_reduced
            amp_ref = np.dot(struct_ref.Brd[:, col_ref], x_ref)
            amp_new = np.dot(struct_new.Brd[:, col_new], x_new)
            
            if abs(amp_new) > 1e-14:
                # We scale E_new so that its native amplitude equals E_ref's scaled amplitude.
                # If amp_new naturally differs in sign (e.g. out of phase normal vectors),
                # this ratio natively captures and flips it.
                alpha_local = cumulative_scale_ref * (amp_ref / amp_new)
                return alpha_local
            else:
                pr.warning(f"New field amplitude too small at interface ({abs(amp_new):.2e}); retaining previous scale.")
                return cumulative_scale_ref
                
        except Exception as e:
            pr.warning(f"Could not compute interface scale for {port_ref}<->{port_new}: {e}")
            return cumulative_scale_ref


    def _reconstruct_field(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int = 0,
        enforce_continuity: bool = True
    ) -> GridFunction:
        """
        Reconstruct the unified field over the entire structure.
        
        Parameters
        ----------
        freq_idx : int
            Frequency index
        excitation_port : str
            Name of the excited port
        excitation_mode : int
            Mode index of excitation
        enforce_continuity : bool
            If True (default), scale fields to enforce continuity at interfaces.
            This corrects for magnitude differences caused by independent POD 
            reductions of each domain.
            
        Returns
        -------
        E_gf : GridFunction
            Reconstructed electric field over the entire unified mesh
        """
        if not self.has_snapshots:
            raise ValueError("No snapshots available. Call solve() first.")

        if self.mesh is None:
            raise ValueError("No mesh available. Provide mesh to constructor.")

        # Ensure we have a unified FES (create if needed)
        self._ensure_unified_fes()

        # Validate excitation port
        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found. Available: {self.ports}")
        col_idx = self.ports.index(excitation_port)

        # Map from coupled to uncoupled stacked coordinates
        x_coupled = self._snapshots[freq_idx, :, col_idx]
        x_uncoupled = self.W_coupled @ x_coupled

        if enforce_continuity and self.n_structures > 1 and self.connections:
            scales = self._compute_interface_scaling_factors(
                freq_idx, excitation_port, excitation_mode
            )
            pr.debug(f"  Interface continuity scales: {[f'{abs(s):.3f}' for s in scales]}")
        else:
            scales = np.ones(self.n_structures, dtype=complex)

        return self._reconstruct_field_from_vector(x_uncoupled, scales)
        
    def _reconstruct_field_from_vector(
        self,
        x_uncoupled: np.ndarray,
        scales: np.ndarray
    ): # -> GridFunction
        """
        Reconstruct unified field across domains from an uncoupled vector array and apply scales. Returns a GridFunction.
        """
        from ngsolve import GridFunction
        
        # Create unified GridFunction on global FES
        E_gf = GridFunction(self.fes, complex=True)

        for struct_idx, struct in enumerate(self.structures):
            # Check that structure can reconstruct
            if not struct.can_reconstruct():
                raise ValueError(
                    f"Structure {struct_idx} ({struct.domain}) cannot reconstruct. "
                    f"W: {'set' if struct.W is not None else 'MISSING'}, "
                    f"Q_L_inv: {'set' if struct.Q_L_inv is not None else 'MISSING'}, "
                    f"fes: {'set' if struct.fes is not None else 'MISSING'}"
                )

            # Extract this structure's reduced solution
            start_r = self._structure_dof_offsets[struct_idx]
            x_reduced = x_uncoupled[start_r:start_r + struct.r]

            # Reconstruct full-order solution for this domain
            x_full_local = struct.reconstruct(x_reduced)
            
            # Apply scaling for interface continuity
            x_full_local = scales[struct_idx] * x_full_local

            # Create per-domain GridFunction
            E_local = GridFunction(struct.fes, complex=True)
            E_local.vec.FV().NumPy()[:] = x_full_local

            # Transfer to global GridFunction using definedon
            domain_region = self.mesh.Materials(struct.domain)
            E_gf.Set(E_local, definedon=domain_region)

        return E_gf


    def plot_field(
        self,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        excitation_mode: int = 0,
        component: Literal['real', 'imag', 'abs'] = 'abs',
        field_type: Literal['E', 'H'] = 'E',
        clipping: Optional[Dict] = None,
        euler_angles: Optional[List] = [45, -45, 0],
        enforce_continuity: bool = True,  # NEW PARAMETER
        **kwargs
    ) -> None:
        """
        Visualize field over the entire unified structure.

        Parameters
        ----------
        freq_idx : int
            Frequency index
        excitation_port : str, optional
            Port used for excitation. If None, uses first port.
        excitation_mode : int
            Mode index for excitation
        component : {'real', 'imag', 'abs'}
            Field component to plot
        field_type : {'E', 'H'}
            Electric or magnetic field
        clipping : dict, optional
            Clipping plane specification
        euler_angles : list, optional
            Euler angles for view orientation
        enforce_continuity : bool
            If True (default), scale fields to enforce continuity at interfaces.
        **kwargs
            Additional arguments passed to Draw()
        """
        # ... existing validation code ...
        
        if self.frequencies is None:
            raise ValueError("No solution available. Call solve() first.")

        if freq_idx >= len(self.frequencies):
            raise ValueError(f"freq_idx {freq_idx} out of range [0, {len(self.frequencies) - 1}]")

        if self._snapshots is None:
            raise ValueError("No snapshots available.")

        freq = self.frequencies[freq_idx]
        omega = 2 * np.pi * freq

        # Default excitation
        if excitation_port is None:
            if not self.ports:
                raise ValueError("No external ports available")
            excitation_port = self.ports[0]

        if excitation_port not in self.ports:
            raise ValueError(f"Port '{excitation_port}' not found. Available: {self.ports}")

        pr.info(f"\nField visualization at f = {freq / 1e9:.4f} GHz")
        pr.info(f"  Excitation: {excitation_port}, mode {excitation_mode}")
        pr.debug(f"  Unified structure with {self.n_structures} domains: {self.domains}")
        pr.debug(f"  Interface continuity: {'enabled' if enforce_continuity else 'disabled'}")

        # Reconstruct unified field WITH continuity enforcement
        E_gf = self._reconstruct_field(
            freq_idx, excitation_port, excitation_mode,
            enforce_continuity=enforce_continuity
        )

        # ... rest of plotting code unchanged ...
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
            plot_name = f"|{field_label}|"
        elif component == 'real':
            cf_plot = field_cf.real
            plot_name = f"Re({field_label})"
        elif component == 'imag':
            cf_plot = field_cf.imag
            plot_name = f"Im({field_label})"
        else:
            raise ValueError(f"Invalid component: {component}")

        pr.debug(f"  Plotting: {plot_name}")

        draw_kwargs = kwargs.copy()
        if clipping:
            draw_kwargs['clipping'] = clipping
        if euler_angles:
            draw_kwargs['euler_angles'] = euler_angles

        Draw(BoundaryFromVolumeCF(cf_plot), self.mesh, plot_name, **draw_kwargs)


    def clear_interface_scale_cache(self) -> None:
        """Clear the cached interface scaling factors."""
        self._interface_scale_cache.clear()


    def get_interface_scales(
        self, 
        freq_idx: int, 
        excitation_port: str,
        excitation_mode: int = 0
    ) -> Dict[str, complex]:
        """
        Get the interface scaling factors for a given excitation.
        
        Returns
        -------
        scales : dict
            Mapping from domain name to scaling factor
        """
        scales = self._compute_interface_scaling_factors(
            freq_idx, excitation_port, excitation_mode
        )
        return {self.domains[i]: scales[i] for i in range(self.n_structures)}
    
    @property
    def singular_values(self) -> np.ndarray:
        return self._singular_values

    @property
    def reduction_level(self) -> int:
        return self._reduction_level

    def print_info(self) -> None:
        """Print reduced system information."""
        pr.info("\n" + "=" * 60)
        pr.info(f"Reduced Concatenated System (Level {self._reduction_level})")
        pr.info("=" * 60)

        pr.debug(f"\nReduction:")
        pr.debug(f"  Parent coupled DOFs: {self._parent_coupled_dofs}")
        pr.debug(f"  This level DOFs: {self.coupled_dofs}")
        if self._parent_coupled_dofs > 0:
            compression = (1 - self.coupled_dofs / self._parent_coupled_dofs) * 100
            pr.debug(f"  Compression: {compression:.1f}%")

        pr.debug(f"\nSingular values (top 5):")
        for i, sv in enumerate(self._singular_values[:5]):
            print(f"  σ_{i} = {sv:.4e}")
        if len(self._singular_values) > 5:
            print(f"  ... ({len(self._singular_values)} total)")

        print(f"Field reconstruction: {'Ready' if self.can_reconstruct() else 'Not available'}")

        if self.frequencies is not None:
            print(f"\nSolution:")
            print(f"  Range: {self.frequencies[0] / 1e9:.4f} - {self.frequencies[-1] / 1e9:.4f} GHz")
            print(f"  Samples: {len(self.frequencies)}")

        print("=" * 60)


def reduce_concatenated_system(
    concat: ConcatenatedSystem,
    tol: float = 1e-6,
    max_rank: Optional[int] = None,
) -> ReducedConcatenatedSystem:
    """
    Reduce a concatenated system via POD.

    Parameters
    ----------
    concat : ConcatenatedSystem
        System to reduce (must have solve() called)
    tol : float
        SVD truncation tolerance (relative to largest singular value)
    max_rank : int, optional
        Maximum rank for reduced system

    Returns
    -------
    ReducedConcatenatedSystem
        Further-reduced unified system
    """
    if concat.A_coupled is None:
        raise ValueError("System must be coupled first")
    if concat._snapshots is None:
        raise ValueError("No snapshots. Call solve() first.")

    r_current = concat.A_coupled.shape[0]

    # Collect snapshots: shape (n_freq, r_coupled, n_ext) -> (r_coupled, n_freq * n_ext)
    W_snap = np.hstack([concat._snapshots[k] for k in range(len(concat._snapshots))])

    # SVD for POD basis
    U, S, _ = np.linalg.svd(W_snap, full_matrices=False)

    # Determine truncation rank
    r_new = max(1, int(np.sum(S > tol * S[0])))
    if max_rank is not None:
        r_new = min(r_new, max_rank)

    W_r = U[:, :r_new]

    # Project system
    A_reduced = W_r.T.conj() @ concat.A_coupled @ W_r
    B_reduced = W_r.T.conj() @ concat.B_coupled

    # Ensure Hermitian
    A_reduced = 0.5 * (A_reduced + A_reduced.T.conj())

    print(f"\nReduced unified system: {r_current} -> {r_new} DOFs")
    print(f"  Compression: {100 * (1 - r_new / r_current):.1f}%")
    print(f"  Singular value decay: {S[0]:.2e} -> {S[min(r_new, len(S) - 1)]:.2e}")

    return ReducedConcatenatedSystem(
        parent=concat,
        A_reduced=A_reduced,
        B_reduced=B_reduced,
        W_reduction=W_r,
        singular_values=S,
    )