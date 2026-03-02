"""
Structure concatenation for multi-cell analysis.

Key fixes / improvements
------------------------
1) Correct handling of multiple modes per port:
   - Port indexing now tracks (port_name, mode_index) pairs
   - B matrices have columns for each port-mode combination

2) Correct incidence matrix F for internal connections

3) Optional per-connection sign control

4) Robust linear algebra with pseudo-inverse
"""

from typing import List, Tuple, Dict, Optional, Callable, Union
import numpy as np
import scipy.linalg as sl

from core.constants import Z0
from solvers.base import BaseEMSolver
from rom.structures import ReducedStructure


# Connection now specifies port names; modes are connected mode-by-mode
Conn = Tuple[Tuple[int, str], Tuple[int, str]]  # ((struct_idx, port_name), (struct_idx, port_name))
ConnSigns = Tuple[float, float]  # (signA, signB) e.g. (+1,-1) or (+1,+1)


class ConcatenatedSystem(BaseEMSolver):
    """
    Coupled system of multiple reduced structures with multiple modes per port.

    Coupling idea
    -------------
    Each ReducedStructure has reduced system:
        (A - ω² I) x = ω B u
    with Z extracted via:
        Z = j B^T x

    B has columns for each (port, mode) pair.

    Concatenation couples internal ports using a Kirchhoff-style constraint:
        F^T B_int^T x = 0
    where F is an incidence matrix encoding internal connections.

    Connections are specified at the port level, and all modes of connected
    ports are coupled (mode 0 ↔ mode 0, mode 1 ↔ mode 1, etc.)
    """

    DEFAULT_MIN_EIGENVALUE = 1.0

    def __init__(
        self,
        structures: List[ReducedStructure],
        port_impedance_func: Optional[Callable[[str, int, float], complex]] = None,
    ):
        super().__init__()

        self.structures = structures
        self.n_structures = len(structures)
        self._port_impedance_func = port_impedance_func or self._default_impedance

        # Validate consistent mode counts
        self._validate_mode_counts()

        # Assign global port-mode indices for each (structure, port_name, mode)
        self._assign_global_port_modes()

        # Coupling results
        self.A_coupled: Optional[np.ndarray] = None
        self.B_coupled: Optional[np.ndarray] = None
        self.W_coupled: Optional[np.ndarray] = None

        # Connection tracking
        self.connections: Optional[List[Conn]] = None
        self.n_connections: int = 0
        self._connection_signs: Optional[List[ConnSigns]] = None

        self._internal_port_modes: List[int] = []  # global indices of internal port-mode pairs
        self._external_port_modes: List[int] = []  # global indices of external port-mode pairs
        self._external_port_mode_names: List[Tuple[str, int]] = []  # (port_name, mode_idx)
        self._external_port_mode_map: Dict[str, Tuple[int, str, int]] = {}  # new_name -> (struct, port, mode)
        self._permutation: Optional[np.ndarray] = None
        self._n_internal: int = 0
        self._n_external: int = 0

        self.domains = [s.domain for s in structures]

    def _validate_mode_counts(self) -> None:
        """Validate that mode counts are consistent across structures."""
        if not self.structures:
            return

        # Get reference mode count from first structure
        ref_n_modes = self.structures[0].n_port_modes

        for i, struct in enumerate(self.structures):
            if struct.n_port_modes != ref_n_modes:
                raise ValueError(
                    f"Inconsistent mode counts: structure 0 has {ref_n_modes} modes/port, "
                    f"but structure {i} has {struct.n_port_modes} modes/port"
                )

        self._n_modes_per_port = ref_n_modes

    @staticmethod
    def _default_impedance(port: str, mode: int, freq: float) -> complex:
        return Z0

    def _assign_global_port_modes(self) -> None:
        """
        Assign global indices to each (structure, port, mode) combination.

        The ordering matches how B matrices are constructed:
        for each structure, columns are ordered as:
            port0_mode0, port0_mode1, ..., port1_mode0, port1_mode1, ...
        """
        offset = 0

        # Maps (struct_idx, port_name, mode_idx) -> global_index
        self.port_mode_map: Dict[Tuple[int, str, int], int] = {}

        # Maps global_index -> (struct_idx, port_name, mode_idx)
        self._global_to_local: Dict[int, Tuple[int, str, int]] = {}

        # Maps (struct_idx, port_name) -> (start_global_idx, n_modes)
        self.port_to_mode_range: Dict[Tuple[int, str], Tuple[int, int]] = {}

        for struct_idx, struct in enumerate(self.structures):
            n_modes = struct.n_port_modes
            for port_idx, port in enumerate(struct.ports):
                start_idx = offset
                for mode_idx in range(n_modes):
                    global_idx = offset
                    self.port_mode_map[(struct_idx, port, mode_idx)] = global_idx
                    self._global_to_local[global_idx] = (struct_idx, port, mode_idx)
                    offset += 1
                self.port_to_mode_range[(struct_idx, port)] = (start_idx, n_modes)

        self.n_total_port_modes = offset

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
        connections:
            List of ((structA, portA), (structB, portB)) tuples.
            All modes of portA are connected to corresponding modes of portB.
        connection_signs:
            Optional list of (signA, signB) per connection.
            Default is (+1, -1) for each connection.
        validate:
            If True, perform sanity checks.
        """
        self.connections = list(connections)
        self.n_connections = len(self.connections)

        if connection_signs is None:
            self._connection_signs = [(+1.0, -1.0) for _ in range(self.n_connections)]
        else:
            if len(connection_signs) != self.n_connections:
                raise ValueError(
                    f"connection_signs length ({len(connection_signs)}) must match "
                    f"number of connections ({self.n_connections})."
                )
            self._connection_signs = [(float(a), float(b)) for (a, b) in connection_signs]

        # Identify internal and external port-mode pairs
        internal_set = set()

        for (sA_idx, pA), (sB_idx, pB) in self.connections:
            if validate:
                if (sA_idx, pA) not in self.port_to_mode_range:
                    raise KeyError(f"Unknown port in connection: ({sA_idx}, '{pA}')")
                if (sB_idx, pB) not in self.port_to_mode_range:
                    raise KeyError(f"Unknown port in connection: ({sB_idx}, '{pB}')")

                # Check mode counts match
                n_modes_A = self.port_to_mode_range[(sA_idx, pA)][1]
                n_modes_B = self.port_to_mode_range[(sB_idx, pB)][1]
                if n_modes_A != n_modes_B:
                    raise ValueError(
                        f"Mode count mismatch in connection: "
                        f"({sA_idx}, '{pA}') has {n_modes_A} modes, "
                        f"({sB_idx}, '{pB}') has {n_modes_B} modes"
                    )

            # Mark all modes of these ports as internal
            for mode_idx in range(self._n_modes_per_port):
                internal_set.add(self.port_mode_map[(sA_idx, pA, mode_idx)])
                internal_set.add(self.port_mode_map[(sB_idx, pB, mode_idx)])

        self._internal_port_modes = sorted(internal_set)
        self._external_port_modes = sorted(
            set(range(self.n_total_port_modes)) - internal_set
        )
        self._n_internal = len(self._internal_port_modes)
        self._n_external = len(self._external_port_modes)

        # Build permutation matrix P: reorder to [internal | external]
        perm = self._internal_port_modes + self._external_port_modes
        P = np.zeros((self.n_total_port_modes, self.n_total_port_modes), dtype=float)
        for new_pos, old_pos in enumerate(perm):
            P[new_pos, old_pos] = 1.0
        self._permutation = P

        # Build external port-mode names and mapping
        self._external_port_mode_names = []
        self._external_port_mode_map = {}

        for i, global_idx in enumerate(self._external_port_modes):
            struct_idx, orig_port, mode_idx = self._global_to_local[global_idx]
            # Create unique name: "port{port_num}({mode_num})"
            # Find port number within external ports
            new_name = f"port{i // self._n_modes_per_port + 1}({mode_idx + 1})"
            self._external_port_mode_names.append((orig_port, mode_idx))
            self._external_port_mode_map[new_name] = (struct_idx, orig_port, mode_idx)

        if validate:
            self._validate_connections()

        return self

    def _validate_connections(self) -> None:
        """Sanity checks for connection definition."""
        assert self.connections is not None
        assert self._permutation is not None

        # Check that connections don't connect a port to itself
        for j, ((sA, pA), (sB, pB)) in enumerate(self.connections):
            if sA == sB and pA == pB:
                raise ValueError(
                    f"Connection {j} connects port '{pA}' of structure {sA} to itself"
                )

    def _build_incidence_matrix(self) -> np.ndarray:
        """
        Build incidence matrix F for Kirchhoff coupling.

        Shape: (n_internal_port_modes, n_connections * n_modes_per_port)

        For each port-level connection j, we create n_modes_per_port mode-level
        connections (mode 0 ↔ mode 0, mode 1 ↔ mode 1, etc.)
        """
        if self.connections is None:
            raise ValueError("Must call define_connections() first")

        n_int = self._n_internal
        n_mode_connections = self.n_connections * self._n_modes_per_port

        F = np.zeros((n_int, n_mode_connections), dtype=float)

        # Map global port-mode index to position in internal array
        int_pos = {g: i for i, g in enumerate(self._internal_port_modes)}

        col = 0
        for conn_idx, (((sA_idx, pA), (sB_idx, pB)), (sgnA, sgnB)) in enumerate(
            zip(self.connections, self._connection_signs)
        ):
            # For each mode, create a connection
            for mode_idx in range(self._n_modes_per_port):
                gA = self.port_mode_map[(sA_idx, pA, mode_idx)]
                gB = self.port_mode_map[(sB_idx, pB, mode_idx)]

                F[int_pos[gA], col] = float(sgnA)
                F[int_pos[gB], col] = float(sgnB)
                col += 1

        # Validation: each column must have exactly two nonzeros
        for j in range(n_mode_connections):
            nz = np.flatnonzero(np.abs(F[:, j]) > 0)
            if len(nz) != 2:
                conn_idx = j // self._n_modes_per_port
                mode_idx = j % self._n_modes_per_port
                raise RuntimeError(
                    f"Bad incidence column {j} (connection {conn_idx}, mode {mode_idx}): "
                    f"expected 2 nonzeros, got {len(nz)}."
                )

        return F

    def couple(self, rcond_pinv: float = 1e-12, rcond_null: float = 1e-12) -> "ConcatenatedSystem":
        """
        Perform structure coupling via null-space projection.
        """
        if self.connections is None:
            raise ValueError("Must call define_connections() first")

        # Block-diagonal assembly
        A_blocks = [np.asarray(s.Ard) for s in self.structures]
        B_blocks = [np.asarray(s.Brd) for s in self.structures]

        A_blk = sl.block_diag(*A_blocks).astype(complex, copy=False)
        B_blk = sl.block_diag(*B_blocks).astype(complex, copy=False)

        # Apply permutation to reorder port-mode columns: [internal | external]
        P = self._permutation
        B_perm = B_blk @ P.T  # (sum r_i) x n_total_port_modes

        B_int = B_perm[:, :self._n_internal]
        B_ext = B_perm[:, self._n_internal:]

        # Incidence matrix
        F = self._build_incidence_matrix()

        # Constraint operator: C = B_int @ F
        C = B_int @ F

        # Projection K = I - C (C^H C)^+ C^H
        CtC = C.T.conj() @ C
        CtC_inv = sl.pinvh(CtC)

        I = np.eye(A_blk.shape[0], dtype=complex)
        K = I - C @ CtC_inv @ C.T.conj()

        # Null space basis of C^T
        M = sl.null_space(C.T.conj(), rcond=rcond_null).astype(complex, copy=False)
        if M.size == 0:
            raise RuntimeError("Null space is empty; constraints overconstrained or inconsistent.")

        KM = K @ M

        # Project system
        self.A_coupled = KM.T.conj() @ A_blk @ KM
        self.B_coupled = KM.T.conj() @ B_ext
        self.W_coupled = KM

        # Hermitian symmetrize A
        self.A_coupled = 0.5 * (self.A_coupled + self.A_coupled.T.conj())

        print(f"\nCoupled system: {A_blk.shape[0]} -> {self.A_coupled.shape[0]} DOFs")
        print(f"External port-modes: {self._n_external}, Internal port-modes: {self._n_internal}")
        print(f"Port connections: {self.n_connections} (× {self._n_modes_per_port} modes each)")

        return self

    # -------------------------------------------------------------------------
    # BaseEMSolver interface
    # -------------------------------------------------------------------------

    @property
    def n_ports(self) -> int:
        """Number of external port-mode combinations."""
        return self._n_external

    @property
    def n_external_ports(self) -> int:
        """Number of unique external ports (not counting modes)."""
        return self._n_external // self._n_modes_per_port

    @property
    def n_modes_per_port(self) -> int:
        """Number of modes per port."""
        return self._n_modes_per_port

    @property
    def ports(self) -> List[str]:
        """External port-mode names."""
        return list(self._external_port_mode_map.keys())

    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        """Get impedance for external port-mode."""
        if port not in self._external_port_mode_map:
            raise KeyError(f"Port '{port}' not found. Available: {self.ports}")
        _, orig_port, orig_mode = self._external_port_mode_map[port]
        return self._port_impedance_func(orig_port, orig_mode, freq)

    def _get_impedance_matrix(self, freq: float) -> np.ndarray:
        """
        Get diagonal reference impedance matrix for external port-modes.

        Override required because ConcatenatedSystem has special port-mode
        indexing that doesn't follow the simple port_idx = matrix_idx // n_modes
        pattern used by regular solvers.
        """
        Z0_diag = []

        for global_idx in self._external_port_modes:
            struct_idx, port_name, mode_idx = self._global_to_local[global_idx]
            Zw = self._port_impedance_func(port_name, mode_idx, freq)
            Z0_diag.append(Zw)

        return np.diag(Z0_diag)

    def solve(
        self,
        fmin: float,
        fmax: float,
        nsamples: int = 100,
        compute_s_params: bool = True,
        rcond_solve: float = 1e-12,
    ) -> Dict:
        """
        Solve coupled reduced system over frequency range.
        """
        if self.A_coupled is None or self.B_coupled is None:
            raise ValueError("Must call couple() first")

        self.frequencies = np.linspace(fmin, fmax, nsamples) * 1e9
        n_ext = self._n_external
        r = self.A_coupled.shape[0]

        self._Z_matrix = np.zeros((nsamples, n_ext, n_ext), dtype=complex)
        x_all = []

        I_ext = np.eye(n_ext, dtype=complex)

        for k, freq in enumerate(self.frequencies):
            omega = 2 * np.pi * freq

            lhs = self.A_coupled - (omega**2) * np.eye(r, dtype=complex)
            rhs = omega * (self.B_coupled @ I_ext)

            try:
                x = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                x = np.linalg.lstsq(lhs, rhs, rcond=rcond_solve)[0]

            x_all.append(x)
            self._Z_matrix[k] = 1j * (self.B_coupled.T.conj() @ x)

        if compute_s_params:
            self._compute_s_from_z()

        self._invalidate_cache()

        return {
            "frequencies": self.frequencies,
            "Z": self._Z_matrix,
            "S": self._S_matrix if compute_s_params else None,
            "Z_dict": self.Z_dict,
            "S_dict": self.S_dict if compute_s_params else None,
            "x": np.array(x_all),
        }

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def verify_kirchhoff(self, x_uncoupled: np.ndarray) -> float:
        """
        Verify constraint satisfaction on an *uncoupled stacked* state vector.
        """
        if self.connections is None:
            raise ValueError("Must call define_connections() first")

        B_blocks = [np.asarray(s.Brd) for s in self.structures]
        B_blk = sl.block_diag(*B_blocks).astype(complex, copy=False)
        P = self._permutation
        B_perm = B_blk @ P.T
        B_int = B_perm[:, :self._n_internal]
        F = self._build_incidence_matrix()

        viol = F.T @ (B_int.T.conj() @ x_uncoupled)
        return float(np.max(np.abs(viol)))

    @staticmethod
    def _filter_eigenvalues(
        eigenvalues: np.ndarray,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> np.ndarray:
        """Filter and sort eigenvalues."""
        if min_eigenvalue is None:
            min_eigenvalue = ConcatenatedSystem.DEFAULT_MIN_EIGENVALUE

        eigs_sorted = np.sort(np.real(eigenvalues))

        if filter_static:
            eigs_sorted = eigs_sorted[eigs_sorted > min_eigenvalue]

        if n_modes is not None and len(eigs_sorted) > n_modes:
            eigs_sorted = eigs_sorted[:n_modes]

        return eigs_sorted

    def get_eigenvalues(
        self,
        domain: str = None,
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Compute eigenvalues of coupled system matrix."""
        if self.A_coupled is None:
            raise ValueError("Must call couple() first")

        def process_eigs(raw_eigs: np.ndarray) -> np.ndarray:
            return self._filter_eigenvalues(raw_eigs, filter_static, min_eigenvalue, n_modes)

        if domain is None or domain == 'global':
            raw_eigs = np.linalg.eigvalsh(self.A_coupled)
            return process_eigs(raw_eigs)

        for struct in self.structures:
            if struct.domain == domain:
                raw_eigs = np.linalg.eigvalsh(struct.Ard)
                return process_eigs(raw_eigs)

        raise KeyError(f"Domain '{domain}' not found. Available: {self.domains}")

    def get_resonant_frequencies(
        self,
        domain: str = None,
        n_modes: int = None,
        filter_static: bool = True,
        min_eigenvalue: float = None
    ) -> np.ndarray:
        """Get resonant frequencies from eigenvalues."""
        eigs = self.get_eigenvalues(
            domain=domain,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=None
        )

        eigs_pos = eigs[eigs > 0]
        freqs = np.sqrt(eigs_pos) / (2 * np.pi)
        freqs = np.sort(freqs)

        if n_modes is not None:
            freqs = freqs[:n_modes]

        return freqs

    def get_coupled_dimensions(self) -> Dict[str, int]:
        """Get information about coupled system dimensions."""
        total_uncoupled = sum(s.r for s in self.structures)
        return {
            'n_structures': self.n_structures,
            'total_uncoupled_dofs': total_uncoupled,
            'coupled_dofs': self.A_coupled.shape[0] if self.A_coupled is not None else None,
            'n_internal_port_modes': self._n_internal,
            'n_external_port_modes': self._n_external,
            'n_modes_per_port': self._n_modes_per_port,
            'n_connections': self.n_connections if self.connections else 0,
            'domains': self.domains
        }

    @property
    def coupled_dofs(self) -> int:
        if self.A_coupled is None:
            return 0
        return self.A_coupled.shape[0]

    @property
    def has_solution(self) -> bool:
        return self.frequencies is not None

    def print_info(self) -> None:
        """Print concatenated system information."""
        print("\n" + "=" * 60)
        print("ConcatenatedSystem Information")
        print("=" * 60)

        print(f"\nStructures ({self.n_structures}):")
        for i, s in enumerate(self.structures):
            print(f"  [{i}] {s.domain}: r={s.r}, ports={s.ports}, modes/port={s.n_port_modes}")

        if self.connections:
            print(f"\nConnections ({len(self.connections)}):")
            for (sA, pA), (sB, pB) in self.connections:
                print(f"  ({sA}, {pA}) <-> ({sB}, {pB}) [× {self._n_modes_per_port} modes]")

        dims = self.get_coupled_dimensions()
        print(f"\nDimensions:")
        print(f"  Uncoupled DOFs: {dims['total_uncoupled_dofs']}")
        print(f"  Coupled DOFs: {dims['coupled_dofs']}")
        print(f"  Modes per port: {dims['n_modes_per_port']}")
        print(f"  Internal port-modes: {dims['n_internal_port_modes']}")
        print(f"  External port-modes: {dims['n_external_port_modes']}")

        print(f"\nExternal port-mode mapping:")
        for new_name, (struct_idx, orig_name, mode_idx) in self._external_port_mode_map.items():
            print(f"  {new_name} -> structure[{struct_idx}].{orig_name}[mode {mode_idx}]")

        if self.frequencies is not None:
            print(f"\nSolution available:")
            print(f"  Frequency range: {self.frequencies[0]/1e9:.4f} - {self.frequencies[-1]/1e9:.4f} GHz")
            print(f"  Number of samples: {len(self.frequencies)}")

        print("=" * 60)


class ReducedConcatenatedSystem(ConcatenatedSystem):
    """Further-reduced concatenated system via POD."""

    def __init__(
        self,
        parent: ConcatenatedSystem,
        A_reduced: np.ndarray,
        B_reduced: np.ndarray,
        W_reduction: np.ndarray,
        singular_values: np.ndarray,
    ):
        BaseEMSolver.__init__(self)

        # Copy parent wiring/state
        self.structures = parent.structures
        self.n_structures = parent.n_structures
        self._port_impedance_func = parent._port_impedance_func
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
        self.domains = parent.domains

        # Reduced operators
        self.A_coupled = np.asarray(A_reduced).astype(complex, copy=False)
        self.B_coupled = np.asarray(B_reduced).astype(complex, copy=False)
        self.W_coupled = np.asarray(W_reduction).astype(complex, copy=False)
        self._singular_values = np.asarray(singular_values)

        self._parent = parent
        self._reduction_level = getattr(parent, "_reduction_level", 0) + 1
        self._parent_coupled_dofs = parent.A_coupled.shape[0] if parent.A_coupled is not None else None

    @property
    def singular_values(self) -> np.ndarray:
        return self._singular_values

    @property
    def reduction_level(self) -> int:
        return self._reduction_level


def reduce_concatenated_system(
    concat: ConcatenatedSystem,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    n_snapshots: Optional[int] = None,
    tol: float = 1e-6,
    max_rank: Optional[int] = None,
) -> ReducedConcatenatedSystem:
    """Reduce a concatenated system via POD."""
    if concat.A_coupled is None or concat.B_coupled is None:
        raise ValueError("ConcatenatedSystem must be coupled first")

    r_current = concat.A_coupled.shape[0]
    n_ext = concat.n_ports

    if fmin is None or fmax is None:
        if concat.frequencies is not None:
            fmin = float(concat.frequencies[0] / 1e9)
            fmax = float(concat.frequencies[-1] / 1e9)
        else:
            raise ValueError("Must specify fmin/fmax or solve the system first")

    if n_snapshots is None:
        n_snapshots = min(2 * r_current, 200)

    freqs = np.linspace(fmin, fmax, n_snapshots) * 1e9
    snapshots = []

    I_ext = np.eye(n_ext, dtype=complex)

    for freq in freqs:
        omega = 2 * np.pi * freq
        lhs = concat.A_coupled - (omega**2) * np.eye(r_current, dtype=complex)
        rhs = omega * (concat.B_coupled @ I_ext)
        x = np.linalg.lstsq(lhs, rhs, rcond=1e-12)[0]
        snapshots.append(x)

    W_snap = np.hstack(snapshots)

    U, S, _ = np.linalg.svd(W_snap, full_matrices=False)

    r_new = int(np.sum(S > tol * S[0]))
    r_new = max(r_new, 1)
    if max_rank is not None:
        r_new = min(r_new, int(max_rank))

    W_r = U[:, :r_new]

    A_reduced = W_r.T.conj() @ concat.A_coupled @ W_r
    B_reduced = W_r.T.conj() @ concat.B_coupled

    A_reduced = 0.5 * (A_reduced + A_reduced.T.conj())

    print(f"\nReduced concatenated system: {r_current} -> {r_new} DOFs")
    print(f"Compression: {100*(1 - r_new/r_current):.1f}%")

    return ReducedConcatenatedSystem(
        parent=concat,
        A_reduced=A_reduced,
        B_reduced=B_reduced,
        W_reduction=W_r,
        singular_values=S,
    )


def _concatenated_reduce(
    self: ConcatenatedSystem,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    n_snapshots: Optional[int] = None,
    tol: float = 1e-6,
    max_rank: Optional[int] = None,
) -> ReducedConcatenatedSystem:
    """Attachable method: concat.reduce(...)"""
    return reduce_concatenated_system(self, fmin, fmax, n_snapshots, tol, max_rank)


ConcatenatedSystem.reduce = _concatenated_reduce