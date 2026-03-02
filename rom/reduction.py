"""Model Order Reduction using Proper Orthogonal Decomposition."""

from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
from solvers.eigen_mixin import ROMEigenMixin

from solvers.base import BaseEMSolver, ParameterConverter
from rom.structures import ReducedStructure


class ModelOrderReduction(BaseEMSolver, ROMEigenMixin):
    """
    POD-based Model Order Reduction for electromagnetic structures.

    Handles both single-domain and compound (multi-domain) structures uniformly.
    Single-domain is just a special case with n_domains=1.

    Reduces the system:
        (K - ω²M)x = ωBu

    To:
        (A_r - ω²I)x_r = ωB_r u

    For multi-domain structures, domains are automatically concatenated
    via Kirchhoff coupling at internal ports. The global system matrices
    (A_r_global, B_r_global) are identical to those of the ConcatenatedSystem.

    **Important**: For multi-domain structures, the solver must be run with
    `per_domain=True` to generate per-domain snapshots for reduction.

    Parameters
    ----------
    solver : FrequencyDomainSolver
        Solved frequency domain solver with snapshots

    Examples
    --------
    >>> # Single domain
    >>> fds = FrequencyDomainSolver(geometry)
    >>> fds.solve(1, 10, 100, store_snapshots=True)
    >>> rom = ModelOrderReduction(fds)
    >>> rom.reduce(tol=1e-6)

    >>> # Multi-domain (compound structure)
    >>> fds = FrequencyDomainSolver(compound_geometry)
    >>> fds.solve(1, 10, 100, store_snapshots=True, per_domain=True)
    >>> rom = ModelOrderReduction(fds)
    >>> rom.reduce(tol=1e-6)
    """

    # Default threshold for filtering static modes (eigenvalues below this are removed)
    DEFAULT_MIN_EIGENVALUE = 1.0  # ω² > 1 means ω > 1 rad/s

    def __init__(self, solver):
        """
        Initialize from FrequencyDomainSolver.
        """
        super().__init__()

        self.solver = solver
        self.mesh = solver.mesh
        self.domains = solver.domains
        self._all_ports = solver.all_ports
        self._external_ports = solver.external_ports
        self.domain_port_map = solver.domain_port_map
        self.n_domains = solver.n_domains
        self._n_ports_total = len(self._all_ports)
        self._n_ports_external = len(self._external_ports)
        self.port_modes = solver.port_modes

        # Compute n_modes_per_port from port_modes
        if self.port_modes:
            first_port = next(iter(self.port_modes.keys()))
            self._n_modes_per_port = len(self.port_modes[first_port])
        else:
            self._n_modes_per_port = solver._n_modes_per_port or 1

        # Port impedance function from solver
        self._port_impedance_func = solver.port_solver.get_port_wave_impedance

        # Per-domain storage
        self._M: Dict[str, sp.csr_matrix] = {}
        self._K: Dict[str, sp.csr_matrix] = {}
        self._B: Dict[str, np.ndarray] = {}
        self._snapshots: Dict[str, np.ndarray] = {}
        self._W: Dict[str, np.ndarray] = {}
        self._A_r: Dict[str, np.ndarray] = {}
        self._B_r: Dict[str, np.ndarray] = {}
        self._Q_L_inv: Dict[str, np.ndarray] = {}
        self._r: Dict[str, int] = {}
        self._singular_values: Dict[str, np.ndarray] = {}

        # Global (concatenated) storage
        self._A_r_global: Optional[np.ndarray] = None
        self._B_r_global: Optional[np.ndarray] = None
        self._W_r_global: Optional[np.ndarray] = None
        self._r_global: Optional[int] = None
        self._concatenated: Optional['ConcatenatedSystem'] = None

        # Load data from solver
        self._load_from_solver()

        # Track if reduced
        self._is_reduced = False

    def _load_from_solver(self) -> None:
        """Load matrices and snapshots from solver."""
        has_per_domain_snapshots = False
        has_global_snapshots = False
        has_matrices = False

        # Load per-domain data
        for domain in self.domains:
            rom_data = self.solver.get_rom_data(domain)

            if rom_data['M'] is not None:
                self._M[domain] = rom_data['M']
                self._K[domain] = rom_data['K']
                self._B[domain] = rom_data['B']
                has_matrices = True

            if rom_data['W'] is not None:
                self._snapshots[domain] = rom_data['W']
                has_per_domain_snapshots = True

        # Check for global snapshots
        if 'global' in self.solver.snapshots:
            global_rom_data = self.solver.get_rom_data('global')
            if global_rom_data['W'] is not None:
                self._snapshots['global'] = global_rom_data['W']
                has_global_snapshots = True

        # Provide helpful warnings
        if not has_matrices:
            print("Warning: No matrices found in solver. "
                  "Ensure assemble_matrices() was called.")

        if not has_per_domain_snapshots and not has_global_snapshots:
            print("Warning: No snapshots found in solver. "
                  "Ensure solve() was called with store_snapshots=True")
        elif self.n_domains > 1 and not has_per_domain_snapshots:
            print("Warning: Multi-domain structure detected but only global snapshots available.")
            print("         For multi-domain ROM, re-run solve() with per_domain=True:")
            print("         solver.solve(fmin, fmax, nsamples, per_domain=True, store_snapshots=True)")

    def _validate_snapshots_for_reduction(self) -> None:
        """Validate that required snapshots are available for reduction."""
        if not self._snapshots:
            raise ValueError(
                "No snapshots available. Ensure solver.solve() was called "
                "with store_snapshots=True"
            )

        # For multi-domain, we need per-domain snapshots
        if self.n_domains > 1:
            missing_domains = [d for d in self.domains if d not in self._snapshots]
            if missing_domains:
                available = list(self._snapshots.keys())
                raise ValueError(
                    f"Per-domain snapshots required for multi-domain ROM.\n"
                    f"  Missing snapshots for domains: {missing_domains}\n"
                    f"  Available snapshots: {available}\n\n"
                    f"Solution: Re-run the solver with per_domain=True:\n"
                    f"  solver.solve(fmin, fmax, nsamples, per_domain=True, store_snapshots=True)\n\n"
                    f"Then create a new ROM:\n"
                    f"  rom = ModelOrderReduction(solver)\n"
                    f"  rom.reduce(tol=1e-6)"
                )

        # For single-domain, we can use either domain or global snapshots
        if self.n_domains == 1:
            domain = self.domains[0]
            if domain not in self._snapshots and 'global' not in self._snapshots:
                raise ValueError(
                    f"No snapshots found for domain '{domain}' or 'global'. "
                    f"Available: {list(self._snapshots.keys())}"
                )

    # === BaseEMSolver abstract implementations ===

    @property
    def n_ports(self) -> int:
        """Number of external ports."""
        return self._n_ports_external

    @property
    def ports(self) -> List[str]:
        """External port names."""
        return self._external_ports.copy()

    @property
    def all_ports(self) -> List[str]:
        """All port names including internal."""
        return self._all_ports.copy()

    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        """Get port impedance from underlying solver."""
        return self._port_impedance_func(port, mode, freq)

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
            Threshold for static mode filtering. Default: DEFAULT_MIN_EIGENVALUE
        n_modes : int, optional
            Return only first n_modes eigenvalues

        Returns
        -------
        filtered_eigenvalues : ndarray
            Sorted, filtered eigenvalues
        """
        if min_eigenvalue is None:
            min_eigenvalue = ModelOrderReduction.DEFAULT_MIN_EIGENVALUE

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
        source: str = 'auto',
        filter_static: bool = True,
        min_eigenvalue: float = None,
        n_modes: int = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute eigenvalues of reduced system matrix A_r.

        For the system (A_r - ω²I)x = ωB_r u, eigenvalues of A_r give ω².

        **Default behavior**: Returns global (concatenated) eigenvalues for
        multi-domain structures, with static modes filtered out.

        Parameters
        ----------
        domain : str, optional
            Specific domain name, or 'global' for concatenated system.
            If None, uses 'source' parameter to determine behavior.
        source : str
            Only used when domain is None:
            - 'auto': Returns global if available, else single domain (default)
            - 'global': Returns only global eigenvalues
            - 'per_domain': Returns dict of per-domain eigenvalues
            - 'all': Returns dict with both global and per-domain
        filter_static : bool
            If True (default), remove static modes (eigenvalues <= min_eigenvalue)
        min_eigenvalue : float, optional
            Threshold for static mode filtering. Default: 1.0 (ω² > 1)
        n_modes : int, optional
            Return only first n_modes eigenvalues (after filtering)

        Returns
        -------
        eigenvalues : ndarray or dict
            Eigenvalues (ω² values), sorted and filtered
        """
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        def process_eigs(raw_eigs: np.ndarray) -> np.ndarray:
            return self._filter_eigenvalues(raw_eigs, filter_static, min_eigenvalue, n_modes)

        # Specific domain requested
        if domain is not None:
            if domain == 'global':
                raw_eigs = self._get_global_eigenvalues_raw()
            elif domain in self._A_r:
                raw_eigs = np.linalg.eigvalsh(self._A_r[domain])
            else:
                raise KeyError(f"Domain '{domain}' not found. "
                               f"Available: {list(self._A_r.keys())} + ['global']")
            return process_eigs(raw_eigs)

        # Auto-detect based on source parameter
        if source == 'auto':
            raw_eigs = self._get_global_eigenvalues_raw()
            return process_eigs(raw_eigs)

        elif source == 'global':
            raw_eigs = self._get_global_eigenvalues_raw()
            return process_eigs(raw_eigs)

        elif source == 'per_domain':
            return {
                d: process_eigs(np.linalg.eigvalsh(self._A_r[d]))
                for d in self.domains
            }

        elif source == 'all':
            results = {
                d: process_eigs(np.linalg.eigvalsh(self._A_r[d]))
                for d in self.domains
            }
            try:
                results['global'] = process_eigs(self._get_global_eigenvalues_raw())
            except ValueError:
                pass
            return results

        else:
            raise ValueError(f"Invalid source: {source}. "
                             "Use 'auto', 'global', 'per_domain', or 'all'")

    def _get_global_eigenvalues_raw(self) -> np.ndarray:
        """Get raw eigenvalues from the global (concatenated) system."""
        if self._A_r_global is not None:
            return np.linalg.eigvalsh(self._A_r_global)

        if self._concatenated is not None and self._concatenated.A_coupled is not None:
            return np.linalg.eigvalsh(self._concatenated.A_coupled)

        if self.n_domains == 1:
            return np.linalg.eigvalsh(self._A_r[self.domains[0]])

        print("Auto-concatenating to compute global eigenvalues...")
        self.concatenate()
        return np.linalg.eigvalsh(self._A_r_global)

    def get_resonant_frequencies(
            self,
            domain: str = None,
            n_modes: int = None,
            source: str = 'auto',
            fmin: float = None,
            filter_static: bool = True
    ) -> np.ndarray:
        """
        Get resonant frequencies from eigenvalues.

        Parameters
        ----------
        domain : str, optional
            Specific domain or 'global' (default behavior is global)
        n_modes : int, optional
            Number of modes to return (sorted by frequency)
        source : str
            Same as get_eigenvalues():
            - 'auto': Returns global if available, else single domain
            - 'global': Returns only global eigenvalues
            - 'per_domain': Returns dict of per-domain eigenvalues
            - 'all': Returns dict with both global and per-domain
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
        # Convert fmin (GHz) to min_eigenvalue (ω²)
        if fmin is not None:
            fmin_hz = fmin * 1e9
            min_eigenvalue = (2 * np.pi * fmin_hz) ** 2
            filter_static = True
        else:
            min_eigenvalue = self.DEFAULT_MIN_EIGENVALUE if filter_static else None

        eigs = self.get_eigenvalues(
            domain=domain,
            source=source,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=None  # Don't limit here, do it after freq conversion
        )

        if isinstance(eigs, dict):
            all_eigs = np.concatenate(list(eigs.values()))
        else:
            all_eigs = eigs

        # Eigenvalues are already filtered, but ensure positive for sqrt
        eigs_pos = all_eigs[all_eigs > 0]
        freqs = np.sqrt(eigs_pos) / (2 * np.pi)
        freqs = np.sort(freqs)

        if n_modes is not None:
            freqs = freqs[:n_modes]

        return freqs

    def reduce(
        self,
        tol: float = 1e-6,
        max_rank: Optional[int] = None,
        ranks: Optional[Dict[str, int]] = None
    ) -> 'ModelOrderReduction':
        """
        Perform model reduction for all domains.

        Parameters
        ----------
        tol : float
            SVD truncation tolerance (relative to largest singular value)
        max_rank : int, optional
            Maximum rank for all domains
        ranks : dict, optional
            Per-domain rank: {domain_name: rank}

        Returns
        -------
        self
            For method chaining
        """
        print("\n" + "=" * 60)
        print("Model Order Reduction")
        print("=" * 60)

        # Validate snapshots
        self._validate_snapshots_for_reduction()

        total_full = 0
        total_reduced = 0

        for domain in self.domains:
            print(f"\nDomain: {domain}")

            # Get snapshots for this domain
            if domain in self._snapshots:
                snapshots = self._snapshots[domain]
            elif 'global' in self._snapshots and self.n_domains == 1:
                # Single domain can use global snapshots
                print("  Using global snapshots (single-domain structure)")
                snapshots = self._snapshots['global']
            else:
                # This shouldn't happen if _validate_snapshots_for_reduction passed
                raise ValueError(
                    f"No snapshots for domain '{domain}'. "
                    f"Available: {list(self._snapshots.keys())}"
                )

            M = self._M[domain]
            K = self._K[domain]
            B = self._B[domain]
            n = M.shape[0]

            # Check snapshot dimensions match
            if snapshots.shape[0] != n:
                raise ValueError(
                    f"Snapshot dimension mismatch for domain '{domain}': "
                    f"snapshots have {snapshots.shape[0]} rows, but M has {n} DOFs. "
                    f"This can happen if global snapshots are used for a multi-domain structure. "
                    f"Solution: Re-run solver.solve() with per_domain=True"
                )

            # Determine rank for this domain
            domain_max_rank = max_rank
            if ranks is not None and domain in ranks:
                domain_max_rank = ranks[domain]

            # SVD for POD basis
            U, S, Vt = np.linalg.svd(snapshots, full_matrices=False)
            self._singular_values[domain] = S

            # Determine truncation rank
            r = np.sum(S > tol * S[0])
            r = max(r, 1)
            if domain_max_rank is not None:
                r = min(r, domain_max_rank)

            self._r[domain] = r
            W = U[:, :r]
            self._W[domain] = W

            print(f"  Full DOFs: {n}")
            print(f"  Snapshots: {snapshots.shape[1]}")
            print(f"  Reduced DOFs: {r}")
            print(f"  Compression: {100*(1-r/n):.1f}%")
            print(f"  Singular value decay: {S[0]:.2e} → {S[min(r, len(S)-1)]:.2e}")

            # Project matrices
            M_r = W.T @ M @ W
            K_r = W.T @ K @ W

            # Ensure symmetry
            M_r = (M_r + M_r.T) / 2
            K_r = (K_r + K_r.T) / 2

            # Mass-weighted transformation: A_r = L^{-T} K_r L^{-1}
            lam, Q = sl.eigh(M_r)
            lam = np.maximum(lam, 1e-14 * np.max(lam))

            inv_sqrt_lam = 1.0 / np.sqrt(lam)
            Q_L_inv = Q @ np.diag(inv_sqrt_lam)
            self._Q_L_inv[domain] = Q_L_inv

            # Transformed system
            A_r = Q_L_inv.T @ K_r @ Q_L_inv
            self._A_r[domain] = (A_r + A_r.T) / 2

            # Transformed port basis
            self._B_r[domain] = Q_L_inv.T @ W.T @ B

            total_full += n
            total_reduced += r

        print("\n" + "-" * 60)
        print(f"Total: {total_full} → {total_reduced} DOFs")
        print(f"Overall compression: {100*(1-total_reduced/total_full):.1f}%")
        print("=" * 60)

        self._is_reduced = True

        # For single domain, set global = domain
        if self.n_domains == 1:
            domain = self.domains[0]
            self._A_r_global = self._A_r[domain]
            self._B_r_global = self._B_r[domain]
            self._W_r_global = self._W[domain]
            self._r_global = self._r[domain]

        return self

    def solve(
        self,
        fmin: float,
        fmax: float,
        nsamples: int = 100,
        **kwargs
    ) -> Dict:
        """
        Solve reduced system over frequency range.

        For multi-domain systems, automatically concatenates internal
        domains and returns results for external ports only.

        Parameters
        ----------
        fmin, fmax : float
            Frequency range [GHz]
        nsamples : int
            Number of frequency samples

        Returns
        -------
        dict
            Results with frequencies, Z, S matrices and dicts
        """
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        self.frequencies = np.linspace(fmin, fmax, nsamples) * 1e9

        if self.n_domains == 1:
            return self._solve_single_domain()
        else:
            return self._solve_multi_domain()

    def _solve_single_domain(self) -> Dict:
        """Solve single-domain reduced system."""
        domain = self.domains[0]
        A_r = self._A_r[domain]
        B_r = self._B_r[domain]
        r = self._r[domain]

        n_freq = len(self.frequencies)
        n_ports = B_r.shape[1]

        self._Z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=complex)
        x_r_all = []

        I_exc = np.eye(n_ports)

        for k, freq in enumerate(self.frequencies):
            omega = 2 * np.pi * freq

            lhs = A_r - omega**2 * np.eye(r)
            rhs = omega * B_r @ I_exc

            x_r = np.linalg.solve(lhs, rhs)
            x_r_all.append(x_r)

            self._Z_matrix[k] = 1j * B_r.T @ x_r

        self._compute_s_from_z()
        self._invalidate_cache()

        return {
            'frequencies': self.frequencies,
            'Z': self._Z_matrix,
            'S': self._S_matrix,
            'Z_dict': self.Z_dict,
            'S_dict': self.S_dict,
            'x_r': {domain: np.array(x_r_all)}
        }

    def _solve_multi_domain(self) -> Dict:
        """Solve multi-domain system by auto-concatenating."""
        concat = self.concatenate()
        results = concat.solve(
            self.frequencies[0] / 1e9,
            self.frequencies[-1] / 1e9,
            len(self.frequencies)
        )

        self.frequencies = results['frequencies']
        self._Z_matrix = results['Z']
        self._S_matrix = results['S']
        self._invalidate_cache()

        return results

    # === ROM-specific methods ===

    def get_reduced_structure(self, domain: str = None) -> ReducedStructure:
        """Get reduced structure data for concatenation."""
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if domain is None:
            if self.n_domains == 1:
                domain = self.domains[0]
            else:
                raise ValueError("Specify domain for multi-domain ROM")

        if domain not in self._A_r:
            raise KeyError(f"Domain '{domain}' not found. Available: {self.domains}")

        # Get port modes for this domain's ports
        domain_ports = self.domain_port_map[domain]
        domain_port_modes = {}
        for port in domain_ports:
            if port in self.port_modes:
                domain_port_modes[port] = self.port_modes[port]

        return ReducedStructure(
            Ard=self._A_r[domain],
            Brd=self._B_r[domain],
            ports=domain_ports,
            port_modes=domain_port_modes,
            domain=domain,
            r=self._r[domain],
            n_full=self._M[domain].shape[0]
        )

    def get_all_structures(self) -> List[ReducedStructure]:
        """Get reduced structures for all domains."""
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")
        return [self.get_reduced_structure(d) for d in self.domains]

    def concatenate(
        self,
        others: List['ModelOrderReduction'] = None,
        connections: List[Tuple[Tuple[int, str], Tuple[int, str]]] = None
    ) -> 'ConcatenatedSystem':
        """Concatenate ROMs via port coupling."""
        from solvers.concatenation import ConcatenatedSystem

        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if others is None and self.n_domains > 1:
            structures = self.get_all_structures()
            connections = self._build_sequential_connections()
        elif others is not None:
            all_roms = [self] + list(others)
            structures = []
            for rom in all_roms:
                if not rom._is_reduced:
                    raise ValueError("All ROMs must be reduced before concatenation")
                structures.extend(rom.get_all_structures())

            if connections is None:
                raise ValueError(
                    "Must specify connections when concatenating multiple ROMs"
                )
        else:
            raise ValueError(
                "Single-domain ROM requires 'others' parameter for concatenation"
            )

        concat = ConcatenatedSystem(
            structures=structures,
            port_impedance_func=self._port_impedance_func
        )
        concat.define_connections(connections)
        concat.couple()

        self._concatenated = concat
        self._A_r_global = concat.A_coupled
        self._B_r_global = concat.B_coupled
        self._W_r_global = concat.W_coupled
        self._r_global = concat.A_coupled.shape[0]

        return concat

    def _build_sequential_connections(self) -> List[Tuple]:
        """Build connections for sequential domains."""
        connections = []
        for i in range(self.n_domains - 1):
            ports_i = self.domain_port_map[self.domains[i]]
            ports_next = self.domain_port_map[self.domains[i + 1]]
            connections.append((
                (i, ports_i[-1]),
                (i + 1, ports_next[0])
            ))
        return connections

    def solve_per_domain(
        self,
        fmin: float = None,
        fmax: float = None,
        nsamples: int = None
    ) -> Dict[str, Dict]:
        """Solve each domain independently and return per-domain results."""
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if fmin is not None and fmax is not None and nsamples is not None:
            frequencies = np.linspace(fmin, fmax, nsamples) * 1e9
        elif self.frequencies is not None:
            frequencies = self.frequencies
        else:
            raise ValueError("Must specify frequency range or call solve() first")

        results = {}

        for domain in self.domains:
            A_r = self._A_r[domain]
            B_r = self._B_r[domain]
            r = self._r[domain]
            domain_ports = self.domain_port_map[domain]
            n_ports_domain = len(domain_ports)

            n_freq = len(frequencies)
            Z_d = np.zeros((n_freq, n_ports_domain, n_ports_domain), dtype=complex)
            S_d = np.zeros((n_freq, n_ports_domain, n_ports_domain), dtype=complex)

            I_exc = np.eye(n_ports_domain)

            for k, freq in enumerate(frequencies):
                omega = 2 * np.pi * freq

                lhs = A_r - omega ** 2 * np.eye(r)
                rhs = omega * B_r @ I_exc

                x_r = np.linalg.solve(lhs, rhs)

                Z_d[k] = 1j * B_r.T @ x_r

                Z0_mat = np.diag([
                    np.real(self._get_port_impedance(p, 0, freq))
                    for p in domain_ports
                ])
                S_d[k] = ParameterConverter.z_to_s(Z_d[k], Z0_mat)

            Z_dict = {}
            S_dict = {}
            for i in range(n_ports_domain):
                for j in range(n_ports_domain):
                    key = f'{i + 1}(1){j + 1}(1)'
                    Z_dict[key] = Z_d[:, i, j]
                    S_dict[key] = S_d[:, i, j]

            results[domain] = {
                'frequencies': frequencies,
                'Z': Z_d,
                'S': S_d,
                'Z_dict': Z_dict,
                'S_dict': S_dict,
                'ports': domain_ports
            }

        return results

    def reconstruct_field(
        self,
        x_r: np.ndarray,
        domain: str = None
    ) -> np.ndarray:
        """Reconstruct full field from reduced solution."""
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if domain is None:
            if self.n_domains == 1:
                domain = self.domains[0]
            else:
                raise ValueError("Specify domain for multi-domain structure")

        W = self._W[domain]
        Q_L_inv = self._Q_L_inv[domain]

        return W @ Q_L_inv @ x_r

    # === Properties ===

    @property
    def reduced_dimensions(self) -> Dict[str, int]:
        """Get reduced dimensions per domain."""
        return self._r.copy()

    @property
    def total_dofs(self) -> int:
        """Total full-order DOFs."""
        return sum(self._M[d].shape[0] for d in self.domains)

    @property
    def total_reduced_dofs(self) -> int:
        """Total reduced DOFs (sum of per-domain)."""
        return sum(self._r.values())

    @property
    def global_reduced_dofs(self) -> Optional[int]:
        """Global (concatenated) reduced DOFs."""
        return self._r_global

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (0 to 1)."""
        if not self._is_reduced:
            return 0.0
        return 1 - self.total_reduced_dofs / self.total_dofs

    @property
    def singular_values(self) -> Dict[str, np.ndarray]:
        """Singular values from POD for each domain."""
        return self._singular_values.copy()

    @property
    def has_global_system(self) -> bool:
        """Check if global (concatenated) system is available."""
        return self._A_r_global is not None

    @property
    def concatenated_system(self) -> Optional['ConcatenatedSystem']:
        """Get the underlying ConcatenatedSystem if available."""
        return self._concatenated

    @property
    def A_global(self) -> Optional[np.ndarray]:
        """Global reduced system matrix (same as ConcatenatedSystem.A_coupled)."""
        return self._A_r_global

    @property
    def B_global(self) -> Optional[np.ndarray]:
        """Global reduced port basis (same as ConcatenatedSystem.B_coupled)."""
        return self._B_r_global

    @property
    def W_global(self) -> Optional[np.ndarray]:
        """Global projection basis (same as ConcatenatedSystem.W_coupled)."""
        return self._W_r_global

    # === Visualization ===

    def plot_singular_values(
        self,
        domain: str = None,
        normalized: bool = True,
        **kwargs
    ):
        """Plot singular value decay."""
        import matplotlib.pyplot as plt

        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        domains = [domain] if domain else self.domains
        n_plots = len(domains)

        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4), squeeze=False)

        for i, d in enumerate(domains):
            ax = axes[0, i]
            S = self._singular_values[d]
            if normalized:
                S = S / S[0]

            ax.semilogy(S, 'o-')
            ax.axvline(self._r[d] - 0.5, color='r', linestyle='--',
                       label=f'r={self._r[d]}')
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            ax.set_title(f'{d}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_eigenfrequencies(
        self,
        n_modes: int = 20,
        domain: str = None,
        source: str = 'auto',
        reference: np.ndarray = None,
        reference_label: str = 'Reference',
        filter_static: bool = True,
        **kwargs
    ):
        """Plot eigenfrequencies (resonant frequencies)."""
        import matplotlib.pyplot as plt

        freqs = self.get_resonant_frequencies(
            domain=domain,
            n_modes=n_modes,
            source=source,
            filter_static=filter_static
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(range(len(freqs)), freqs / 1e9, marker='o', s=50, label='ROM')

        if reference is not None:
            ref_freqs = np.sort(reference)[:n_modes]
            for f in ref_freqs:
                ax.axhline(f / 1e9, color='red', alpha=0.5, linewidth=0.8)
            ax.plot([], [], 'r-', label=reference_label)

        ax.set_xlabel('Mode Index')
        ax.set_ylabel('Frequency (GHz)')
        ax.set_title(f'Resonant Frequencies (first {len(freqs)} modes)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def print_info(self) -> None:
        """Print ROM information."""
        print("\n" + "=" * 60)
        print("ModelOrderReduction Information")
        print("=" * 60)
        print(f"Structure: {'Compound' if self.n_domains > 1 else 'Single'}")
        print(f"Domains: {self.domains}")
        print(f"External ports: {self._external_ports}")
        if self.n_domains > 1:
            print(f"All ports: {self._all_ports}")

        print(f"\nSnapshots available: {list(self._snapshots.keys())}")

        if self._is_reduced:
            print("\nPer-domain reduction:")
            for domain in self.domains:
                n = self._M[domain].shape[0]
                r = self._r[domain]
                ports = self.domain_port_map[domain]
                print(f"  {domain}: {n} → {r} DOFs, ports: {ports}")

            print(f"\nTotal: {self.total_dofs} → {self.total_reduced_dofs} DOFs")
            print(f"Compression: {100*self.compression_ratio:.1f}%")

            if self.has_global_system:
                print(f"\nGlobal system:")
                print(f"  Global reduced DOFs: {self._r_global}")
                print(f"  A_global shape: {self._A_r_global.shape}")
                print(f"  B_global shape: {self._B_r_global.shape}")
                if self._concatenated is not None:
                    print(f"  External ports: {self._concatenated.ports}")
        else:
            print("\nNot yet reduced. Call reduce() first.")

        if self.frequencies is not None:
            print(f"\nSolution available:")
            print(f"  Frequency range: {self.frequencies[0]/1e9:.4f} - "
                  f"{self.frequencies[-1]/1e9:.4f} GHz")
            print(f"  Number of samples: {len(self.frequencies)}")

        print("=" * 60)

    def print_eigenfrequency_comparison(
        self,
        reference: np.ndarray,
        n_modes: int = 10,
        reference_label: str = 'Reference',
        filter_static: bool = True
    ) -> None:
        """Print eigenfrequency comparison table."""
        rom_freqs = self.get_resonant_frequencies(n_modes=n_modes, filter_static=filter_static)
        ref_freqs = np.sort(reference)[:n_modes]

        print("\n" + "=" * 60)
        print("Eigenfrequency Comparison")
        print("=" * 60)
        print(f"{'Mode':<6}{reference_label:<15}{'ROM':<15}{'Error (%)':<12}")
        print("-" * 48)

        n_compare = min(n_modes, len(rom_freqs), len(ref_freqs))
        for i in range(n_compare):
            err = abs(rom_freqs[i] - ref_freqs[i]) / ref_freqs[i] * 100
            print(f"{i:<6}{ref_freqs[i]/1e9:<15.4f}{rom_freqs[i]/1e9:<15.4f}{err:<12.2f}")

        print("=" * 60)