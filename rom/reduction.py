from __future__ import annotations
"""Model Order Reduction using Proper Orthogonal Decomposition."""

from typing import Tuple, Optional, Dict, List, Union, Literal
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from solvers.eigen_mixin import ROMEigenMixin

from solvers.base import BaseEMSolver, ParameterConverter
from utils.plot_mixin import PlotMixin
from rom.structures import ReducedStructure
from ngsolve import GridFunction, Norm, curl, BoundaryFromVolumeCF, HCurl
from ngsolve.webgui import Draw
from core.constants import mu0
from core.persistence import H5Serializer, ProjectManager
import h5py
import json
from pathlib import Path
from datetime import datetime
import cavsim3d.utils.printing as pr


class ModelOrderReduction(BaseEMSolver, ROMEigenMixin, PlotMixin):
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

    # Threshold: below this matrix dimension, use direct solve (LU is fast);
    # above, use iterative (GMRES handles large/sparse systems better).
    ITERATIVE_SIZE_THRESHOLD = 500

    @property
    def project_sub_path(self) -> Path:
        """Relative path from project root: fds/foms/roms or fds/fom/rom."""
        parent_path = self.solver.project_sub_path
        if self.n_domains > 1:
            return parent_path / "roms"
        return parent_path / "rom"

    @property
    def _project_path(self) -> Optional[str]:
        """Proxy project path from parent solver."""
        if self.solver is not None:
            return getattr(self.solver, '_project_path', None)
        return None

    def __init__(self, solver):
        """
        Initialize from FrequencyDomainSolver.
        """
        super().__init__()

        self.solver = solver
        self.mesh = solver.mesh
        self.order = getattr(solver, 'order', 3)
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

        # Caches
        self._resonant_mode_cache = {}

        # Snapshot storage for field reconstruction
        self._x_r_snapshots: Optional[Dict[str, np.ndarray]] = None

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
            pr.warning("No matrices found in solver. Ensure assemble_matrices() was called.")
            return
        if not self.solver.snapshots:
            pr.warning("No snapshots found in solver. Ensure solve(..., store_snapshots=True) was called.")
            if self.solver.n_domains > 1:
                pr.warning("Multi-domain structure detected but only global snapshots available.")
                pr.warning("         For multi-domain ROM, re-run solve() with per_domain=True:")
                pr.warning("         solver.solve(fmin, fmax, nsamples, per_domain=True, store_snapshots=True)")

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

    # =========================================================================
    # BaseEMSolver abstract implementations
    # =========================================================================

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

    # ------------------------------------------------------------------
    # Logical Matrix Access (Reduced)
    # ------------------------------------------------------------------

    @property
    def A(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Access the reduced system matrix.
        Returns the global coupled matrix if available, otherwise a dictionary of per-domain matrices.
        """
        if self._A_r_global is not None:
            return self._A_r_global
        return self._A_r

    @property
    def B(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Access the reduced port basis (mass-weighted).
        Returns the global coupled matrix if available, otherwise a dictionary of per-domain matrices.
        """
        if self._B_r_global is not None:
            return self._B_r_global
        return self._B_r

    @property
    def W(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Access the POD projection basis.
        Returns the global coupled basis if available, otherwise a dictionary of per-domain bases.
        """
        if self._W_r_global is not None:
            return self._W_r_global
        return self._W

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

    def calculate_resonant_modes(
            self,
            domain: str = None,
            source: str = 'auto',
            filter_static: bool = True,
            min_eigenvalue: float = None,
            n_modes: int = None
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute eigenvalues and eigenvectors for the reduced system.
        """
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        # Check cache
        cache_params = {
            'domain': domain,
            'source': source,
            'filter_static': filter_static,
            'min_eigenvalue': min_eigenvalue,
            'n_modes': n_modes
        }
        cache_key = tuple(sorted(cache_params.items()))
        if cache_key in self._resonant_mode_cache:
            return self._resonant_mode_cache[cache_key]

        def process_modes(raw_eigs: np.ndarray, raw_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Use FrequencyDomainSolver's filter helper
            from solvers.frequency_domain import FrequencyDomainSolver
            return FrequencyDomainSolver._filter_eigenvalues(raw_eigs, raw_vecs, filter_static, min_eigenvalue, n_modes)

        def get_raw_modes(A):
             eigs, vecs = np.linalg.eigh(A)
             return eigs, vecs

        # Specific domain requested
        if domain is not None:
            if domain == 'global':
                raw_eigs, raw_vecs = get_raw_modes(self._get_global_matrix_raw())
            elif domain in self._A_r:
                raw_eigs, raw_vecs = get_raw_modes(self._A_r[domain])
            else:
                raise KeyError(f"Domain '{domain}' not found. "
                               f"Available: {list(self._A_r.keys())} + ['global']")
            res = process_modes(raw_eigs, raw_vecs)
            self._resonant_mode_cache[cache_key] = res
            return res

        # Auto-detect based on source parameter
        if source == 'auto' or source == 'global':
            raw_eigs, raw_vecs = get_raw_modes(self._get_global_matrix_raw())
            res = process_modes(raw_eigs, raw_vecs)
            self._resonant_mode_cache[cache_key] = res
            return res

        elif source == 'per_domain':
            res = {
                d: process_modes(*get_raw_modes(self._A_r[d]))
                for d in self.domains
            }
            self._resonant_mode_cache[cache_key] = res
            return res

        elif source == 'all':
            results = {
                d: process_modes(*get_raw_modes(self._A_r[d]))
                for d in self.domains
            }
            try:
                results['global'] = process_modes(*get_raw_modes(self._get_global_matrix_raw()))
            except (ValueError, RuntimeError):
                pass
            self._resonant_mode_cache[cache_key] = results
            return results

        else:
            raise ValueError(f"Invalid source: {source}. "
                             "Use 'auto', 'global', 'per_domain', or 'all'")

    def get_eigenvalues(self, **kwargs):
        """Deprecated alias for calculate_resonant_modes."""
        import warnings
        warnings.warn("get_eigenvalues() is deprecated. Use calculate_resonant_modes() instead.",
                      DeprecationWarning, stacklevel=2)
        res = self.calculate_resonant_modes(**kwargs)
        if isinstance(res, dict):
            return {k: v[0] for k, v in res.items()}
        return res[0]

    def _get_global_matrix_raw(self, auto_concatenate: bool = False) -> np.ndarray:
        """Get raw global (concatenated) matrix."""
        if self._A_r_global is not None:
            return self._A_r_global

        if self._concatenated is not None and self._concatenated.A_coupled is not None:
            return self._concatenated.A_coupled

        if self.n_domains == 1:
            return self._A_r[self.domains[0]]

        if not auto_concatenate:
            raise RuntimeError("Global matrix not available. Call concatenate() first.")

        pr.debug("Auto-concatenating to get global matrix...")
        self.concatenate()
        return self._A_r_global

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

        modes = self.calculate_resonant_modes(
            domain=domain,
            source=source,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=None  # Don't limit here, do it after freq conversion
        )

        if isinstance(modes, dict):
            all_eigs = np.concatenate([v[0] for v in modes.values()])
        else:
            all_eigs = modes[0]

        # Eigenvalues are already filtered, but ensure positive for sqrt
        eigs_pos = all_eigs[all_eigs > 0]
        freqs = np.sqrt(eigs_pos) / (2 * np.pi)
        freqs = np.sort(freqs)

        if n_modes is not None:
            freqs = freqs[:n_modes]

        return freqs

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """
        Compute or retrieve eigenmodes for the reduced structure(s).
        """
        res = self.calculate_resonant_modes(**kwargs)
        
        # Hierarchical save
        if _auto_save:
            self._auto_save_eigenmodes(res, **kwargs)
        
        return res

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        if not hasattr(self, 'solver') or not getattr(self.solver, '_project_path', None):
            return
        
        save_path = Path(self.solver._project_path) / self.project_sub_path / "eigenmodes"
        try:
            self.save_eigenmodes(save_path, **kwargs)
        except Exception as e:
            pr.warning(f"Could not auto-save eigenmodes for ModelOrderReduction to {save_path}: {e}")

    # =========================================================================
    # Model Reduction
    # =========================================================================

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
        pr.running("\n" + "=" * 60)
        pr.running("Model Order Reduction")
        pr.running("=" * 60)

        # Validate snapshots
        self._validate_snapshots_for_reduction()

        total_full = 0
        total_reduced = 0

        for domain in self.domains:
            pr.info(f"\nDomain: {domain}")

            # Get snapshots for this domain
            if domain in self._snapshots:
                snapshots = self._snapshots[domain]
            elif 'global' in self._snapshots and self.n_domains == 1:
                # Single domain can use global snapshots
                pr.debug("  Using global snapshots (single-domain structure)")
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

            pr.debug(f"  Full DOFs: {n}")
            pr.debug(f"  Snapshots: {snapshots.shape[1]}")
            pr.debug(f"  Reduced DOFs: {r}")
            pr.debug(f"  Compression: {100*(1-r/n):.1f}%")
            pr.debug(f"  Singular value decay: {S[0]:.2e} → {S[min(r, len(S)-1)]:.2e}")

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

        pr.done(f"Total: {total_full} → {total_reduced} DOFs")
        pr.done(f"Overall compression: {100*(1-total_reduced/total_full):.1f}%")
        pr.done("=" * 60)

        self._is_reduced = True

        # For single domain, set global = domain
        if self.n_domains == 1:
            domain = self.domains[0]
            self._A_r_global = self._A_r[domain]
            self._B_r_global = self._B_r[domain]
            self._W_r_global = self._W[domain]
            self._r_global = self._r[domain]

        # Automatic save after reduction
        if hasattr(self.solver, '_project_ref') and self.solver._project_ref:
            self.solver._project_ref.save()

        return self

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
        Solve reduced system over frequency range.

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
            Individual solve parameters (solver_type, rerun, etc.)
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
        solver_type = cfg.get('solver_type', 'auto')
        rerun = cfg.get('rerun', False)
        verbose = cfg.get('verbose', False)

        # Set verbosity level
        pr.set_verbosity(verbose)

        # Clear results if rerunning
        if rerun:
            self._Z_matrix = None
            self._S_matrix = None
            self._x_r_snapshots = None
            self._invalidate_cache()

        # --- Rerun protection ---
        has_results = (self._Z_matrix is not None)
        
        # Check disk if in-memory is missing
        if not has_results and not rerun and self.solver and hasattr(self.solver, '_project_path'):
            sub_folder = "fom/rom" if self.n_domains == 1 else "foms/roms"
            rom_dir = Path(self.solver._project_path) / "fds" / sub_folder
            z_path = rom_dir / "z" / "z.h5"
            if z_path.exists():
                try:
                    with h5py.File(z_path, "r") as f:
                        self._Z_matrix = H5Serializer.load_dataset(f["data"])
                    s_path = rom_dir / "s" / "s.h5"
                    if s_path.exists():
                        with h5py.File(s_path, "r") as f:
                            self._S_matrix = H5Serializer.load_dataset(f["data"])
                    
                    # Load frequencies
                    snap_path = rom_dir / "snapshots" / "snapshots.h5"
                    if not snap_path.exists(): snap_path = rom_dir / "snapshots.h5"
                    if snap_path.exists():
                        with h5py.File(snap_path, "r") as f:
                            self.frequencies = H5Serializer.load_dataset(f["frequencies"])
                    
                    has_results = True
                    pr.info(f"  Loaded existing ROM results from {rom_dir}")
                except Exception as e:
                    pr.warning(f"  Could not load existing ROM results: {e}")

        if has_results and not rerun:
            import warnings
            warnings.warn(
                "Results already exist for this ROM solver. "
                "To overwrite, call solve(..., rerun=True).",
                UserWarning,
                stacklevel=2,
            )
            return self._build_results_dict()

        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        self.frequencies = np.linspace(fmin, fmax, nsamples) * 1e9

        if self.n_domains == 1:
            return self._solve_single_domain(solver_type=solver_type)
        else:
            return self._solve_multi_domain(solver_type=solver_type)

    def _build_results_dict(self) -> Dict:
        """Build results dictionary for reduced solver."""
        return {
            'frequencies': self.frequencies,
            'Z': self._Z_matrix,
            'S': self._S_matrix,
            'Z_dict': self.Z_dict,
            'S_dict': self.S_dict,
            'x_r': getattr(self, '_x_r_snapshots', None),
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Union[str, Path]):
        """
        Save ModelOrderReduction data to disk.
        
        Saves reduced matrices (A_r, B_r, W, Q_L_inv) to separate files in matrices/
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

        # 1. Save reduced and projection matrices to modular files
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(mat_path / "A_r.h5", "a") as fa, \
             h5py.File(mat_path / "B_r.h5", "a") as fb, \
             h5py.File(mat_path / "W.h5", "a") as fw, \
             h5py.File(mat_path / "Q_L_inv.h5", "a") as fq:
            for domain in self.domains:
                if domain in self._A_r:
                    # Save with domain suffix for modularity
                    H5Serializer.save_dataset(fa, domain, self._A_r.get(domain))
                    H5Serializer.save_dataset(fb, domain, self._B_r.get(domain))
                    H5Serializer.save_dataset(fw, domain, self._W.get(domain))
                    H5Serializer.save_dataset(fq, domain, self._Q_L_inv.get(domain))
                    
                    # Also save individual files for user-friendly access
                    for mname, mdict in [("A_r", self._A_r), ("B_r", self._B_r), ("W", self._W), ("Q_L_inv", self._Q_L_inv)]:
                         with h5py.File(mat_path / f"{mname}_{domain}.h5", "a") as f_indiv:
                             H5Serializer.save_dataset(f_indiv, "data", mdict.get(domain))

        # 2. Save S and Z results
        if self._Z_matrix is not None:
             z_file = "z.h5" 
             if self.n_domains == 1: z_file = f"z_{self.domains[0]}.h5"
             with h5py.File(z_path_dir / z_file, "a") as f:
                H5Serializer.save_dataset(f, "data", self._Z_matrix)
        if self._S_matrix is not None:
             s_file = "s.h5"
             if self.n_domains == 1: s_file = f"s_{self.domains[0]}.h5"
             with h5py.File(s_path_dir / s_file, "a") as f:
                H5Serializer.save_dataset(f, "data", self._S_matrix)

        # 3. Save snapshots and frequencies
        snap_file = "snapshots.h5"
        if self.n_domains == 1: snap_file = f"snapshots_{self.domains[0]}.h5"
        
        with h5py.File(snap_path_dir / snap_file, "a") as f:
            if self.frequencies is not None:
                H5Serializer.save_dataset(f, "frequencies", self.frequencies)
            if self._x_r_snapshots is not None:
                group = f.require_group("x_r_snapshots")
                for domain, snapshots_data in self._x_r_snapshots.items():
                    H5Serializer.save_dataset(group, domain, snapshots_data)

        # 4. Save eigenmodes
        self.save_eigenmodes()
        
        # 5. Save metadata
        metadata = {
            "domains": self.domains,
            "n_domains": self.n_domains,
            "is_reduced": self._is_reduced,
            "r": self._r,
            "n_ports_total": self._n_ports_total,
            "n_ports_external": self._n_ports_external,
            "n_modes_per_port": self._n_modes_per_port,
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)
        
        # 5. Save cached concatenation if available
        if self._concatenated is not None:
            self._concatenated.save(path / "concat")

    @classmethod
    def load(cls, path: Union[str, Path], solver=None) -> ModelOrderReduction:
        """Load ModelOrderReduction from disk."""
        path = Path(path)
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # solver reference can be passed if we want to link it back
        # If solver is None, we create a skeleton ROM
        rom = cls.__new__(cls)
        # Manually initialize minimal state
        rom.solver = solver
        rom.domains = metadata["domains"]
        rom.n_domains = metadata["n_domains"]
        rom._is_reduced = metadata["is_reduced"]
        rom._r = metadata["r"]
        rom._n_ports_total = metadata["n_ports_total"]
        rom._n_ports_external = metadata["n_ports_external"]
        rom._n_modes_per_port = metadata["n_modes_per_port"]
        
        # Initialize caches
        rom._resonant_mode_cache = {}
        
        # Initialize result dicts for PlotMixin
        rom._S_dict = None
        rom._Z_dict = None
        
        # We also need port lists and mappings which are usually in solver
        # If solver is None, some functionality might be limited
        if solver:
            rom.mesh = solver.mesh
            rom._all_ports = solver.all_ports
            rom._external_ports = solver.external_ports
            rom.domain_port_map = solver.domain_port_map
            rom._port_impedance_func = solver._get_port_impedance
            rom.port_modes = solver.port_modes
        else:
            rom.mesh = None
            rom._all_ports = []
            rom._external_ports = []
            rom.domain_port_map = {}
            rom.port_modes = {}
            
        rom._M = {}
        rom._K = {}
        rom._B = {}
        rom._snapshots = {}
        rom._W = {}
        rom._A_r = {}
        rom._B_r = {}
        rom._Q_L_inv = {}
        rom._singular_values = {}

        # 1. Load matrices from modular files or legacy matrices.h5
        mat_path = path / "matrices"
        if mat_path.exists():
             for mname in ["A_r", "B_r", "W", "Q_L_inv"]:
                 target_dict = getattr(rom, f"_{mname}")
                 mfile_agg = mat_path / f"{mname}.h5"
                 
                 # Try individual files first, then fallback to aggregated
                 for domain in rom.domains:
                     mfile_indiv = mat_path / f"{mname}_{domain}.h5"
                     if mfile_indiv.exists():
                         with h5py.File(mfile_indiv, "r") as f:
                             target_dict[domain] = H5Serializer.load_dataset(f["data"])
                     elif mfile_agg.exists():
                         with h5py.File(mfile_agg, "r") as f:
                             if domain in f:
                                 target_dict[domain] = H5Serializer.load_dataset(f[domain])
        elif (path / "matrices.h5").exists():
            with h5py.File(path / "matrices.h5", "r") as f:
                for domain in rom.domains:
                    if domain in f:
                        group = f[domain]
                        rom._A_r[domain] = H5Serializer.load_dataset(group["A_r"])
                        rom._B_r[domain] = H5Serializer.load_dataset(group["B_r"])
                        rom._W[domain] = H5Serializer.load_dataset(group["W"])
                        rom._Q_L_inv[domain] = H5Serializer.load_dataset(group["Q_L_inv"])

        # 2. Load S/Z results
        rom._Z_matrix = None
        rom._S_matrix = None
        
        z_files = ["z.h5"]
        if rom.n_domains == 1: z_files.insert(0, f"z_{rom.domains[0]}.h5")
        for zf in z_files:
            zp = path / "z" / zf
            if not zp.exists(): zp = path / zf
            if zp.exists():
                with h5py.File(zp, "r") as f:
                    rom._Z_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None
                    if rom._Z_matrix is not None: break
        
        s_files = ["s.h5"]
        if rom.n_domains == 1: s_files.insert(0, f"s_{rom.domains[0]}.h5")
        for sf in s_files:
            sp = path / "s" / sf
            if not sp.exists(): sp = path / sf
            if sp.exists():
                with h5py.File(sp, "r") as f:
                    rom._S_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None
                    if rom._S_matrix is not None: break

        # 3. Load snapshots and frequencies
        rom.frequencies = None
        rom._x_r_snapshots = {}
        snap_files = ["snapshots.h5"]
        if rom.n_domains == 1: snap_files.insert(0, f"snapshots_{rom.domains[0]}.h5")
        
        for snap_f in snap_files:
            snapp = path / "snapshots" / snap_f
            if not snapp.exists(): snapp = path / snap_f
            if snapp.exists():
                with h5py.File(snapp, "r") as f:
                    if rom.frequencies is None:
                        rom.frequencies = H5Serializer.load_dataset(f["frequencies"]) if "frequencies" in f else None
                    
                    if "x_r_snapshots" in f:
                        group = f["x_r_snapshots"]
                        if isinstance(group, h5py.Group):
                            for domain in rom.domains:
                                if domain in group:
                                    rom._x_r_snapshots[domain] = H5Serializer.load_dataset(group[domain])
                        else:
                            rom._x_r_snapshots = H5Serializer.load_dataset(group)
                    
                    # Support legacy loading of Z/S
                    if rom._Z_matrix is None and "Z_matrix" in f:
                        rom._Z_matrix = H5Serializer.load_dataset(f["Z_matrix"])
                    if rom._S_matrix is None and "S_matrix" in f:
                        rom._S_matrix = H5Serializer.load_dataset(f["S_matrix"])

        # 4. Load eigenmodes
        rom.load_eigenmodes()

        # 5. Load concatenated system if it exists
        concat_path = path / "concat"
        rom._concatenated = None
        if concat_path.exists():
            from solvers.concatenation import ConcatenatedSystem
            try:
                rom._concatenated = ConcatenatedSystem.load(concat_path, solver_ref=solver)
            except Exception as e:
                print(f"Warning: Could not load concatenated system from {concat_path}: {e}")

        return rom

    def _solve_single_domain(self, solver_type: str = 'auto') -> Dict:
        """Solve single-domain reduced system."""
        domain = self.domains[0]
        A_r = self._A_r[domain]
        B_r = self._B_r[domain]
        r = self._r[domain]

        # Resolve 'auto' solver type
        if solver_type == 'auto':
            solver_type = 'iterative' if r >= self.ITERATIVE_SIZE_THRESHOLD else 'direct'
            pr.debug(f"  Solver: {solver_type} (system size {r})")

        n_freq = len(self.frequencies)
        n_ports = B_r.shape[1]

        self._Z_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=complex)

        import time
        t0 = time.time()

        omegas = 2 * np.pi * self.frequencies  # (n_freq,)

        if solver_type in ('auto', 'direct'):
            # ============================================================
            # Eigendecomposition approach (fast for reduced systems)
            # A = V Λ V^{-1}  →  (A - ω²I)^{-1} = V diag(1/(λ-ω²)) V^{-1}
            # Z = jω B^T V diag(1/(λ-ω²)) V^{-1} B
            # ============================================================
            is_hermitian = np.allclose(A_r, A_r.T.conj(), atol=1e-10)

            if is_hermitian:
                eigenvalues, V = np.linalg.eigh(A_r)
                Vinv_B = V.T.conj() @ B_r   # V^H B = V^{-1} B
            else:
                eigenvalues, V = np.linalg.eig(A_r)
                Vinv_B = np.linalg.solve(V, B_r)  # V^{-1} B

            D = B_r.T @ V  # B^T V, shape (n_ports, r)

            # d[k, i] = 1 / (λ_i - ω_k²)
            d = 1.0 / (eigenvalues[None, :] - omegas[:, None]**2)

            # Z[k] = jω D diag(d[k]) (V^{-1} B)
            for k in range(n_freq):
                self._Z_matrix[k] = 1j * omegas[k] * (D * d[k, :]) @ Vinv_B

            # Snapshots: x_r[k] = ω V diag(d[k]) V^{-1} B
            x_r_all = []
            for k in range(n_freq):
                x_r_all.append(omegas[k] * V @ (d[k, :, None] * Vinv_B))

        else:
            # ============================================================
            # Iterative solver (GMRES)
            # ============================================================
            I_exc = np.eye(n_ports)
            x_r_all = []
            gmres_failures = 0

            for k, freq in enumerate(self.frequencies):
                omega = omegas[k]
                lhs = A_r - omega**2 * np.eye(r)
                rhs = omega * B_r @ I_exc

                lhs_sp = sp.csr_matrix(lhs)
                x_r = np.zeros_like(rhs)
                for col in range(rhs.shape[1]):
                    x_r[:, col], info = spla.gmres(lhs_sp, rhs[:, col])
                    if info != 0:
                        gmres_failures += 1

                x_r_all.append(x_r)
                self._Z_matrix[k] = 1j * B_r.T @ x_r

            if gmres_failures > 0:
                total_solves = n_freq * n_ports
                pr.warning(f"  GMRES: {gmres_failures}/{total_solves} solves did NOT converge.")

        t1 = time.time()
        pr.done(f"  Solve loop: {t1 - t0:.3f}s ({n_freq} freq points)")

        # ================================================================
        # Store reduced snapshots for field reconstruction
        # Shape: (n_freq, r, n_port_modes)
        # ================================================================
        self._x_r_snapshots = {domain: np.array(x_r_all)}

        self._compute_s_from_z()
        self._invalidate_cache()

        # Automatic save after simulation
        if hasattr(self.solver, '_project_ref') and self.solver._project_ref:
            self.solver._project_ref.save()

        return {
            'frequencies': self.frequencies,
            'Z': self._Z_matrix,
            'S': self._S_matrix,
            'Z_dict': self.Z_dict,
            'S_dict': self.S_dict,
            'x_r': self._x_r_snapshots,
        }

    def _solve_multi_domain(self, solver_type: str = 'auto') -> Dict:
        """Solve multi-domain system by auto-concatenating."""
        concat = self.concatenate()
        results = concat.solve(
            self.frequencies[0] / 1e9,
            self.frequencies[-1] / 1e9,
            len(self.frequencies),
            solver_type=solver_type,
        )

        self.frequencies = results['frequencies']
        self._Z_matrix = results['Z']
        self._S_matrix = results['S']
        self._invalidate_cache()

        return results

    # =========================================================================
    # ROM-specific methods
    # =========================================================================

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

        domain_ports = self.domain_port_map[domain]
        domain_port_modes = {p: self.port_modes[p] for p in domain_ports if p in self.port_modes}

        # Get FES and mesh for this domain
        fes = None
        mesh = self.mesh
        if hasattr(self.solver, '_fes'):
            fes = self.solver._fes.get(domain)

        return ReducedStructure(
            Ard=self._A_r[domain],
            Brd=self._B_r[domain],
            ports=domain_ports,
            port_modes=domain_port_modes,
            domain=domain,
            r=self._r[domain],
            n_full=self._M[domain].shape[0],
            W=self._W[domain],
            Q_L_inv=self._Q_L_inv[domain],
            fes=fes,
            mesh=mesh,
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

        # Pass mesh and solver_ref for field reconstruction
        concat = ConcatenatedSystem(
            structures=structures,
            mesh=self.mesh,
            port_impedance_func=self._port_impedance_func,
            solver_ref=self,
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

        n_modes = self._n_modes_per_port or 1
        results = {}

        for domain in self.domains:
            A_r = self._A_r[domain]
            B_r = self._B_r[domain]
            r = self._r[domain]
            domain_ports = self.domain_port_map[domain]
            n_ports_domain = len(domain_ports)

            # B_r columns = n_ports_domain * n_modes  (all port-mode combos)
            n_pm = B_r.shape[1]

            n_freq = len(frequencies)
            Z_d = np.zeros((n_freq, n_pm, n_pm), dtype=complex)
            S_d = np.zeros((n_freq, n_pm, n_pm), dtype=complex)

            I_exc = np.eye(n_pm)

            for k, freq in enumerate(frequencies):
                omega = 2 * np.pi * freq

                lhs = A_r - omega ** 2 * np.eye(r)
                rhs = omega * B_r @ I_exc

                x_r = np.linalg.solve(lhs, rhs)

                Z_d[k] = 1j * B_r.T @ x_r

                # Build impedance matrix for all port-mode combinations
                Z0_diag = []
                for p_idx, p in enumerate(domain_ports):
                    for m in range(n_modes):
                        Z0_diag.append(
                            np.real(self._get_port_impedance(p, m, freq))
                        )
                Z0_mat = np.diag(Z0_diag)
                S_d[k] = ParameterConverter.z_to_s(Z_d[k], Z0_mat)

            # Build dicts with proper port(mode) keys
            Z_dict = {}
            S_dict = {}
            for i in range(n_pm):
                pi = i // n_modes + 1
                mi = i % n_modes + 1
                for j in range(n_pm):
                    pj = j // n_modes + 1
                    mj = j % n_modes + 1
                    key = f'{pi}({mi}){pj}({mj})'
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

    # =========================================================================
    # Field Reconstruction
    # =========================================================================

    def _ensure_fes(self, domain: str = None) -> HCurl:
        """Ensure FES is available for field reconstruction."""
        if domain is None:
            if self.n_domains == 1:
                domain = self.domains[0]
            else:
                raise ValueError("Specify domain for multi-domain structure")

        # Try to get per-domain FES
        if hasattr(self.solver, '_fes') and isinstance(self.solver._fes, dict):
            fes = self.solver._fes.get(domain)
            if fes is not None:
                return fes

        # Try global FES for single domain
        if self.n_domains == 1:
            if hasattr(self.solver, '_fes_global') and self.solver._fes_global is not None:
                return self.solver._fes_global
            if hasattr(self.solver, 'fes') and self.solver.fes is not None:
                return self.solver.fes

        # Create FES if we have mesh
        if self.mesh is not None:
            order = getattr(self.solver, 'order', 3)
            bc = getattr(self.solver, 'bc', 'default')
            fes = HCurl(self.mesh, order=order, complex=True, dirichlet=bc)
            pr.debug(f"  Created FES for {domain}: {fes.ndof} DOFs")
            return fes

        raise ValueError(
            f"No FES available for domain '{domain}'. "
            "Ensure solver has _fes dict or provide mesh."
        )

    def _get_snapshot_for_excitation(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int,
        domain: str
    ) -> np.ndarray:
        """
        Get reduced solution snapshot for a given excitation.

        Returns the reduced coordinates x_r for the specified frequency and excitation.
        """
        # For multi-domain, delegate to concatenated system
        if self.n_domains > 1:
            if self._concatenated is None:
                raise ValueError(
                    "Multi-domain ROM requires concatenation. Call solve() first."
                )
            # The concatenated system handles its own snapshots
            raise NotImplementedError(
                "Use concatenated_system.plot_field() for multi-domain structures"
            )

        # Single domain case
        # Check if we have stored reduced snapshots from solve()
        if self._x_r_snapshots is None:
            raise ValueError(
                "No reduced solution snapshots available. "
                "Call solve() first to generate snapshots."
            )

        domain_snapshots = self._x_r_snapshots.get(domain)
        if domain_snapshots is None:
            raise ValueError(f"No snapshots for domain '{domain}'")

        # Get port index
        domain_ports = self.domain_port_map[domain]
        if excitation_port not in domain_ports:
            raise ValueError(
                f"Port '{excitation_port}' not in domain '{domain}'. "
                f"Available: {domain_ports}"
            )
        port_idx = domain_ports.index(excitation_port)

        # Compute column index: port_idx * n_modes + excitation_mode
        n_modes = self._n_modes_per_port or 1
        col_idx = port_idx * n_modes + excitation_mode

        # domain_snapshots shape: (n_freq, r, n_port_modes)
        if freq_idx >= domain_snapshots.shape[0]:
            raise ValueError(
                f"freq_idx {freq_idx} out of range [0, {domain_snapshots.shape[0] - 1}]"
            )
        if col_idx >= domain_snapshots.shape[2]:
            raise ValueError(
                f"Excitation column {col_idx} out of range. "
                f"port_idx={port_idx}, mode={excitation_mode}"
            )

        return domain_snapshots[freq_idx, :, col_idx]

    def reconstruct_field(
        self,
        x_r: np.ndarray = None,
        freq_idx: int = None,
        excitation_port: str = None,
        excitation_mode: int = 0,
        domain: str = None
    ) -> np.ndarray:
        """
        Reconstruct full field from reduced solution.

        Can be called with either:
        - x_r: directly provide reduced solution vector
        - freq_idx + excitation_port: use stored snapshot

        Parameters
        ----------
        x_r : ndarray, optional
            Reduced solution vector. If None, uses freq_idx and excitation_port.
        freq_idx : int, optional
            Frequency index (required if x_r is None)
        excitation_port : str, optional
            Excitation port name (required if x_r is None)
        excitation_mode : int
            Mode index for excitation
        domain : str, optional
            Domain name (auto-detected for single domain)

        Returns
        -------
        x_full : ndarray
            Full-order solution vector
        """
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if domain is None:
            if self.n_domains == 1:
                domain = self.domains[0]
            else:
                raise ValueError("Specify domain for multi-domain structure")

        # Get x_r from snapshots if not provided
        if x_r is None:
            if freq_idx is None or excitation_port is None:
                raise ValueError(
                    "Either provide x_r directly, or specify freq_idx and excitation_port"
                )
            x_r = self._get_snapshot_for_excitation(
                freq_idx, excitation_port, excitation_mode, domain
            )

        W = self._W[domain]
        Q_L_inv = self._Q_L_inv[domain]

        return W @ (Q_L_inv @ x_r)

    def _reconstruct_field_gf(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int = 0,
        domain: str = None
    ) -> GridFunction:
        """
        Reconstruct E-field GridFunction from reduced solution.

        Parameters
        ----------
        freq_idx : int
            Frequency index
        excitation_port : str
            Name of the excited port
        excitation_mode : int
            Mode index of excitation
        domain : str, optional
            Domain name (required for multi-domain, auto-detected for single)

        Returns
        -------
        E_gf : GridFunction
            Reconstructed electric field
        """
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        # Handle multi-domain case
        if self.n_domains > 1:
            if self._concatenated is None:
                raise ValueError(
                    "Multi-domain ROM: call solve() first, then use "
                    "concatenated_system.plot_field() for visualization."
                )
            # Delegate to concatenated system
            return self._concatenated._reconstruct_field(
                freq_idx, excitation_port, excitation_mode
            )

        # Single domain case
        if domain is None:
            domain = self.domains[0]

        # Get FES
        fes = self._ensure_fes(domain)

        # Get reduced solution
        x_r = self._get_snapshot_for_excitation(
            freq_idx, excitation_port, excitation_mode, domain
        )

        # Reconstruct full solution: x_full = W @ Q_L_inv @ x_r
        W = self._W[domain]
        Q_L_inv = self._Q_L_inv[domain]
        x_full = W @ (Q_L_inv @ x_r)

        # Verify dimensions
        if len(x_full) != fes.ndof:
            raise ValueError(
                f"Dimension mismatch: reconstructed {len(x_full)} DOFs, "
                f"but FES has {fes.ndof} DOFs"
            )

        # Create GridFunction
        E_gf = GridFunction(fes, complex=True)
        E_gf.vec.FV().NumPy()[:] = x_full

        return E_gf

    def can_reconstruct(self, domain: str = None) -> bool:
        """Check if field reconstruction is possible."""
        if not self._is_reduced:
            return False

        if self.n_domains > 1:
            if self._concatenated is not None:
                return self._concatenated.can_reconstruct()
            return False

        # Single domain
        if domain is None:
            domain = self.domains[0]

        if domain not in self._W or domain not in self._Q_L_inv:
            return False

        # Check if we have snapshots from solve()
        if self._x_r_snapshots is None:
            return False

        return True

    def plot_field(
        self,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        excitation_mode: int = 0,
        domain: Optional[str] = None,
        component: Literal['real', 'imag', 'abs'] = 'abs',
        field_type: Literal['E', 'H'] = 'E',
        clipping: Optional[Dict] = None,
        euler_angles: Optional[List] = [45, -45, 0],
        **kwargs
    ) -> None:
        """
        Visualize reconstructed field at a specific frequency.

        Parameters
        ----------
        freq_idx : int
            Frequency index
        excitation_port : str, optional
            Port used for excitation. If None, uses first port.
        excitation_mode : int
            Mode index for excitation
        domain : str, optional
            Domain to visualize (for multi-domain, uses concatenated system)
        component : {'real', 'imag', 'abs'}
            Field component to plot
        field_type : {'E', 'H'}
            Electric or magnetic field
        clipping : dict, optional
            Clipping plane specification
        **kwargs
            Additional arguments passed to Draw()
        """
        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if self.frequencies is None:
            raise ValueError("No solution available. Call solve() first.")

        if freq_idx >= len(self.frequencies):
            raise ValueError(
                f"freq_idx {freq_idx} out of range [0, {len(self.frequencies) - 1}]"
            )

        # For multi-domain, delegate to concatenated system
        if self.n_domains > 1:
            if self._concatenated is None:
                raise ValueError(
                    "Multi-domain ROM: call solve() first to create concatenated system."
                )
            self._concatenated.plot_field(freq_idx=freq_idx, excitation_port=excitation_port,
                                          excitation_mode=excitation_mode, component=component, field_type=field_type,
                                          clipping=clipping, **kwargs)
            return

        # Single domain case
        if domain is None:
            domain = self.domains[0]

        freq = self.frequencies[freq_idx]
        omega = 2 * np.pi * freq

        # Default excitation port
        domain_ports = self.domain_port_map[domain]
        if excitation_port is None:
            excitation_port = domain_ports[0]

        if excitation_port not in domain_ports:
            raise ValueError(
                f"Port '{excitation_port}' not in domain '{domain}'. "
                f"Available: {domain_ports}"
            )

        pr.info(f"\nField visualization at f = {freq / 1e9:.4f} GHz")
        pr.info(f"  Domain: {domain}")
        pr.info(f"  Excitation: {excitation_port}, mode {excitation_mode}")

        # Reconstruct field
        E_gf = self._reconstruct_field_gf(freq_idx, excitation_port, excitation_mode, domain)

        # Select field type
        if field_type == 'E':
            field_cf = E_gf
            field_label = "E"
        elif field_type == 'H':
            field_cf = (1 / (1j * omega * mu0)) * curl(E_gf)
            field_label = "H"
        else:
            raise ValueError(f"Invalid field_type: {field_type}. Use 'E' or 'H'.")

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
            raise ValueError("No solution available. Call solve() first.")

        freq_idx = int(np.argmin(np.abs(self.frequencies - freq)))
        actual_freq = self.frequencies[freq_idx]

        if abs(actual_freq - freq) / max(freq, 1e-10) > 0.01:
            pr.debug(f"  Note: Using nearest frequency {actual_freq / 1e9:.4f} GHz")

        self.plot_field(freq_idx=freq_idx, **kwargs)

    # =========================================================================
    # Properties
    # =========================================================================

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

    # =========================================================================
    # Visualization
    # =========================================================================

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

        return fig, ax

    # =========================================================================
    # Info and Diagnostics
    # =========================================================================

    def print_info(self) -> None:
        """Print ROM information."""
        pr.info("\n" + "=" * 60)
        pr.info("ModelOrderReduction Information")
        pr.info("=" * 60)
        pr.info(f"Structure: {'Compound' if self.n_domains > 1 else 'Single'}")
        pr.info(f"Domains: {self.domains}")
        pr.info(f"External ports: {self._external_ports}")
        if self.n_domains > 1:
            pr.info(f"All ports: {self._all_ports}")

        print(f"\nSnapshots available: {list(self._snapshots.keys())}")

        if self._is_reduced:
            pr.info("\nPer-domain reduction:")
            for domain in self.domains:
                n = self._M[domain].shape[0]
                r = self._r[domain]
                ports = self.domain_port_map[domain]
                pr.info(f"  {domain}: {n} → {r} DOFs, ports: {ports}")

            pr.info(f"\nTotal: {self.total_dofs} → {self.total_reduced_dofs} DOFs")
            pr.info(f"Compression: {100*self.compression_ratio:.1f}%")

            if self.has_global_system:
                pr.debug(f"\nGlobal system:")
                pr.debug(f"  Global reduced DOFs: {self._r_global}")
                pr.debug(f"  A_global shape: {self._A_r_global.shape}")
                pr.debug(f"  B_global shape: {self._B_r_global.shape}")
                if self._concatenated is not None:
                    pr.debug(f"  External ports: {self._concatenated.ports}")

            pr.debug(f"\nField reconstruction: {'Available' if self.can_reconstruct() else 'Not available'}")
        else:
            pr.info("\nNot yet reduced. Call reduce() first.")

        if self.frequencies is not None:
            pr.info(f"\nSolution available:")
            pr.info(f"  Frequency range: {self.frequencies[0]/1e9:.4f} - "
                  f"{self.frequencies[-1]/1e9:.4f} GHz")
            pr.info(f"  Number of samples: {len(self.frequencies)}")

        pr.info("=" * 60)

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

    def get_reconstruction_info(self) -> Dict:
        """Get information about field reconstruction capability."""
        info = {
            'can_reconstruct': self.can_reconstruct(),
            'is_reduced': self._is_reduced,
            'has_snapshots': self._x_r_snapshots is not None,
            'n_domains': self.n_domains,
            'domains': {}
        }

        for domain in self.domains:
            info['domains'][domain] = {
                'has_W': domain in self._W,
                'has_Q_L_inv': domain in self._Q_L_inv,
                'r': self._r.get(domain),
                'n_full': self._M[domain].shape[0] if domain in self._M else None,
            }

        if self.n_domains > 1 and self._concatenated is not None:
            info['concatenated'] = self._concatenated.get_reconstruction_info()

        return info