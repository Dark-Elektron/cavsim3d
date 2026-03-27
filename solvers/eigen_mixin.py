"""
Eigenvalue/eigenvector computation and visualization mixins.

Provides shared functionality for eigenmode analysis across different solver types:
- FrequencyDomainSolver (full-order)
- ModelOrderReduction (reduced-order)
- ConcatenatedSystem (coupled reduced-order)
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Literal, Any
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
from scipy.sparse.linalg import eigsh
from core.persistence import H5Serializer
from pathlib import Path
import h5py


class EigenMixinBase:
    """
    Base mixin providing eigenvalue/eigenvector computation and visualization.

    Subclasses must implement the abstract methods to provide access to
    system matrices and mesh/FES for their specific solver type.

    This mixin assumes the generalized eigenvalue problem:
        K @ x = λ * M @ x
    where λ = ω² (squared angular frequency).
    """

    # Default threshold for filtering static modes
    DEFAULT_MIN_EIGENVALUE = 1.0  # ω² > 1 means ω > 1 rad/s

    # Cache storage (initialized by subclasses or on first use)
    _eigenvalues_cache: Dict[str, np.ndarray] = None
    _eigenvectors_cache: Dict[str, np.ndarray] = None

    # =========================================================================
    # Abstract methods - must be implemented by each solver type
    # =========================================================================

    @abstractmethod
    def _get_eigen_system_matrices(
            self,
            domain: str
    ) -> Tuple[Any, Any, Any, int]:
        """
        Get system matrices for eigenvalue computation.

        Parameters
        ----------
        domain : str
            Domain name or 'global'

        Returns
        -------
        M : sparse matrix or ndarray
            Mass matrix
        K : sparse matrix or ndarray
            Stiffness matrix
        free_dofs : array-like or None
            Indices of free DOFs (None if all DOFs are free, e.g., for reduced systems)
        n_dof : int
            Total number of DOFs
        """
        pass

    @abstractmethod
    def _get_available_eigen_domains(self) -> List[str]:
        """
        Get list of domains available for eigenvalue computation.

        Returns
        -------
        domains : list of str
            Available domain names (may include 'global')
        """
        pass

    @abstractmethod
    def _can_reconstruct_field(self, domain: str) -> bool:
        """
        Check if field reconstruction is possible for a domain.

        Parameters
        ----------
        domain : str
            Domain name

        Returns
        -------
        bool
            True if field reconstruction is supported
        """
        pass

    @abstractmethod
    def _reconstruct_eigenmode_field(
            self,
            eigenvector: np.ndarray,
            domain: str
    ) -> Any:
        """
        Reconstruct eigenmode field as a GridFunction or CoefficientFunction.

        Parameters
        ----------
        eigenvector : ndarray
            Eigenvector (in reduced or full space depending on solver)
        domain : str
            Domain name

        Returns
        -------
        field : GridFunction or CoefficientFunction
            Reconstructed field for visualization
        """
        pass

    @abstractmethod
    def _get_mesh_for_plotting(self, domain: str) -> Any:
        """
        Get mesh object for plotting.

        Parameters
        ----------
        domain : str
            Domain name

        Returns
        -------
        mesh : Mesh
            NGSolve mesh object
        """
        pass

    # =========================================================================
    # Shared implementation
    # =========================================================================

    def _init_eigen_cache(self) -> None:
        """Initialize eigenvalue/eigenvector cache if needed."""
        if self._eigenvalues_cache is None:
            self._eigenvalues_cache = {}
        if self._eigenvectors_cache is None:
            self._eigenvectors_cache = {}

    def _clear_eigen_cache(self, domain: str = None) -> None:
        """Clear eigenvalue cache for a domain or all domains."""
        self._init_eigen_cache()
        if domain is None:
            self._eigenvalues_cache = {}
            self._eigenvectors_cache = {}
        else:
            self._eigenvalues_cache.pop(domain, None)
            self._eigenvectors_cache.pop(domain, None)

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
            min_eigenvalue = EigenMixinBase.DEFAULT_MIN_EIGENVALUE

        # Sort eigenvalues
        eigs_sorted = np.sort(np.real(eigenvalues))

        # Filter static modes
        if filter_static:
            eigs_sorted = eigs_sorted[eigs_sorted > min_eigenvalue]

        # Limit to n_modes
        if n_modes is not None and len(eigs_sorted) > n_modes:
            eigs_sorted = eigs_sorted[:n_modes]

        return eigs_sorted

    @staticmethod
    def _filter_eigenpairs(
            eigenvalues: np.ndarray,
            eigenvectors: np.ndarray,
            filter_static: bool = True,
            min_eigenvalue: float = None,
            n_modes: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter and sort eigenvalue/eigenvector pairs.

        Parameters
        ----------
        eigenvalues : ndarray
            Raw eigenvalues
        eigenvectors : ndarray
            Raw eigenvectors as columns (n_dof x n_eigs)
        filter_static : bool
            If True, remove static modes
        min_eigenvalue : float, optional
            Threshold for static mode filtering
        n_modes : int, optional
            Return only first n_modes

        Returns
        -------
        eigenvalues : ndarray
            Filtered, sorted eigenvalues
        eigenvectors : ndarray
            Corresponding eigenvectors
        """
        if min_eigenvalue is None:
            min_eigenvalue = EigenMixinBase.DEFAULT_MIN_EIGENVALUE

        if len(eigenvalues) == 0:
            return eigenvalues, eigenvectors

        # Sort by eigenvalue
        sort_idx = np.argsort(np.real(eigenvalues))
        eigs = eigenvalues[sort_idx]
        vecs = eigenvectors[:, sort_idx]

        # Filter static modes
        if filter_static:
            mask = np.real(eigs) > min_eigenvalue
            eigs = eigs[mask]
            vecs = vecs[:, mask]

        # Limit to n_modes
        if n_modes is not None and len(eigs) > n_modes:
            eigs = eigs[:n_modes]
            vecs = vecs[:, :n_modes]

        return eigs, vecs

    def _compute_eigenpairs_dense(
            self,
            M: np.ndarray,
            K: np.ndarray,
            n_modes: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenpairs using dense solver.

        For reduced systems where matrices are already small and dense.
        """
        try:
            eigenvalues, eigenvectors = sl.eigh(K, M)
        except Exception as e:
            print(f"Warning: Dense eigh failed: {e}")
            print("Trying with regularization...")

            # Add small regularization to M
            eps = 1e-10 * np.max(np.abs(np.diag(M)))
            M_reg = M + eps * np.eye(M.shape[0])
            eigenvalues, eigenvectors = sl.eigh(K, M_reg)

        # Limit to n_modes if requested
        if n_modes is not None and len(eigenvalues) > n_modes:
            eigenvalues = eigenvalues[:n_modes]
            eigenvectors = eigenvectors[:, :n_modes]

        return eigenvalues, eigenvectors

    def _compute_eigenpairs_sparse(
            self,
            M: sp.spmatrix,
            K: sp.spmatrix,
            free_dofs: np.ndarray,
            n_dof_full: int,
            n_modes: int = 50,
            sigma: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenpairs using sparse solver with shift-invert.

        For full-order systems with sparse matrices.
        """
        if len(free_dofs) == 0:
            return np.array([]), np.zeros((n_dof_full, 0))

        # Extract submatrices for free DOFs
        if sp.issparse(M):
            M_free = M[free_dofs, :][:, free_dofs]
            K_free = K[free_dofs, :][:, free_dofs]
        else:
            M_free = M[np.ix_(free_dofs, free_dofs)]
            K_free = K[np.ix_(free_dofs, free_dofs)]

        n_free = len(free_dofs)
        k = min(n_modes, n_free - 2)
        k = max(k, 1)

        # Default shift for shift-invert
        if sigma is None:
            sigma = 1e18  # Target around 1 GHz

        eigenvalues = None
        eigenvectors_free = None

        try:
            # Try sparse solver with shift-invert
            M_csr = sp.csr_matrix(M_free) if sp.issparse(M_free) else sp.csr_matrix(M_free)
            K_csr = sp.csr_matrix(K_free) if sp.issparse(K_free) else sp.csr_matrix(K_free)

            eigenvalues, eigenvectors_free = eigsh(
                K_csr, k=k, M=M_csr,
                sigma=sigma, which='LM',
                return_eigenvectors=True
            )

        except Exception as e1:
            print(f"Note: Sparse eigsh failed: {e1}")
            print("Falling back to dense solver...")

            try:
                M_dense = M_free.toarray() if sp.issparse(M_free) else np.array(M_free)
                K_dense = K_free.toarray() if sp.issparse(K_free) else np.array(K_free)

                eigenvalues, eigenvectors_free = sl.eigh(K_dense, M_dense)

                if n_modes and len(eigenvalues) > n_modes:
                    eigenvalues = eigenvalues[:n_modes]
                    eigenvectors_free = eigenvectors_free[:, :n_modes]

            except Exception as e2:
                print(f"Error: Dense solver also failed: {e2}")
                return np.array([]), np.zeros((n_dof_full, 0))

        # Expand eigenvectors to full DOF space
        n_vecs = eigenvectors_free.shape[1]
        eigenvectors_full = np.zeros((n_dof_full, n_vecs), dtype=eigenvectors_free.dtype)
        eigenvectors_full[free_dofs, :] = eigenvectors_free

        return eigenvalues, eigenvectors_full

    def get_eigenvectors(
            self,
            domain: str = None,
            filter_static: bool = True,
            min_eigenvalue: float = None,
            n_modes: int = None,
            sigma: float = None,
            return_eigenvalues: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute eigenvectors from generalized eigenvalue problem.

        Parameters
        ----------
        domain : str, optional
            Specific domain or 'global'. If None, returns all available.
        filter_static : bool
            If True (default), remove static modes
        min_eigenvalue : float, optional
            Threshold for static mode filtering. Default: 1.0
        n_modes : int, optional
            Number of eigenmodes to compute/return
        sigma : float, optional
            Shift for shift-invert mode (sparse solver only)
        return_eigenvalues : bool
            If True, return (eigenvalues, eigenvectors) tuple

        Returns
        -------
        eigenvectors : ndarray or dict
            Eigenvectors as columns, shape (n_dof, n_modes)
        eigenvalues : ndarray (optional)
            Corresponding eigenvalues if return_eigenvalues=True
        """
        available_domains = self._get_available_eigen_domains()

        if not available_domains:
            raise ValueError("No domains available for eigenvalue computation")

        def compute_for_domain(d: str) -> Tuple[np.ndarray, np.ndarray]:
            M, K, free_dofs, n_dof = self._get_eigen_system_matrices(d)

            if free_dofs is None:
                # Reduced system - use dense solver
                M_arr = np.asarray(M)
                K_arr = np.asarray(K)
                raw_eigs, raw_vecs = self._compute_eigenpairs_dense(
                    M_arr, K_arr, n_modes=n_modes or 50
                )
            else:
                # Full system - use sparse solver
                raw_eigs, raw_vecs = self._compute_eigenpairs_sparse(
                    M, K, free_dofs, n_dof,
                    n_modes=n_modes or 50,
                    sigma=sigma
                )

            return self._filter_eigenpairs(
                raw_eigs, raw_vecs,
                filter_static=filter_static,
                min_eigenvalue=min_eigenvalue,
                n_modes=n_modes
            )

        # Handle specific domain request
        if domain is not None:
            if domain not in available_domains:
                raise KeyError(
                    f"Domain '{domain}' not found. Available: {available_domains}"
                )
            eigs, vecs = compute_for_domain(domain)

            # Populate cache for save_eigenmodes
            self._init_eigen_cache()
            self._eigenvalues_cache[domain] = eigs
            self._eigenvectors_cache[domain] = vecs

            if return_eigenvalues:
                return eigs, vecs
            return vecs

        # Return all available
        results_eigs = {}
        results_vecs = {}

        for d in available_domains:
            try:
                eigs, vecs = compute_for_domain(d)
                results_eigs[d] = eigs
                results_vecs[d] = vecs
                # Populate cache for save_eigenmodes
                self._init_eigen_cache()
                self._eigenvalues_cache[d] = eigs
                self._eigenvectors_cache[d] = vecs
            except Exception as e:
                print(f"Warning: Could not compute eigenvectors for {d}: {e}")

        if return_eigenvalues:
            return results_eigs, results_vecs
        return results_vecs

    def get_eigenmode(
            self,
            mode_index: int,
            domain: str = None,
            filter_static: bool = True,
            min_eigenvalue: float = None
    ) -> Tuple[float, Any]:
        """
        Get a specific eigenmode as a reconstructed field.

        Parameters
        ----------
        mode_index : int
            Index of the mode (0-based, after filtering)
        domain : str, optional
            Domain to get eigenmode from. Default: first available or 'global'.
        filter_static : bool
            Whether to filter static modes
        min_eigenvalue : float, optional
            Threshold for static mode filtering

        Returns
        -------
        frequency : float
            Resonant frequency in Hz
        mode_field : GridFunction or CoefficientFunction
            Eigenmode field for visualization
        """
        # Default domain selection
        available_domains = self._get_available_eigen_domains()
        if domain is None:
            if 'global' in available_domains:
                domain = 'global'
            else:
                domain = available_domains[0]

        if not self._can_reconstruct_field(domain):
            raise ValueError(
                f"Field reconstruction not supported for domain '{domain}'. "
                f"Ensure mesh and FES are available."
            )

        # Get eigenvectors
        eigs, vecs = self.get_eigenvectors(
            domain=domain,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=mode_index + 10,  # Get enough modes
            return_eigenvalues=True
        )

        if mode_index >= vecs.shape[1]:
            raise IndexError(
                f"Mode index {mode_index} out of range. "
                f"Only {vecs.shape[1]} modes available."
            )

        eigenvalue = eigs[mode_index]
        frequency = np.sqrt(np.maximum(np.real(eigenvalue), 0)) / (2 * np.pi)

        # Reconstruct field
        eigenvector = vecs[:, mode_index]
        mode_field = self._reconstruct_eigenmode_field(eigenvector, domain)

        return frequency, mode_field

    def plot_eigenmode(
            self,
            mode_index: int = 0,
            domain: str = None,
            component: Literal['real', 'imag', 'abs', 'all'] = 'abs',
            field_type: Literal['E', 'H'] = 'E',
            filter_static: bool = True,
            min_eigenvalue: float = None,
            clipping: Optional[Dict] = None,
            show_info: bool = True,
            **kwargs
    ) -> None:
        """
        Visualize an eigenmode field pattern.

        Parameters
        ----------
        mode_index : int
            Index of the mode to visualize (0-based, after filtering)
        domain : str, optional
            Domain to visualize. Default: 'global' if available.
        component : {'real', 'imag', 'abs', 'all'}
            Field component to plot
        field_type : {'E', 'H'}
            Electric or magnetic field
        filter_static : bool
            Whether to filter out static modes
        min_eigenvalue : float, optional
            Threshold for static mode filtering
        clipping : dict, optional
            Clipping plane specification for 3D visualization
        show_info : bool
            Print mode information
        **kwargs
            Additional arguments passed to Draw()
        """
        # Import NGSolve visualization
        try:
            from ngsolve import Norm, curl
            from ngsolve.webgui import Draw
        except ImportError:
            raise ImportError(
                "NGSolve is required for eigenmode plotting. "
                "Install with: pip install ngsolve"
            )

        # Get eigenmode
        frequency, mode_field = self.get_eigenmode(
            mode_index=mode_index,
            domain=domain,
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue
        )

        # Determine domain for display
        available_domains = self._get_available_eigen_domains()
        if domain is None:
            domain = 'global' if 'global' in available_domains else available_domains[0]

        mesh = self._get_mesh_for_plotting(domain)

        if show_info:
            print(f"\n{'=' * 60}")
            print(f"Eigenmode Visualization")
            print(f"{'=' * 60}")
            print(f"Domain: {domain}")
            print(f"Mode index: {mode_index}")
            print(f"Resonant frequency: {frequency / 1e9:.6f} GHz")
            print(f"Angular frequency ω: {2 * np.pi * frequency:.4e} rad/s")
            print(f"Field type: {field_type}")
            print(f"Component: {component}")
            print(f"{'=' * 60}")

        # Select field type
        if field_type == 'E':
            field_cf = mode_field
            field_label = "E"
        elif field_type == 'H':
            field_cf = curl(mode_field)
            field_label = "H (∝ curl E)"
        else:
            raise ValueError(f"Invalid field_type: {field_type}")

        # Prepare draw kwargs
        draw_kwargs = kwargs.copy()
        if clipping:
            draw_kwargs['clipping'] = clipping

        # Try to use BoundaryFromVolumeCF for better visualization
        try:
            from ngsolve import BoundaryFromVolumeCF
            use_boundary = True
        except ImportError:
            use_boundary = False

        def draw_field(cf, label):
            if use_boundary:
                from ngsolve import BoundaryFromVolumeCF
                Draw(BoundaryFromVolumeCF(cf), mesh, **draw_kwargs)
            else:
                Draw(cf, mesh, **draw_kwargs)

        # Plot based on component selection
        if component == 'all':
            print(f"\nPlotting Real({field_label}):")
            draw_field(field_cf.real, f"Re({field_label})")

            print(f"\nPlotting Imag({field_label}):")
            draw_field(field_cf.imag, f"Im({field_label})")

            print(f"\nPlotting |{field_label}|:")
            draw_field(Norm(field_cf), f"|{field_label}|")

        else:
            if component == 'abs':
                cf_plot = Norm(field_cf)
                comp_label = f"|{field_label}|"
            elif component == 'real':
                cf_plot = field_cf.real
                comp_label = f"Re({field_label})"
            elif component == 'imag':
                cf_plot = field_cf.imag
                comp_label = f"Im({field_label})"
            else:
                cf_plot = field_cf
                comp_label = field_label

            print(f"\nPlotting {comp_label}:")
            draw_field(cf_plot, comp_label)

    def plot_eigenmodes(
            self,
            mode_indices: List[int] = None,
            n_modes: int = 4,
            domain: str = None,
            component: Literal['real', 'imag', 'abs'] = 'abs',
            field_type: Literal['E', 'H'] = 'E',
            filter_static: bool = True,
            min_eigenvalue: float = None,
            **kwargs
    ) -> None:
        """
        Plot multiple eigenmodes.

        Parameters
        ----------
        mode_indices : list of int, optional
            Specific mode indices to plot. If None, plots first n_modes.
        n_modes : int
            Number of modes to plot if mode_indices is None
        domain : str, optional
            Domain to visualize
        component : {'real', 'imag', 'abs'}
            Field component to plot
        field_type : {'E', 'H'}
            Electric or magnetic field
        filter_static : bool
            Whether to filter static modes
        min_eigenvalue : float, optional
            Threshold for static mode filtering
        **kwargs
            Additional arguments passed to Draw()
        """
        if mode_indices is None:
            mode_indices = list(range(n_modes))

        print(f"\n{'=' * 60}")
        print(f"Plotting {len(mode_indices)} Eigenmodes")
        print(f"{'=' * 60}")

        for idx in mode_indices:
            try:
                self.plot_eigenmode(
                    mode_index=idx,
                    domain=domain,
                    component=component,
                    field_type=field_type,
                    filter_static=filter_static,
                    min_eigenvalue=min_eigenvalue,
                    show_info=True,
                    **kwargs
                )
            except Exception as e:
                print(f"Warning: Could not plot mode {idx}: {e}")

    def print_eigenfrequencies(
            self,
            n_modes: int = 20,
            domain: str = None,
            filter_static: bool = True,
            min_eigenvalue: float = None
    ) -> None:
        """
        Print table of eigenfrequencies.

        Parameters
        ----------
        n_modes : int
            Number of modes to display
        domain : str, optional
            Domain to show. If None, shows all available.
        filter_static : bool
            Whether to filter static modes
        min_eigenvalue : float, optional
            Threshold for static mode filtering
        """
        available_domains = self._get_available_eigen_domains()

        print("\n" + "=" * 70)
        print("Eigenfrequencies")
        print("=" * 70)

        if domain is not None:
            domains_to_show = [domain]
        else:
            domains_to_show = available_domains

        for d in domains_to_show:
            try:
                eigs, _ = self.get_eigenvectors(
                    domain=d,
                    filter_static=filter_static,
                    min_eigenvalue=min_eigenvalue,
                    n_modes=n_modes,
                    return_eigenvalues=True
                )

                freqs = np.sqrt(np.maximum(eigs, 0)) / (2 * np.pi)

                print(f"\nDomain: {d}")
                print(f"{'Index':<8} {'Frequency (GHz)':<18} {'ω² (rad²/s²)':<20}")
                print("-" * 50)

                for i, (f, e) in enumerate(zip(freqs, eigs)):
                    print(f"{i:<8} {f / 1e9:<18.6f} {e:<20.4e}")

            except Exception as e:
                print(f"\nDomain: {d} - Error: {e}")

        print("=" * 70)

    @property
    def eigenvalues(self) -> Dict[str, np.ndarray]:
        """
        Cached eigenvalues for all available domains.

        Returns dictionary with keys for each domain.
        Eigenvalues are ω² (angular frequency squared).
        """
        self._init_eigen_cache()

        available_domains = self._get_available_eigen_domains()

        for d in available_domains:
            if d not in self._eigenvalues_cache:
                try:
                    eigs, _ = self.get_eigenmodes(
                        domain=d,
                        filter_static=True,
                        n_modes=50,
                        return_eigenvalues=True,
                        _auto_save=True
                    )
                    self._eigenvalues_cache[d] = eigs
                except Exception as e:
                    print(f"Warning: Could not compute eigenvalues for {d}: {e}")

        return self._eigenvalues_cache.copy()

    @property
    def eigenvectors(self) -> Dict[str, np.ndarray]:
        """
        Cached eigenvectors for all available domains.

        Returns dictionary with keys for each domain.
        Each value is array of shape (n_dof, n_modes).
        """
        self._init_eigen_cache()

        available_domains = self._get_available_eigen_domains()

        for d in available_domains:
            if d not in self._eigenvectors_cache:
                try:
                    _, vecs = self.get_eigenmodes(
                        domain=d,
                        filter_static=True,
                        n_modes=50,
                        return_eigenvalues=True,
                        _auto_save=True
                    )
                    self._eigenvectors_cache[d] = vecs
                except Exception as e:
                    print(f"Warning: Could not compute eigenvectors for {d}: {e}")

        return self._eigenvectors_cache.copy()

    def get_eigenmodes(self, _auto_save: bool = True, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Standardized API for retrieving eigenvalues and eigenvectors.
        
        Returns (eigenvalues, eigenvectors).
        """
        kwargs.setdefault('return_eigenvalues', True)
        res = self.get_eigenvectors(**kwargs)
        if _auto_save:
            try:
                # Cache is already populated by get_eigenvectors, just flush to disk
                self.save_eigenmodes(auto_compute=False)
            except Exception as e:
                pr.warning(f"Failed to auto-save eigenmodes: {e}")
        return res

    def get_eigenvalues(self, **kwargs) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Standardized API for retrieving only eigenvalues.
        
        Returns eigenvalues for specified domain or dict of all available domains.
        """
        kwargs['return_eigenvalues'] = True
        kwargs['return_eigenvectors'] = False
        return self.get_eigenvectors(**kwargs)

    def save_eigenmodes(self, path: Union[str, Path, None] = None, domain: str = None, auto_compute: bool = False, **kwargs):
        """
        Save computed eigenmodes to disk in HDF5 format.
        
        Parameters
        ----------
        path : Path, optional
            Directory to save 'eigenmodes.h5' into. If None, uses project path if available.
        domain : str, optional
            Specific domain to save. If None, saves all computed domains.
        auto_compute : bool, optional
            If False (default), only saves if eigenmodes are already cached.
        **kwargs
            Arguments passed to get_eigenmodes (e.g., n_modes)
        """
        self._init_eigen_cache()
        if not auto_compute:
            # Check if any eigenmodes exist for the given domain(s)
            if domain is not None:
                if domain not in self._eigenvalues_cache:
                    return
            else:
                if not self._eigenvalues_cache:
                    return
        
        # Determine path
        if path is None:
            if hasattr(self, '_project_path') and self._project_path:
                base = Path(self._project_path)
                if hasattr(self, 'is_compound') and self.is_compound:
                    path = base / "fds" / "foms" / "eigenmodes"
                else:
                    path = base / "fds" / "fom" / "eigenmodes"
            else:
                raise ValueError("No path provided and no project_path available.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        h5_file = path / "eigenmodes.h5"
        print(h5_file)
        
        if auto_compute:
            kwargs.pop('_auto_save', None)
            res = self.get_eigenmodes(domain=domain, **kwargs)
            if isinstance(res[0], dict):
                eigs_dict, vecs_dict = res
            else:
                d_name = domain or (self._get_available_eigen_domains()[0] if self._get_available_eigen_domains() else 'global')
                eigs_dict = {d_name: res[0]}
                vecs_dict = {d_name: res[1]}
        else:
            eigs_dict = {d: self._eigenvalues_cache[d] for d in self._eigenvalues_cache if (domain is None or d == domain)}
            vecs_dict = {d: self._eigenvectors_cache[d] for d in self._eigenvectors_cache if (domain is None or d == domain)}

        with h5py.File(h5_file, "a") as f:
            for d in vecs_dict:
                if d in eigs_dict:
                    grp = f.require_group(d)
                    H5Serializer.save_dataset(grp, "eigenvalues", eigs_dict[d])
                    H5Serializer.save_dataset(grp, "eigenvectors", vecs_dict[d])
        
        # pr.debug(f"Eigenmodes saved to {h5_file}")

    def load_eigenmodes(self, path: Union[str, Path, None] = None):
        """
        Load eigenmodes from disk and populate internal cache.
        
        Parameters
        ----------
        path : Path, optional
            Directory containing 'eigenmodes.h5'. If None, uses project path.
        """
        import h5py
        
        if path is None:
            if hasattr(self, '_project_path') and self._project_path:
                base = Path(self._project_path)
                # Try standard locations
                potential_paths = [
                    base / "fds" / "fom" / "eigenmodes",
                    base / "fds" / "foms" / "eigenmodes",
                    base / "fds" / "eigenmodes",
                    base / "eigenmodes"
                ]
                for p in potential_paths:
                    if (p / "eigenmodes.h5").exists():
                        path = p
                        break
                if path is None: return # Not found, silent return
            else:
                return # No path, silent return

        path = Path(path)
        h5_file = path / "eigenmodes.h5"
        if not h5_file.exists():
            return

        self._init_eigen_cache()
        
        try:
            with h5py.File(h5_file, "r") as f:
                for domain in f.keys():
                    group = f[domain]
                    if "eigenvalues" in group and "eigenvectors" in group:
                        eigs = H5Serializer.load_dataset(group["eigenvalues"])
                        vecs = H5Serializer.load_dataset(group["eigenvectors"])
                        self._eigenvalues_cache[domain] = eigs
                        self._eigenvectors_cache[domain] = vecs
            print(f"Eigenmodes loaded from {h5_file}")
        except Exception as e:
            print(f"Warning: Could not load eigenmodes from {h5_file}: {e}")


# =============================================================================
# Solver-specific mixins
# =============================================================================

class FDSEigenMixin(EigenMixinBase):
    """
    Eigenvalue mixin for FrequencyDomainSolver.

    Uses full-order sparse matrices (M, K) and NGSolve FES for field reconstruction.
    """

    def _get_eigen_system_matrices(
            self,
            domain: str
    ) -> Tuple[Any, Any, Any, int]:
        """Get system matrices from FDS."""
        if domain == 'global':
            if self.M_global is None:
                raise ValueError("Global matrices not assembled")
            M = self.M_global
            K = self.K_global
            fes = self._fes_global
        else:
            if domain not in self.M:
                raise KeyError(f"Domain '{domain}' not found")
            M = self.M[domain]
            K = self.K[domain]
            fes = self._fes[domain]

        # Get free DOFs from FES
        freedofs = fes.FreeDofs()
        free_idx = np.array([i for i in range(fes.ndof) if freedofs[i]])

        return M, K, free_idx, fes.ndof

    def _get_available_eigen_domains(self) -> List[str]:
        """Get domains available in FDS."""
        domains = [d for d in self.domains if d in self.M]
        if self.M_global is not None:
            domains.append('global')
        return domains

    def _can_reconstruct_field(self, domain: str) -> bool:
        """FDS can always reconstruct fields."""
        if domain == 'global':
            return self._fes_global is not None
        return domain in self._fes

    def _reconstruct_eigenmode_field(
            self,
            eigenvector: np.ndarray,
            domain: str
    ) -> Any:
        """Reconstruct field as GridFunction."""
        from ngsolve import GridFunction

        if domain == 'global':
            fes = self._fes_global
        else:
            fes = self._fes[domain]

        gf = GridFunction(fes)
        gf.vec.FV().NumPy()[:] = np.real(eigenvector)

        return gf

    def _get_mesh_for_plotting(self, domain: str) -> Any:
        """Get mesh from FDS."""
        return self.mesh


class ROMEigenMixin(EigenMixinBase):
    """
    Eigenvalue mixin for ModelOrderReduction.

    Uses reduced matrices (A_r) directly. Field reconstruction requires
    projection basis (W, Q_L_inv) and access to original FES via solver.
    """
    def calculate_resonant_modes(self, **kwargs) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute eigenvalues and eigenvectors of the reduced system matrix (A_r).
        Returns either a tuple (eigenvalues, eigenvectors) for a single domain/global
        or a dictionary mapping domain to tuples.
        """
        return self.get_eigenmodes(_auto_save=True, **kwargs)

    def _get_eigen_system_matrices(
            self,
            domain: str
    ) -> Tuple[Any, Any, Any, int]:
        """Get reduced system matrices from ROM."""
        if domain == 'global':
            if self._A_r_global is None:
                raise ValueError("Global reduced matrices not available")
            # For reduced system: (A_r - ω²I)x = ωB_r u
            # Eigenvalue problem: A_r x = ω² x (with M = I)
            A_r = self._A_r_global
            r = A_r.shape[0]
            return np.eye(r), A_r, None, r  # M=I, K=A_r, no free_dofs
        else:
            if domain not in self._A_r:
                raise KeyError(f"Domain '{domain}' not found")
            A_r = self._A_r[domain]
            r = A_r.shape[0]
            return np.eye(r), A_r, None, r

    def _get_available_eigen_domains(self) -> List[str]:
        """Get domains available in ROM."""
        domains = list(self._A_r.keys())
        if self._A_r_global is not None and 'global' not in domains:
            domains.append('global')
        return domains

    def _can_reconstruct_field(self, domain: str) -> bool:
        """Check if field reconstruction is possible."""
        # Need projection basis and access to solver's FES
        if domain == 'global':
            has_basis = self._W_r_global is not None
        else:
            has_basis = domain in self._W

        has_fes = hasattr(self, 'solver') and self.solver is not None

        return has_basis and has_fes

    def _reconstruct_eigenmode_field(
            self,
            eigenvector: np.ndarray,
            domain: str
    ) -> Any:
        """Reconstruct field from reduced eigenvector."""
        from ngsolve import GridFunction

        if domain == 'global':
            # For global, need to handle multi-domain case
            if self.n_domains == 1:
                domain = self.domains[0]
            else:
                # Global eigenvector is in coupled reduced space
                # Project back through W_coupled
                if self._W_r_global is None:
                    raise ValueError("Global projection basis not available")

                # This gives us the uncoupled reduced coordinates
                # For now, just use the first domain's visualization
                # TODO: Proper multi-domain field reconstruction
                print("Warning: Global eigenmode visualization for multi-domain "
                      "shows first domain only")
                domain = self.domains[0]

                # Extract portion of eigenvector for this domain
                r_domain = self._r[domain]
                eigenvector = eigenvector[:r_domain]

        # Get projection matrices
        W = self._W[domain]
        Q_L_inv = self._Q_L_inv[domain]

        # Reconstruct: x_full = W @ Q_L_inv @ x_r
        x_full = W @ Q_L_inv @ np.real(eigenvector)

        # Get FES from solver
        if domain in self.solver._fes:
            fes = self.solver._fes[domain]
        else:
            fes = self.solver._fes_global

        gf = GridFunction(fes)
        gf.vec.FV().NumPy()[:] = x_full

        return gf

    def _get_mesh_for_plotting(self, domain: str) -> Any:
        """Get mesh from underlying solver."""
        return self.mesh


class ConcatEigenMixin(EigenMixinBase):
    """
    Eigenvalue mixin for ConcatenatedSystem.

    Uses coupled reduced matrices (A_coupled). Field reconstruction requires
    W_coupled and access to original structures' projection bases.
    """

    def _get_eigen_system_matrices(
            self,
            domain: str
    ) -> Tuple[Any, Any, Any, int]:
        """Get system matrices from ConcatenatedSystem."""
        if domain is None or domain == 'global':
            if self.A_coupled is None:
                raise ValueError("System not coupled yet")
            A = self.A_coupled
            r = A.shape[0]
            return np.eye(r), A, None, r
        else:
            # Per-structure matrices
            for struct in self.structures:
                if struct.domain == domain:
                    A = struct.Ard
                    r = A.shape[0]
                    return np.eye(r), A, None, r
            raise KeyError(f"Domain '{domain}' not found")

    def _get_available_eigen_domains(self) -> List[str]:
        """Get available domains."""
        domains = [s.domain for s in self.structures]
        if self.A_coupled is not None:
            domains.append('global')
        return domains

    def _can_reconstruct_field(self, domain: str) -> bool:
        """Check if field reconstruction is possible."""
        # Need solver reference for mesh and FES
        return hasattr(self, '_solver_ref') and self._solver_ref is not None

    def _reconstruct_eigenmode_field(
            self,
            eigenvector: np.ndarray,
            domain: str
    ) -> Any:
        """Reconstruct field from coupled eigenvector."""
        if not self._can_reconstruct_field(domain):
            raise ValueError(
                "Field reconstruction requires solver reference. "
                "Pass solver to ConcatenatedSystem constructor or use "
                "set_solver_reference()."
            )

        from ngsolve import GridFunction
        
        # Uncouple the global eigenvector
        x_uncoupled = self.W_coupled @ eigenvector

        if domain == 'global' or domain is None:
            # Reconstruct the continuous global field using concatenation scaling logic!
            if self.n_structures > 1 and self.connections:
                if hasattr(self, '_compute_vector_scaling_factors'):
                    scales = self._compute_vector_scaling_factors(x_uncoupled)
                else:
                    import warnings
                    warnings.warn(
                        "Interface scaling not available (_compute_vector_scaling_factors missing). "
                        "Eigenmode fields may show discontinuities at interfaces.",
                        UserWarning, stacklevel=3
                    )
                    scales = np.ones(self.n_structures, dtype=complex)
            else:
                scales = np.ones(self.n_structures, dtype=complex)
            
            # --- Inline field reconstruction (was _reconstruct_field_from_vector) ---
            # Create unified GridFunction on global FES
            E_gf = GridFunction(self.fes, complex=True)

            for struct_idx, struct in enumerate(self.structures):
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

        # Find the structure for this domain
        struct = None
        struct_idx = None
        for i, s in enumerate(self.structures):
            if s.domain == domain:
                struct = s
                struct_idx = i
                break

        if struct is None:
            raise KeyError(f"Domain '{domain}' not found")

        # Extract this structure's reduced solution
        start_r = self._structure_dof_offsets[struct_idx]
        x_reduced = x_uncoupled[start_r:start_r + struct.r]

        # Reconstruct native scaled field
        x_full = struct.reconstruct(x_reduced)

        gf = GridFunction(struct.fes, complex=True)
        gf.vec.FV().NumPy()[:] = x_full

        return gf

    def _get_mesh_for_plotting(self, domain: str) -> Any:
        """Get mesh from solver reference."""
        if not hasattr(self, '_solver_ref') or self._solver_ref is None:
            raise ValueError("Solver reference required for plotting")

        solver = self._solver_ref
        if hasattr(solver, 'mesh'):
            return solver.mesh
        elif hasattr(solver, 'solver'):
            return solver.solver.mesh
        else:
            raise ValueError("Cannot find mesh in solver reference")

    def set_solver_reference(self, solver: Any) -> 'ConcatenatedSystem':
        """
        Set reference to parent solver for field reconstruction.

        Parameters
        ----------
        solver : FrequencyDomainSolver or ModelOrderReduction
            Parent solver with mesh and FES information

        Returns
        -------
        self
            For method chaining
        """
        self._solver_ref = solver
        return self