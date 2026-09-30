"""
Eigenvalue/eigenvector computation and visualization mixins.

Provides shared functionality for eigenmode analysis across different solver types:
- FrequencyDomainSolver (full-order)
- ModelOrderReduction (reduced-order)
- ConcatenatedSystem (coupled reduced-order)
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Literal, Any
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sl
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, eigs as nearest_eigs, eigsh
from cavsim3d.core.persistence import H5Serializer
from cavsim3d.core.constants import MIN_EIGENVALUE, SIGMA_COPPER
from cavsim3d.solvers.figures_of_merit import (ModePiece, beam_line, field_on_line,
                                               figures_of_merit, voltage)
import cavsim3d.utils.printing as pr
from pathlib import Path
from cavsim3d.geometry.base import _display_webgui_fallback

if TYPE_CHECKING:
    from cavsim3d.solvers.concatenation import ConcatenatedSystem


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
    DEFAULT_MIN_EIGENVALUE = MIN_EIGENVALUE  # omega^2 of 1 MHz: below is static

    # A reduced model is accurate near the band its snapshots cover; far from
    # it the projection leaves spurious modes (e.g. 0.4 and 0.7 GHz for a
    # guide reduced over 1.5-3 GHz).  Its default spectrum -- the listing and
    # the mode indices of get_eigenmode/get_rq/get_figures_of_merit -- keeps
    # the modes within this fraction of the band's edges.
    TRAINING_BAND_MARGIN = 0.1

    # Cache storage (initialized by subclasses or on first use)
    _eigenvalues_cache: Dict[str, np.ndarray] = None
    _eigenvectors_cache: Dict[str, np.ndarray] = None

    def _eigen_training_band(self) -> Optional[Tuple[float, float]]:
        """Training band ``(fmin, fmax)`` [GHz] of a reduced model, else None.

        None (a full-order model) keeps every mode in the default spectrum.
        """
        return None

    def _training_window(self) -> Optional[Tuple[float, float]]:
        """Default spectrum window ``(lam_lo, lam_hi)`` in omega^2, or None."""
        band = self._eigen_training_band()
        if not band:
            return None
        m = self.TRAINING_BAND_MARGIN
        lo, hi = band
        return ((2 * np.pi * (1 - m) * lo * 1e9) ** 2,
                (2 * np.pi * (1 + m) * hi * 1e9) ** 2)

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

        # No truncation here: the ascending spectrum starts with the static
        # (near-zero) modes, so cutting to n_modes BEFORE the static filter
        # would return fewer physical modes than asked.  _filter_eigenpairs
        # applies n_modes after filtering.
        return eigenvalues, eigenvectors

    def _default_eigen_shift(self) -> float:
        """Shift-invert target omega^2 when the caller gives none.

        The centre (in omega^2) of the solved band: every in-band mode then
        lies closer to it than the static null space does.  Before any solve,
        aim at f ~ c0/L for the model's largest extent L -- the lowest few
        modes of a closed structure of that size.
        """
        from cavsim3d.core.constants import c0
        f = getattr(self, 'frequencies', None)
        if f is not None and len(f):
            w = 2 * np.pi * np.asarray(f, dtype=float)
            return float(0.5 * (w.min() ** 2 + w.max() ** 2))
        mesh = getattr(self, 'mesh', None)
        if mesh is not None:
            try:
                pts = np.array([v.point for v in mesh.vertices])
                L = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
                if L > 0:
                    return float((2 * np.pi * c0 / L) ** 2)
            except Exception:
                pass
        return float((2 * np.pi * 1e9) ** 2)  # 1 GHz

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

        # Shift-invert returns the modes NEAREST sigma.  The curl-curl operator
        # has a huge static (gradient) null space at omega^2 = 0, which is at
        # distance sigma -- so a physical mode is only found if it lies below
        # 2*sigma.  A fixed shift (the old 1e18 ~ 159 MHz) therefore returned
        # nothing but static modes for GHz structures.
        if sigma is None:
            sigma = self._default_eigen_shift()

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

                # Same selection as shift-invert: the k modes nearest sigma
                keep = np.sort(np.argsort(np.abs(eigenvalues - sigma))[:k])
                eigenvalues = eigenvalues[keep]
                eigenvectors_free = eigenvectors_free[:, keep]

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
            Threshold (omega^2) for static mode filtering. Default: omega^2 of 1 MHz
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

            eigs, vecs = self._filter_eigenpairs(
                raw_eigs, raw_vecs,
                filter_static=filter_static,
                min_eigenvalue=min_eigenvalue,
                n_modes=None
            )
            # a reduced model: only the modes near its training band
            window = (self._training_window()
                      if filter_static and min_eigenvalue is None else None)
            if window is not None:
                keep = (np.real(eigs) >= window[0]) & (np.real(eigs) <= window[1])
                eigs, vecs = eigs[keep], vecs[:, keep]
            if n_modes is not None:
                eigs, vecs = eigs[:n_modes], vecs[:, :n_modes]
            return eigs, vecs

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
        domain = domain or self._default_eigen_domain()
        if not self._can_reconstruct_field(domain):
            raise ValueError(
                f"Field reconstruction not supported for domain '{domain}'. "
                f"Ensure mesh and FES are available."
            )
        frequency, eigenvector = self._eigenpair(mode_index, domain, filter_static,
                                                 min_eigenvalue)
        return frequency, self._reconstruct_eigenmode_field(eigenvector, domain)

    def _eigenpair(self, mode_index: int, domain: str, filter_static: bool = True,
                   min_eigenvalue: float = None) -> Tuple[float, np.ndarray]:
        """``(frequency [Hz], eigenvector)`` of mode *mode_index* of *domain*."""
        # mode_index points into the spectrum computed last (e.g. by
        # get_resonant_frequencies), so it is the mode the user just listed.
        # Recomputing a different number of modes around the shift would
        # return a different set, and index 1 would be another mode.
        cached_vecs = (getattr(self, '_eigenvectors_cache', None) or {}).get(domain)
        if (filter_static and min_eigenvalue is None and cached_vecs is not None
                and np.ndim(cached_vecs) == 2 and cached_vecs.shape[1] > mode_index):
            eigs, vecs = self._eigenvalues_cache[domain], cached_vecs
        else:
            eigs, vecs = self.get_eigenvectors(
                domain=domain,
                filter_static=filter_static,
                min_eigenvalue=min_eigenvalue,
                n_modes=max(50, mode_index + 10),  # as get_resonant_frequencies
                return_eigenvalues=True
            )

        if mode_index >= vecs.shape[1]:
            raise IndexError(
                f"Mode index {mode_index} out of range. "
                f"Only {vecs.shape[1]} modes available."
            )
        frequency = np.sqrt(np.maximum(np.real(eigs[mode_index]), 0)) / (2 * np.pi)
        return float(frequency), vecs[:, mode_index]

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
            print("Eigenmode Visualization")
            print(f"{'=' * 60}")
            print(f"Domain: {domain}")
            print(f"Mode index: {mode_index}")
            print(f"Resonant frequency: {frequency / 1e9:.6f} GHz")
            pr.echo(f"Angular frequency ω: {2 * np.pi * frequency:.4e} rad/s")
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
            from ngsolve import BoundaryFromVolumeCF  # noqa: F401  (availability)
            use_boundary = True
        except ImportError:
            use_boundary = False

        def draw_field(cf, label):
            if use_boundary:
                from ngsolve import BoundaryFromVolumeCF
                _display_webgui_fallback(Draw(BoundaryFromVolumeCF(cf), mesh, **draw_kwargs))
            else:
                _display_webgui_fallback(Draw(cf, mesh, **draw_kwargs))
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
                pr.echo(f"{'Index':<8} {'Frequency (GHz)':<18} {'ω² (rad²/s²)':<20}")
                print("-" * 50)

                for i, (f, e) in enumerate(zip(freqs, eigs)):
                    print(f"{i:<8} {f / 1e9:<18.6f} {e:<20.4e}")

            except Exception as e:
                print(f"\nDomain: {d} - Error: {e}")

        print("=" * 70)

    # =========================================================================
    # Figures of merit: external Q, R/Q, wall Q, peak fields, ...
    # =========================================================================

    def _eigen_port_coupling(self, domain: str) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """``(B, [(port, mode), ...])``: B's rows follow the eigenvector
        entries of *domain*, its columns the port modes."""
        raise NotImplementedError(
            f"{type(self).__name__} gives no port coupling for eigenmodes of '{domain}'")

    def _eigen_energy_operators(self, domain: str) -> Tuple[Any, Any, Any]:
        """``(M, C, D)`` in the eigenvector coordinates of *domain*.

        ``x^H M x / 2`` is the stored energy (M None: the coordinates are
        mass-normalised), ``x^H (C + w D) x / 2`` the material loss; C and D
        are None without conductivity or loss tangent.
        """
        raise NotImplementedError(f"{type(self).__name__} gives no energy operators")

    def _eigen_fds(self) -> Any:
        """The full-order solver: boundary names, materials and geometry."""
        return None

    def _eigen_mode_pieces(self, vector: np.ndarray, domain: str,
                           axis: str) -> List[ModePiece]:
        """The field of an eigenvector, one :class:`ModePiece` per mesh."""
        return [self._mode_piece(self._get_mesh_for_plotting(domain),
                                 self._reconstruct_eigenmode_field(vector, domain))]

    def _mode_piece(self, mesh, E, shift: float = 0.0) -> ModePiece:
        """A :class:`ModePiece` with the walls and materials of the solver."""
        from ngsolve import CoefficientFunction
        fds = self._eigen_fds()
        names = list(mesh.GetMaterials())
        props = {}
        for name in names:
            try:
                props[name] = fds._material_props(name)
            except Exception:
                props[name] = (1.0, 1.0, 0.0, 0.0)
        return ModePiece(mesh=mesh, E=E, shift=shift,
                         walls=getattr(fds, 'bc', None) or 'default',
                         mu_r=CoefficientFunction([props[n][1] for n in names]),
                         eps_r={n: props[n][0] for n in names})

    def _eigen_energy(self, vector: np.ndarray, domain: str, w: float) -> Tuple[float, float]:
        """``(U, P_diel)`` of an eigenvector at its own amplitude."""
        M, C, D = self._eigen_energy_operators(domain)
        x = np.asarray(vector)

        def quad(A):
            return 0.0 if A is None else float(np.real(np.vdot(x, A @ x)))

        U = 0.5 * (quad(M) if M is not None else float(np.vdot(x, x).real))
        return U, 0.5 * quad(C) + 0.5 * w * quad(D)

    def _default_eigen_domain(self) -> str:
        available = self._get_available_eigen_domains()
        return 'global' if 'global' in available else available[0]

    def get_external_q(
            self,
            fmin: float = None,
            fmax: float = None,
            domain: str = None,
            refine: int = 10,
    ) -> Dict[str, Any]:
        """Loaded resonances: frequency, loaded Q and external Q per port.

        Every port mode is terminated in its reference impedance Z0, the
        matched load the S-parameters assume, and the loaded eigenproblem of
        the reduced model

            (A + j w B Y0 B^T - w^2) x = 0,        Y0 = diag(1 / Z0)

        is solved exactly (linearised to size 2r).  Its complex eigenvalues
        are the loaded resonances, ``Q_L = Re(w) / (2 Im(w))``.  The external
        Q of a port splits that damping by the power the port takes from the
        loaded mode, ``Re(Y0) |B^T x|^2`` summed over the port's modes.

        The residues of the closed problem (port faces as magnetic walls)
        give Qext only when nothing else couples to the port.  A feed line
        between coupler and port face, or a strongly coupled neighbouring
        mode, adds reactance at the port that can change Qext by an order of
        magnitude; the loaded eigenproblem includes it.

        Only the external loading is included, no wall or dielectric losses.

        Each loaded resonance belongs to one mode of the closed problem, the
        one its eigenvector overlaps most, and each closed mode has at most
        one: a strongly damped mode whose best match is another mode's loaded
        resonance is left out.  The Z0 of a TE/TM mode depends on frequency:
        it is evaluated at the mode's own resonance, re-solved until the
        frequency settles, at most *refine* times.

        Parameters
        ----------
        fmin, fmax : float, optional
            Band in GHz; loaded resonances outside it are left out.
        domain : str, optional
            As :meth:`get_eigenmode`.
        refine : int
            Most re-solves with Z0 at the resonance (TE/TM ports only).

        Returns
        -------
        dict
            ``frequencies`` [Hz] and ``Q_L`` of the loaded resonances,
            ``Qext`` ({port: array}), ``Qext_mode`` ({'port(m)': array},
            1-based mode), and ``mode_index`` / ``f_closed``: the closed-problem
            mode each one belongs to (for :meth:`get_eigenmode`, :meth:`get_rq`
            and :meth:`get_figures_of_merit`), a different one for each.
        """
        domain = domain or self._default_eigen_domain()
        _M, K, free, n_dof = self._get_eigen_system_matrices(domain)
        if free is not None:
            raise NotImplementedError(
                "get_external_q() needs a reduced model, which holds the in-band "
                "response exactly: fds.fom.reduce(tol).get_external_q().")
        A = np.asarray(K)
        A = 0.5 * (A + A.conj().T)
        B, pairs = self._eigen_port_coupling(domain)
        B = np.asarray(B)
        r = A.shape[0]

        def admittance(f):
            y = []
            for port, mode in pairs:
                z0 = self._port_wave_impedance(port, mode, f)
                if z0 is None:
                    z0 = self._get_port_impedance(port, mode, f)
                y.append(1.0 / complex(z0))
            return np.array(y)

        f_lo = max((fmin or 0) * 1e9, np.sqrt(self.DEFAULT_MIN_EIGENVALUE) / (2 * np.pi))
        f_hi = np.inf if fmax is None else fmax * 1e9
        band = getattr(self, 'frequencies', None)
        f_ref = (0.5 * (f_lo + f_hi) if np.isfinite(f_hi)
                 else float(np.mean(band)) if band is not None and len(band) else 1e9)
        w0 = 2 * np.pi * f_ref

        def system(f):
            """Linearised loaded problem with Z0 at *f*; w = s * w0 keeps it well scaled."""
            y0 = admittance(f)
            G = (B * y0) @ B.T
            return np.block([[np.zeros((r, r)), np.eye(r)], [A / w0 ** 2, 1j * G / w0]]), y0

        def unit_pairs(s, X):
            keep = s.real > 0
            X = X[:r, keep]
            return s[keep] * w0, X / np.linalg.norm(X, axis=0)

        def loaded(f):
            L, y0 = system(f)
            return (*unit_pairs(*sl.eig(L)), y0)

        def match(modes, w_all, X_all):
            """One loaded eigenpair per closed mode, the one it overlaps most."""
            overlap = np.abs(modes.conj().T @ X_all)
            rows, picks = linear_sum_assignment(-overlap)
            return w_all[picks], X_all[:, picks], overlap[rows, picks]

        def rematch(f, modes, w, fit):
            """:func:`match` with Z0 at *f*, starting from the current eigenvalues *w*.

            A re-solve moves the eigenvalues only a little, so shift-invert at
            each w finds the candidates; if a mode then matches clearly worse
            than before, every eigenpair is computed."""
            L, y0 = system(f)
            shifts = []
            for wi in w:
                if all(abs(wi - s) > 1e-3 * abs(wi) for s in shifts):
                    shifts.append(wi)
            try:
                if len(w) + 4 >= 2 * r - 2:
                    raise ValueError("small system: solve it whole")
                parts = [nearest_eigs(L, k=len(w) + 4, sigma=s / w0) for s in shifts]
                w_all, X_all = unit_pairs(np.concatenate([s for s, _ in parts]),
                                          np.hstack([X for _, X in parts]))
                once = [j for j in range(len(w_all)) if not any(
                    abs(np.vdot(X_all[:, i], X_all[:, j])) > 0.999 for i in range(j))]
                if len(once) < len(w):
                    raise ValueError("too few candidates: solve it whole")
                result = match(modes, w_all[once], X_all[:, once])
                if np.any(result[2] < 0.9 * fit):
                    raise ValueError("a mode matches worse: solve it whole")
            except (ValueError, ArpackNoConvergence):
                result = match(modes, *unit_pairs(*sl.eig(L)))
            return (*result, y0)

        y_ref = admittance(f_ref)
        dispersive = not np.allclose(admittance(1.01 * f_ref), y_ref)
        same_everywhere = None if dispersive else loaded(f_ref)

        # the closed-problem modes (and the get_eigenmode() cache) to map onto
        eigs, V = self.get_eigenvectors(domain=domain, n_modes=n_dof, return_eigenvalues=True)
        f_closed = np.sqrt(np.maximum(np.real(eigs), 0.0)) / (2 * np.pi)
        V = np.asarray(V)

        # Each loaded resonance grows from a closed-problem mode: the loaded problem
        # is solved with Z0 near that mode's frequency, and the mode takes the
        # eigenpair its eigenvector overlaps most, one each. Modes within 0.1 % of
        # each other are matched together, so a degenerate pair keeps both members.
        # The re-solves with Z0 at the new frequency follow those eigenpairs.
        # Without dispersion one solve holds every loaded resonance.
        margin = 1.0 + self.TRAINING_BAND_MARGIN
        near = np.flatnonzero((f_closed >= f_lo / margin) & (f_closed <= f_hi * margin))
        near = near[np.argsort(f_closed[near])]
        if not dispersive:
            groups = [near] if len(near) else []
        else:
            breaks = np.flatnonzero(np.diff(f_closed[near]) > 1e-3 * f_closed[near][1:]) + 1
            groups = np.split(near, breaks) if len(near) else []

        found = []                          # (w, x, y0, closed mode, overlap, group)
        for g, group in enumerate(groups):
            f = float(np.mean(f_closed[group]))
            w_all, X_all, y0 = same_everywhere or loaded(f)
            w, x, fit = match(V[:, group], w_all, X_all)
            for _ in range(refine if dispersive else 0):
                f_new = float(np.mean(w.real)) / (2 * np.pi)
                if abs(f_new - f) <= 1e-7 * f:
                    break
                f = f_new
                w, x, fit, y0 = rematch(f, V[:, group], w, fit)
            found += [(w[k], x[:, k], y0, int(i), fit[k], g) for k, i in enumerate(group)]

        # a strongly damped mode can grow out of two groups: keep the closer match
        found.sort(key=lambda item: -item[4])
        rows, kept = [], []
        for w, x, y0, mode, _, g in found:
            if not f_lo <= w.real / (2 * np.pi) <= f_hi:
                continue
            if any(g2 != g and abs(w - w2) <= 1e-4 * abs(w) and abs(np.vdot(x2, x)) > 0.99
                   for w2, x2, g2 in kept):
                continue
            kept.append((w, x, g))
            power = np.real(y0) * np.abs(B.T @ x) ** 2        # per port mode
            # evanescent port modes (imaginary Z0) take no power: Q_L = inf
            q_l = w.real / (2 * w.imag) if w.imag > 0 and power.sum() > 0 else np.inf
            rows.append((w.real / (2 * np.pi), q_l, power, mode))
        rows.sort(key=lambda row: row[0])

        n = len(rows)
        freqs = np.array([row[0] for row in rows])
        q_l = np.array([row[1] for row in rows])
        power = np.array([row[2] for row in rows]).reshape(n, len(pairs))
        total = power.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            def q_of(cols):
                share = power[:, cols].sum(axis=1)
                return np.where(share > 0, q_l * total / share, np.inf)
            ports = list(dict.fromkeys(p for p, _ in pairs))
            mode_index = np.array([row[3] for row in rows], dtype=int)
            return {
                'frequencies': freqs,
                'Q_L': q_l,
                'Qext': {p: q_of([j for j, (pp, _) in enumerate(pairs) if pp == p])
                         for p in ports},
                'Qext_mode': {f"{p}({m + 1})": q_of([j]) for j, (p, m) in enumerate(pairs)},
                'mode_index': mode_index,
                'f_closed': f_closed[mode_index] if n else np.array([]),
            }

    def get_rq(
            self,
            mode_index: int,
            domain: str = None,
            axis: str = 'Z',
            offset: Tuple[float, float] = (0.0, 0.0),
            span: Tuple[float, float] = None,
            n_points: int = 2001,
    ) -> Dict[str, float]:
        """R/Q of one eigenmode for a beam (v = c) along a line.

        ``R/Q = V^2 / (w U)`` with ``V = |int E_s exp(j w s / c) ds|`` along
        the line parallel to *axis* through the transverse point *offset*,
        and ``U = (eps0/2) int eps_r |E|^2 dV``, the stored energy.  This is
        the accelerator (linac) convention, as in cavsim2d.  A dipole needs
        an offset from the axis: take it in both transverse directions to
        catch both polarisations, or use the transverse R/Q of
        :meth:`get_figures_of_merit`.

        Parameters
        ----------
        mode_index : int
            As :meth:`get_eigenmode` (the spectrum computed last).
        axis : {'X', 'Y', 'Z'}
            Beam direction.
        offset : (float, float)
            Transverse position [m], in the order of the two other axes
            (x, y for Z; y, z for X; x, z for Y).
        span : (float, float), optional
            Line start and end along *axis* [m]; default: the model's extent.
        n_points : int
            Samples along the line.

        Returns
        -------
        dict
            ``frequency`` [Hz], ``V`` [V], ``U`` [J] and ``RQ`` [Ohm], for the
            mode scaled to a stored energy of 1 J.
        """
        domain = domain or self._default_eigen_domain()
        freq, x = self._eigenpair(mode_index, domain)
        w = 2 * np.pi * freq
        U, _ = self._eigen_energy(x, domain, w)
        pieces = self._eigen_mode_pieces(x, domain, axis)
        a = 'XYZ'.index(axis.upper())
        s = beam_line(pieces, a, span, n_points)
        V = abs(voltage(field_on_line(pieces, a, offset, s)[0], s, w)) / np.sqrt(U)
        return {'frequency': freq, 'V': V, 'U': 1.0, 'RQ': V ** 2 / w}

    def get_figures_of_merit(
            self,
            mode_index: int,
            domain: str = None,
            axis: str = 'Z',
            offset: Tuple[float, float] = (0.0, 0.0),
            span: Tuple[float, float] = None,
            n_points: int = 2001,
            beta: float = 1.0,
            active_length: float = None,
            n_cells: int = None,
            conductivity: float = SIGMA_COPPER,
            surface_resistance: float = None,
            walls: str = None,
            kick_step: float = None,
    ) -> Dict[str, float]:
        """Cavity figures of merit of one eigenmode, as cavsim2d reports them.

        The keys and units are cavsim2d's (``'R/Q [Ohm]'``, ``'Epk/Eacc []'``,
        ``'Bpk/Eacc [mT/MV/m]'``, ...).  Absolute quantities (voltages,
        fields, losses) are for the mode scaled to a stored energy of 1 J.

        - Beam: ``Vacc`` along the line through *offset* parallel to *axis*
          for a charge at ``beta * c0``; ``Eacc = Vacc / active_length``;
          ``R/Q = Vacc^2 / (w U)`` (linac convention, twice the circuit one);
          the mode's loss factor ``k_loss = Vacc^2 / (4 U)``.
        - Transverse kick (Panofsky-Wenzel): ``Vt = (beta c0 / w) |grad_t V|``
          at *offset*, ``Et``, ``R/Q_t = Vt^2 / (w U)`` and the kick factor
          ``k_kick = (w / (beta c0)) Vt^2 / (4 U)``.  For a dipole it is the
          cavsim2d m = 1 value; a monopole on the axis gives ~0.
        - Walls: peak surface fields ``Epk``, ``Hpk``, ``Bpk``; the wall
          loss ``Ploss = (Rs / 2) int |H|^2 dS`` with the surface resistance
          of *conductivity* (copper by default) or *surface_resistance*;
          the geometry factor ``G = Q_wall Rs`` (independent of the wall
          material), ``Rsh = R/Q Q`` and ``GR/Q``.
        - Materials: with a loss tangent or conductivity, ``Q_diel`` and
          ``Pdiel`` from the loss terms, and ``Q`` the unloaded Q of walls and
          materials together (``1/Q = 1/Q_wall + 1/Q_diel``).  With several
          materials, each one's share of the electric energy (``U_frac_*``)
          and its peak field (``Epk_*``).
        - Multi-cell (*n_cells* > 1): field flatness ``ff``, min/max of the
          cells' on-axis peaks of ``|E_s|``.

        The ports are magnetic walls in this eigenproblem and are not
        counted as walls; the external Q is :meth:`get_external_q`.
        Re-entrant edges have singular fields, so Epk (and Hpk at a sharp
        edge) grows as the mesh is refined there.

        Parameters
        ----------
        mode_index : int
            As :meth:`get_eigenmode` (the spectrum computed last).
        domain : str, optional
            As :meth:`get_eigenmode`.
        axis, offset, span, n_points
            The beam line, as :meth:`get_rq`.
        beta : float
            Particle velocity over c0, for the transit-time phase.
        active_length : float, optional
            Length [m] that ``Eacc`` is normalised to.  Default: the
            geometry's ``active_length()`` (an elliptical cavity:
            ``2 L n_cells``, as cavsim2d), else the length of the line.
        n_cells : int, optional
            Cells, for the field flatness.  Default: the geometry's.
        conductivity : float
            Wall conductivity [S/m].
        surface_resistance : float, optional
            Wall surface resistance [Ohm], replacing *conductivity*.
        walls : str, optional
            Boundaries that are conducting walls (a region pattern such as
            ``'default|coupler'``).  Default: the solver's ``bc``.
        kick_step : float, optional
            Transverse step [m] of the Panofsky-Wenzel gradient; default
            2 % of the smallest transverse extent, halved until the shifted
            lines stay inside the aperture.

        Returns
        -------
        dict
        """
        domain = domain or self._default_eigen_domain()
        freq, x = self._eigenpair(mode_index, domain)
        w = 2 * np.pi * freq
        U, P_diel = self._eigen_energy(x, domain, w)
        pieces = self._eigen_mode_pieces(x, domain, axis)
        if walls is not None:
            for p in pieces:
                p.walls = walls

        geometry = getattr(self._eigen_fds(), 'geometry', None)
        if active_length is None:
            length = getattr(geometry, 'active_length', None)
            active_length = length() if callable(length) else None
        if n_cells is None:
            n_cells = int(getattr(geometry, 'n_cells', 1) or 1) * int(
                getattr(geometry, 'chain', 1) or 1)
        return figures_of_merit(
            pieces, freq, U, P_diel, axis=axis, offset=offset, span=span,
            n_points=n_points, beta=beta, active_length=active_length,
            n_cells=n_cells, conductivity=conductivity,
            surface_resistance_ohm=surface_resistance, kick_step=kick_step)

    def get_cell_coupling(self, first: int, last: int, domain: str = None) -> float:
        """Cell-to-cell coupling [%] of a passband, as cavsim2d.

        ``kcc = 2 (f_last - f_first) / (f_last + f_first)``, with *first* and
        *last* the passband's lowest (0) and highest (pi) mode, indexed as
        :meth:`get_eigenmode` (the spectrum computed last).
        """
        domain = domain or self._default_eigen_domain()
        f0 = self._eigenpair(first, domain)[0]
        f1 = self._eigenpair(last, domain)[0]
        return 200 * (f1 - f0) / (f1 + f0)

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
        eigs, _vecs = self.get_eigenvectors(**kwargs)
        return eigs

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

        import h5py
        
        # Determine path
        if path is None:
            if hasattr(self, '_project_path') and self._project_path:
                path = Path(self._project_path) / self.project_sub_path / "eigenmodes"
            else:
                raise ValueError("No path provided and no project_path available.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        h5_file = path / "eigenmodes.h5"
        
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
                path = Path(self._project_path) / self.project_sub_path / "eigenmodes"
                if not (path / "eigenmodes.h5").exists():
                    return  # Not found, silent return
            else:
                return  # No path, silent return

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

    def _eigen_fds(self):
        return self

    def _eigen_energy_operators(self, domain: str):
        if domain == 'global':
            return self.M_global, self.C_global, self.D_global
        return self.M[domain], self.C.get(domain), self.D.get(domain)


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

    def _eigen_training_band(self) -> Optional[Tuple[float, float]]:
        band = getattr(self, '_band', None)
        return (band["fmin_GHz"], band["fmax_GHz"]) if band else None

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

    def _eigen_port_coupling(self, domain: str):
        """Reduced port basis B_r of *domain* and its column order."""
        if domain not in self._B_r:
            raise NotImplementedError(
                f"No port coupling for the reduced domain '{domain}'; use one of "
                f"{list(self._B_r)} (a joined model: its concat object).")
        pairs = self._domain_port_mode_pairs(domain, self._n_modes_per_port or 1)
        return self._B_r[domain], [(p, m) for (_i, p, m) in pairs]

    def _eigen_fds(self):
        return getattr(self, 'solver', None)

    def _eigen_energy_operators(self, domain: str):
        # A single domain's 'global' spectrum is that domain's (see
        # _reconstruct_eigenmode_field); several domains join in a concat.
        if domain not in self._A_r and domain == 'global' and self.n_domains == 1:
            domain = self.domains[0]
        if domain not in self._A_r:
            raise NotImplementedError(
                f"No energy operators for the reduced domain '{domain}'; use one of "
                f"{list(self._A_r)} (a joined model: its concat object).")
        return None, self._C_r.get(domain), self._D_r.get(domain)


class ConcatEigenMixin(EigenMixinBase):
    """
    Eigenvalue mixin for ConcatenatedSystem.

    Uses coupled reduced matrices (A_coupled). Field reconstruction requires
    W_coupled and access to original structures' projection bases.
    """

    def _eigen_training_band(self) -> Optional[Tuple[float, float]]:
        # where the joined parts' training bands overlap
        return getattr(self, '_training_band', None)

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
                
            return self._reconstruct_field_from_vector(x_uncoupled, scales)

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

    def _eigen_fds(self):
        ref = getattr(self, '_solver_ref', None)
        return getattr(ref, 'solver', ref)

    def _structure_index(self, domain: str) -> int:
        for i, s in enumerate(self.structures):
            if s.domain == domain:
                return i
        raise KeyError(f"Domain '{domain}' not found")

    def _eigen_energy_operators(self, domain: str):
        # W_coupled is orthonormal and every section's reduced coordinates
        # are mass-normalised, so the coupled coordinates are too.
        if domain in (None, 'global'):
            return None, getattr(self, 'C_coupled', None), getattr(self, 'D_coupled', None)
        s = self.structures[self._structure_index(domain)]
        return None, getattr(s, 'Crd', None), getattr(s, 'Drd', None)

    def _eigen_port_coupling(self, domain: str):
        """The coupled system's external port basis; columns follow ``ports``
        (``'port1(1)'`` is port1, mode 0)."""
        import re
        if domain not in (None, 'global') or self.B_coupled is None:
            raise NotImplementedError(
                "The loaded eigenproblem of a joined model is its coupled one "
                "(domain='global').")
        pairs = []
        for name in self.ports:
            m = re.match(r'^(.*)\((\d+)\)$', name)
            pairs.append((m.group(1), int(m.group(2)) - 1) if m else (name, 0))
        return np.asarray(self.B_coupled), pairs

    def _eigen_mode_pieces(self, vector, domain, axis):
        """One mesh for a glued model; for a netlist, each section on its own
        mesh, the sections laid end to end along *axis* in list order."""
        vector = np.asarray(vector)
        if domain not in (None, 'global'):
            i = self._structure_index(domain)
            if self.mesh is not None:
                return super()._eigen_mode_pieces(vector, domain, axis)
            return [self._section_mode_piece(i, self.structures[i].reconstruct(vector))]
        x = self.W_coupled @ vector
        if self.mesh is not None:
            E = self._reconstruct_field_from_vector(
                x, np.ones(self.n_structures, dtype=complex))
            return [self._mode_piece(self.mesh, E)]
        a = 'XYZ'.index(axis.upper())
        pieces, end = [], None
        for i, st in enumerate(self.structures):
            start = self._structure_dof_offsets[i]
            piece = self._section_mode_piece(i, st.reconstruct(x[start:start + st.r]))
            z = np.asarray(piece.mesh.ngmesh.Coordinates())[:, a]
            piece.shift = 0.0 if end is None else end - float(z.min())
            end = float(z.max()) + piece.shift
            pieces.append(piece)
        return pieces

    def _section_mode_piece(self, section_idx: int, x_full) -> ModePiece:
        """A netlist section's field on its own mesh (a complex space: the
        coupled eigenvector need not be real)."""
        from ngsolve import GridFunction, HCurl
        from cavsim3d.solvers.nedelec import hcurl_flags, kind_of
        mesh, fes = self._section_mesh_fes(section_idx)
        if not fes.is_complex:
            fes = HCurl(mesh, order=fes.globalorder, complex=True,
                        **hcurl_flags(kind_of(fes)))
        E = GridFunction(fes)
        vec = E.vec.FV().NumPy()
        n = min(len(vec), len(x_full))
        vec[:] = 0
        vec[:n] = np.asarray(x_full)[:n]
        return self._mode_piece(mesh, E)

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