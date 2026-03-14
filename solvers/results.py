from __future__ import annotations
"""
Result wrapper objects for the cavsim3d computation graph.

These lightweight objects wrap solver results and make the computation
graph navigable via attribute access.  Each node in the graph inherits
from PlotMixin so it can plot itself directly.

Graph overview
--------------

Single-solid
  fds.fom                         -> FOMResult
  fds.fom.reduce()                -> ModelOrderReduction
  mor.solve(fmin, fmax, n)        -> (updates MOR in-place)
  mor.reduce()                    -> ModelOrderReduction  (2nd level)

Multi-solid
  fds.foms                        -> FOMCollection      (per-domain list)
  fds.foms[0]                     -> FOMResult           (first domain)
  fds.foms.reduce()               -> ROMCollection      (new each call)
  fds.foms.concatenate()          -> ConcatenatedSystem  (W=I, full-order)
  roms.concatenate()              -> ConcatenatedSystem  (reduced)
  cs.solve(fmin, fmax, n)         -> (updates CS in-place)
  cs.reduce()                     -> ModelOrderReduction (2nd-level POD)
"""


from typing import Dict, Iterator, List, Optional, Tuple, Union
import warnings
import numpy as np

from utils.plot_mixin import PlotMixin
from core.persistence import H5Serializer, ProjectManager
import h5py
from pathlib import Path
from datetime import datetime
import json


# =============================================================================
# FOMResult
# =============================================================================

class FOMResult(PlotMixin):
    """
    Wrapper around a single solved FOM (one domain or a global coupled result).

    Attributes
    ----------
    domain : str
        Domain name, or ``'global'`` for the coupled/entire-mesh result.
    frequencies : np.ndarray
        Frequency array in Hz.
    Z_dict, S_dict : dict
        Parameter dictionaries with keys like ``'1(1)1(1)'``.
    n_ports : int
    ports : list of str
    """

    def __init__(
        self,
        *,
        domain: str,
        frequencies: np.ndarray,
        Z_matrix: Optional[np.ndarray],
        S_matrix: Optional[np.ndarray],
        Z_dict: Optional[Dict],
        S_dict: Optional[Dict],
        n_ports: int,
        ports: List[str],
        n_modes_per_port: int = 1,
        # Residual data from iterative solver
        residual_data: Optional[Dict] = None,
        # Back-reference to the FDS
        _solver_ref=None,
    ):
        self.domain = domain
        self.frequencies = frequencies
        self._Z_matrix = Z_matrix
        self._S_matrix = S_matrix
        self._Z_dict = Z_dict
        self._S_dict = S_dict
        self.n_ports = n_ports
        self.ports = ports
        self._n_modes_per_port = n_modes_per_port
        self._residual_data = residual_data
        self._solver_ref = _solver_ref

        # Lazy cache for backward-compatible .rom property
        self._rom_cache = None

    # ------------------------------------------------------------------
    # PlotMixin requirements
    # ------------------------------------------------------------------

    @property
    def Z_dict(self) -> Optional[Dict]:
        if self._Z_dict is None and self._Z_matrix is not None:
            self._Z_dict = self._rebuild_dict(self._Z_matrix)
        return self._Z_dict

    @property
    def S_dict(self) -> Optional[Dict]:
        if self._S_dict is None and self._S_matrix is not None:
            self._S_dict = self._rebuild_dict(self._S_matrix)
        return self._S_dict

    def _rebuild_dict(self, matrix: np.ndarray) -> Dict:
        """Utility to reconstruct port/mode mapping dictionary from a matrix."""
        res_dict = {'frequencies': self.frequencies}
        n_p = matrix.shape[1]
        n_modes = self._n_modes_per_port
        
        for row in range(n_p):
            prow = row // n_modes + 1
            mrow = row % n_modes + 1
            for col in range(n_p):
                pcol = col // n_modes + 1
                mcol = col % n_modes + 1
                key = f'{pcol}({mcol}){prow}({mrow})'
                res_dict[key] = matrix[:, row, col]
        return res_dict

    # ------------------------------------------------------------------
    # Backward-compatible ROM accessor
    # ------------------------------------------------------------------

    @property
    def rom(self):
        """
        Access the cached reduced-order model.

        Returns the ROM only if it has already been computed via
        ``fom.reduce()`` or loaded from disk. Does **not** trigger
        reduction automatically.

        Raises
        ------
        RuntimeError
            If no ROM has been computed yet.
        """
        if self._rom_cache is None:
            raise RuntimeError(
                "No reduced-order model available. "
                "Call fom.reduce() first to compute the ROM."
            )
        return self._rom_cache

    # ------------------------------------------------------------------
    # Explicit reduce / concatenate
    # ------------------------------------------------------------------

    def reduce(self, tol: float = 1e-6, max_rank: Optional[int] = None):
        """
        Reduce this FOM via POD model-order reduction.

        Parameters
        ----------
        tol : float
            SVD truncation tolerance (relative to largest singular value).
        max_rank : int, optional
            Maximum rank for the reduced model.

        Returns
        -------
        ModelOrderReduction
            A new, independent reduced-order model. Call ``.solve(fmin, fmax, n)``
            on it to compute Z/S over a frequency range.
        """
        if self._solver_ref is None:
            raise RuntimeError(
                "Cannot reduce: no solver reference available. "
                "Ensure this FOMResult was created by FrequencyDomainSolver."
            )
        from rom.reduction import ModelOrderReduction

        mor = ModelOrderReduction(self._solver_ref)
        mor.reduce(tol=tol, max_rank=max_rank)
        self._rom_cache = mor
        return mor

    def concatenate(self):
        """
        Concatenate this FOM with other domains.

        Not available for single-solid systems — raises a warning.
        For multi-solid concatenation, use ``fds.foms.concatenate()`` instead.
        """
        warnings.warn(
            "concatenate() is not available on a single FOMResult. "
            "Concatenation requires multiple solids — use fds.foms.concatenate() instead.",
            UserWarning,
            stacklevel=2,
        )
        return None

    # ------------------------------------------------------------------
    # Eigenvalue access
    # ------------------------------------------------------------------

    def get_eigenvalues(self, **kwargs):
        """
        Compute eigenvalues from (K, M) generalized eigenproblem.

        Delegates to the underlying FrequencyDomainSolver.
        """
        if self._solver_ref is not None and hasattr(self._solver_ref, 'get_eigenvalues'):
            return self._solver_ref.get_eigenvalues(**kwargs)
        raise RuntimeError("Eigenvalues not available for this FOMResult.")

    def get_resonant_frequencies(self, **kwargs):
        if self._solver_ref is not None and hasattr(self._solver_ref, 'get_resonant_frequencies'):
            return self._solver_ref.get_resonant_frequencies(**kwargs)
        raise RuntimeError("Resonant frequencies not available for this FOMResult.")

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """
        Standardized API for retrieving eigenvalues and eigenvectors.
        """
        if self._solver_ref is not None and hasattr(self._solver_ref, 'get_eigenmodes'):
            # Delegate to solver, but filter for this domain if not global
            domain = self.domain if self.domain != 'global' else None
            res = self._solver_ref.get_eigenmodes(domain=domain, **kwargs)
            
            # Hierarchical save if possible
            if _auto_save:
                self._auto_save_eigenmodes(res, **kwargs)
            
            return res
        raise RuntimeError("Eigenmodes not available for this FOMResult.")

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        """Helper to save eigenmodes to the mirrored project structure."""
        if self._solver_ref is None or not hasattr(self._solver_ref, '_project_path'):
            return
        
        project_path = Path(self._solver_ref._project_path)
        # Mirror: fds/fom -> fds/fom/eigenmode, fds/foms/solid -> fds/foms/solid/eigenmode
        sub_path = "fom" if self.domain == 'global' else f"foms/{self.domain}"
        save_path = project_path / "fds" / sub_path / "eigenmode"
        
        try:
            # We need FrequencyDomainSolver to have save_eigenmodes (from mixin)
            if hasattr(self._solver_ref, 'save_eigenmodes'):
                self._solver_ref.save_eigenmodes(save_path, domain=(self.domain if self.domain != 'global' else None), **kwargs)
        except Exception as e:
            print(f"Warning: Could not auto-save eigenmodes to {save_path}: {e}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """
        Save FOM results to disk.
        
        Saves matrices (K, M, B) to separate files in matrices/ and 
        S/Z parameters to s.h5/z.h5.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. Save matrices (K, M, B) separately
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        
        if self._solver_ref is not None:
            if self.domain in getattr(self._solver_ref, 'K', {}):
                with h5py.File(mat_path / "K.h5", "a") as f:
                    H5Serializer.save_dataset(f, "data", self._solver_ref.K.get(self.domain))
                with h5py.File(mat_path / "M.h5", "a") as f:
                    H5Serializer.save_dataset(f, "data", self._solver_ref.M.get(self.domain))
                with h5py.File(mat_path / "B.h5", "a") as f:
                    H5Serializer.save_dataset(f, "data", self._solver_ref.B.get(self.domain))

        # 2. Save S and Z results separately
        if self._Z_matrix is not None:
            with h5py.File(path / "z.h5", "a") as f:
                H5Serializer.save_dataset(f, "data", self._Z_matrix)
        if self._S_matrix is not None:
            with h5py.File(path / "s.h5", "a") as f:
                H5Serializer.save_dataset(f, "data", self._S_matrix)

        # 3. Save remaining data to snapshots.h5
        with h5py.File(path / "snapshots.h5", "a") as f:
            if self.frequencies is not None:
                H5Serializer.save_dataset(f, "frequencies", self.frequencies)
            if self._residual_data:
                H5Serializer.save_dataset(f, "residual_data", self._residual_data)
        
        # 3. Save metadata
        metadata = {
            "domain": self.domain,
            "n_ports": self.n_ports,
            "ports": self.ports,
            "n_modes_per_port": self._n_modes_per_port,
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)

    @classmethod
    def load(cls, path: Union[str, Path], _solver_ref=None) -> FOMResult:
        """Load FOM results from disk."""
        path = Path(path)
        
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load S/Z from their new dedicated files
        Z_matrix = None
        if (path / "z.h5").exists():
            with h5py.File(path / "z.h5", "r") as f:
                Z_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None
        
        S_matrix = None
        if (path / "s.h5").exists():
            with h5py.File(path / "s.h5", "r") as f:
                S_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None

        # Load frequencies and residual_data from snapshots.h5 (legacy compatibility or current remaining)
        frequencies = None
        residual_data = None
        if (path / "snapshots.h5").exists():
            with h5py.File(path / "snapshots.h5", "r") as f:
                frequencies = H5Serializer.load_dataset(f["frequencies"]) if "frequencies" in f else None
                # Support legacy loading if matrices were here
                if Z_matrix is None and "Z_matrix" in f:
                    Z_matrix = H5Serializer.load_dataset(f["Z_matrix"])
                if S_matrix is None and "S_matrix" in f:
                    S_matrix = H5Serializer.load_dataset(f["S_matrix"])
                residual_data = H5Serializer.load_dataset(f["residual_data"]) if "residual_data" in f else None

        # Restore results into _solver_ref if available
        if _solver_ref is not None:
            if not hasattr(_solver_ref, '_residuals') or _solver_ref._residuals is None:
                _solver_ref._residuals = {}
            if Z_matrix is not None: _solver_ref._Z_global_coupled = Z_matrix
            if S_matrix is not None: _solver_ref._S_global_coupled = S_matrix
            if residual_data is not None: _solver_ref._residuals['global'] = residual_data

        # Restore matrices into _solver_ref if available
        if _solver_ref is not None:
            domain = metadata["domain"]
            mat_path = path / "matrices"
            if mat_path.exists():
                if (mat_path / "K.h5").exists():
                    with h5py.File(mat_path / "K.h5", "r") as f:
                        _solver_ref.K[domain] = H5Serializer.load_sparse_csr(f["data"])
                if (mat_path / "M.h5").exists():
                    with h5py.File(mat_path / "M.h5", "r") as f:
                        _solver_ref.M[domain] = H5Serializer.load_sparse_csr(f["data"])
                if (mat_path / "B.h5").exists():
                    with h5py.File(mat_path / "B.h5", "r") as f:
                        _solver_ref.B[domain] = H5Serializer.load_dataset(f["data"])
            elif (path / "matrices.h5").exists():
                with h5py.File(path / "matrices.h5", "r") as f:
                    if "K" in f: _solver_ref.K[domain] = H5Serializer.load_sparse_csr(f["K"])
                    if "M" in f: _solver_ref.M[domain] = H5Serializer.load_sparse_csr(f["M"])
                    if "B" in f: _solver_ref.B[domain] = H5Serializer.load_dataset(f["B"])

        # Build Z/S dicts if matrices are loaded
        res = cls(
            domain=metadata["domain"],
            frequencies=frequencies,
            Z_matrix=Z_matrix,
            S_matrix=S_matrix,
            Z_dict=None, # will be build lazily or we can build now
            S_dict=None,
            n_ports=metadata["n_ports"],
            ports=metadata["ports"],
            n_modes_per_port=metadata.get("n_modes_per_port", 1),
            residual_data=residual_data,
            _solver_ref=_solver_ref
        )
        return res

    def __repr__(self) -> str:
        n_freq = len(self.frequencies) if self.frequencies is not None else 0
        return (f"FOMResult(domain='{self.domain}', "
                f"n_ports={self.n_ports}, n_freq={n_freq})")


# =============================================================================
# FOMCollection
# =============================================================================

class FOMCollection(PlotMixin):
    """
    Ordered collection of per-domain :class:`FOMResult` objects.

    Supports indexing (``fds.foms[0]``), iteration, and length.

    Methods
    -------
    reduce(tol, max_rank) -> ROMCollection
        Reduce all domains. Returns a new ROMCollection each call.
    concatenate() -> ConcatenatedSystem
        Concatenate all domains via Kirchhoff coupling (W=I for FOM-level).
    """

    def __init__(
        self,
        fom_list: List[FOMResult],
        *,
        _fds_ref=None,
    ):
        if not fom_list:
            raise ValueError("FOMCollection requires at least one FOMResult.")
        self._foms = fom_list
        self._fds_ref = _fds_ref

        # Lazy caches for backward-compatible properties
        self._roms_cache: Optional[ROMCollection] = None
        self._concat_cache = None

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> FOMResult:
        return self._foms[idx]

    def __len__(self) -> int:
        return len(self._foms)

    def __iter__(self) -> Iterator[FOMResult]:
        return iter(self._foms)

    # ------------------------------------------------------------------
    # PlotMixin — aggregate: plot all domains overlaid
    # ------------------------------------------------------------------

    @property
    def frequencies(self) -> np.ndarray:
        return self._foms[0].frequencies

    @property
    def Z_dict(self) -> Optional[Dict]:
        # Return first domain's Z_dict for the mixin; multi-domain plots
        # are handled by overriding plot_s/plot_z below.
        return self._foms[0].Z_dict

    @property
    def S_dict(self) -> Optional[Dict]:
        return self._foms[0].S_dict

    def plot_s(self, params=None, plot_type='db', ax=None, label=None,
               title=None, show=False, **kwargs):
        """Overlay S-parameters for every domain on a single Axes."""
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax)
        for fom in self._foms:
            lbl = f"{label or ''}{fom.domain}" if label else fom.domain
            fig, ax = fom.plot_s(params=params, plot_type=plot_type, ax=ax,
                                 label=lbl, title=title, **kwargs)
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def plot_z(self, params=None, plot_type='db', ax=None, label=None,
               title=None, show=False, **kwargs):
        """Overlay Z-parameters for every domain on a single Axes."""
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax)
        for fom in self._foms:
            lbl = f"{label or ''}{fom.domain}" if label else fom.domain
            fig, ax = fom.plot_z(params=params, plot_type=plot_type, ax=ax,
                                 label=lbl, title=title, **kwargs)
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def plot_eigenvalues(self, n_modes=30, ax=None, label=None,
                         title=None, show=False, **kwargs):
        """Overlay eigenfrequencies for every domain."""
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax, figsize=(10, 3))
        for fom in self._foms:
            lbl = f"{label or ''}{fom.domain}" if label else fom.domain
            try:
                fig, ax = fom.plot_eigenvalues(n_modes=n_modes, ax=ax,
                                               label=lbl, **kwargs)
            except RuntimeError:
                pass
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def plot_residual(self, what: str = 'both', ax=None, label: Optional[str] = None,
                      title: Optional[str] = None, show: bool = False, **kwargs):
        """Overlay iterative solver residuals for every domain."""
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax)
        for fom in self._foms:
            lbl = f"{label or ''}{fom.domain}" if label else fom.domain
            # Set a generic title for sub-plots to avoid flickering titles in overlays
            sub_title = title if title else (f"{fom.domain} Convergence" if len(self._foms) == 1 else None)
            try:
                fig, res = fom.plot_residual(what=what, ax=ax, label=lbl,
                                             title=sub_title, show=False, **kwargs)
                # If what='both', 'res' is (ax1, ax2). Use them for next domain.
                if what == 'both':
                    ax = res
                else:
                    ax = res
            except RuntimeError:
                pass

        if not title and len(self._foms) > 1:
            title = "Per-Domain Iterative Solver Convergence"

        if title:
            # Handle twin axes case returns tuple (ax1, ax2)
            if isinstance(ax, tuple):
                ax[0].set_title(title)
            else:
                ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Backward-compatible ROM/Concat accessors
    # ------------------------------------------------------------------

    @property
    def roms(self) -> 'ROMCollection':
        """
        Access the cached per-domain ROMs.

        Returns the ROM collection only if it has already been computed
        via ``foms.reduce()`` or loaded from disk.  Does **not** trigger
        reduction automatically.

        Raises
        ------
        RuntimeError
            If no ROMs have been computed yet.
        """
        if self._roms_cache is None:
            raise RuntimeError(
                "No reduced-order models available. "
                "Call foms.reduce() first to compute the ROMs."
            )
        return self._roms_cache

    @property
    def concat(self):
        """
        Access the cached concatenated system.

        Returns the concatenated system only if it has already been
        computed via ``foms.concatenate()`` or loaded from disk.
        Does **not** trigger concatenation automatically.

        Raises
        ------
        RuntimeError
            If no concatenated system has been computed yet.
        """
        if self._concat_cache is None:
            raise RuntimeError(
                "No concatenated system available. "
                "Call foms.concatenate() first."
            )
        return self._concat_cache

    # ------------------------------------------------------------------
    # Reduce and Concatenate
    # ------------------------------------------------------------------

    def reduce(self, tol: float = 1e-6, max_rank: Optional[int] = None) -> 'ROMCollection':
        """
        Reduce all domains via POD. Returns a new ROMCollection each call.

        Parameters
        ----------
        tol : float
            SVD truncation tolerance.
        max_rank : int, optional
            Maximum rank for all domains.

        Returns
        -------
        ROMCollection
            Collection of per-domain reduced models.
        """
        if self._fds_ref is None:
            raise RuntimeError("FOMCollection has no reference to the FrequencyDomainSolver.")

        from rom.reduction import ModelOrderReduction

        fds = self._fds_ref
        mor = ModelOrderReduction(fds)
        mor.reduce(tol=tol, max_rank=max_rank)

        self._roms_cache = ROMCollection(_fds_ref=self._fds_ref, _mor_ref=mor)
        return self._roms_cache

    def concatenate(self):
        """
        Concatenate per-domain FOMs via Kirchhoff coupling (W=I).

        Wraps each domain's full-order (K, M, B) matrices as ReducedStructure
        objects with ``is_full_order=True`` and ``W=I``, then builds a
        ConcatenatedSystem.

        .. warning::
            This creates large dense matrices since W=I preserves the full
            dimensionality. Intended primarily for testing and validation.

        Returns
        -------
        ConcatenatedSystem
            Coupled full-order system with ``.solve()`` and ``.reduce()`` methods.
        """
        if self._fds_ref is None:
            raise RuntimeError("FOMCollection has no reference to the FrequencyDomainSolver.")

        import scipy.sparse as sp
        from rom.structures import ReducedStructure
        from solvers.concatenation import ConcatenatedSystem

        fds = self._fds_ref

        # Warn about matrix size
        total_ndof = sum(fds._fes[d].ndof for d in fds.domains if d in fds._fes)
        warnings.warn(
            f"FOM-level concatenation creates dense matrices from full-order systems "
            f"(total DOFs: {total_ndof}). This may consume significant memory. "
            f"For large problems, consider fds.foms.reduce().concatenate() instead.",
            UserWarning,
            stacklevel=2,
        )

        structures = []
        for domain in fds.domains:
            K_d = fds.K[domain]
            M_d = fds.M[domain]
            B_d = fds.B[domain]

            # Get free DOFs
            fes_d = fds._fes[domain]
            free_dofs = np.array([i for i in range(fes_d.ndof) if fes_d.FreeDofs()[i]])
            n_free = len(free_dofs)

            # Extract free DOF submatrices and convert to dense
            K_free = K_d[np.ix_(free_dofs, free_dofs)].toarray()
            M_free = M_d[np.ix_(free_dofs, free_dofs)].toarray()
            B_free = B_d[free_dofs, :]

            # A = M⁻¹ K is not formed directly; for the reduced system
            # the convention is (A_r - ω²I)x = ωB_r u
            # where A_r = W^T K W and B_r = W^T B, with W=I here
            # So Ard = K_free (acting on free DOFs with M factored out via the eigenproblem)
            # Actually for consistency with the ROM convention:
            # A_r = V^T K V where V is the POD basis. With V=I (full order),
            # A_r = K_free. The mass matrix is implicit in the ω² I term because
            # the ROM formulation assumes mass-orthonormal basis.
            # For FOM W=I concatenation, we need to be consistent with how
            # ConcatenatedSystem.solve() works: (A - ω²I)x = ωBu
            # This means A should encode the stiffness-to-mass ratio.

            # Solve the generalized problem via: M^{-1}K x = ω² x
            # i.e. A = M_free^{-1} @ K_free (dense, but that's the W=I path)
            import scipy.linalg as sl
            # Use Cholesky or direct solve for M^{-1} K
            try:
                L = sl.cholesky(M_free, lower=True)
                M_inv_K = sl.cho_solve((L, True), K_free.T).T
                Ard = 0.5 * (M_inv_K + M_inv_K.T)  # Symmetrize
            except sl.LinAlgError:
                # M may be singular on free DOFs — use pseudo-inverse
                M_inv_K = sl.solve(M_free, K_free, assume_a='sym')
                Ard = 0.5 * (M_inv_K + M_inv_K.T)

            # B_r with W=I: B_rd = M_free^{-1} @ B_free (mass-weighted port basis)
            try:
                Brd = sl.cho_solve((L, True), B_free)
            except (sl.LinAlgError, NameError):
                Brd = sl.solve(M_free, B_free, assume_a='sym')

            domain_ports = fds.domain_port_map[domain]
            port_modes_d = {p: fds.port_modes[p] for p in domain_ports if p in fds.port_modes}

            structures.append(ReducedStructure(
                Ard=Ard,
                Brd=Brd,
                ports=domain_ports,
                port_modes=port_modes_d,
                domain=domain,
                r=n_free,
                n_full=n_free,
                is_full_order=True,
            ))

        concat = ConcatenatedSystem(
            structures=structures,
            port_impedance_func=fds._get_port_impedance,
            solver_ref=fds,
        )

        # Auto-detect sequential connections
        connections = []
        for i in range(len(fds.domains) - 1):
            port_a = fds.domain_port_map[fds.domains[i]][-1]    # last port of domain i
            port_b = fds.domain_port_map[fds.domains[i + 1]][0]  # first port of domain i+1
            connections.append(((i, port_a), (i + 1, port_b)))

        concat.define_connections(connections)
        concat.couple()

        self._concat_cache = concat
        return concat

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Save FOMCollection to multi-solid path."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save matrices in dedicated files inside a matrices/ folder
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(mat_path / "K.h5", "a") as fk, \
             h5py.File(mat_path / "M.h5", "a") as fm, \
             h5py.File(mat_path / "B.h5", "a") as fb:
            
            for fom in self._foms:
                if fom._solver_ref is not None:
                    domain = fom.domain
                    if domain in getattr(fom._solver_ref, 'K', {}):
                        H5Serializer.save_dataset(fk, domain, fom._solver_ref.K.get(domain))
                        H5Serializer.save_dataset(fm, domain, fom._solver_ref.M.get(domain))
                        H5Serializer.save_dataset(fb, domain, fom._solver_ref.B.get(domain))

        # 2. Save S and Z results separately
        with h5py.File(path / "z.h5", "a") as fz, \
             h5py.File(path / "s.h5", "a") as fs:
            for fom in self._foms:
                if fom._Z_matrix is not None:
                    H5Serializer.save_dataset(fz, fom.domain, fom._Z_matrix)
                if fom._S_matrix is not None:
                    H5Serializer.save_dataset(fs, fom.domain, fom._S_matrix)

        # 3. Save frequencies and residual data to snapshots.h5
        with h5py.File(path / "snapshots.h5", "a") as fsnap:
            if self.frequencies is not None:
                H5Serializer.save_dataset(fsnap, "frequencies", self.frequencies)
            for fom in self._foms:
                if fom._residual_data:
                    H5Serializer.save_dataset(fsnap, f"{fom.domain}/residual_data", fom._residual_data)
        
        # Metadata
        metadata = {
            "n_solids": len(self._foms),
            "solids": [
                {
                    "domain": f.domain,
                    "n_ports": f.n_ports,
                    "ports": f.ports,
                    "n_modes_per_port": f._n_modes_per_port
                } for f in self._foms
            ],
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)

        # Save cached concatenation if available
        if self._concat_cache is not None:
            self._concat_cache.save(path / "concat")

    @classmethod
    def load(cls, path: Union[str, Path], _fds_ref=None) -> FOMCollection:
        """Load FOMCollection from disk."""
        path = Path(path)
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        fom_list = []
        # Multi-file loading
        snap_path = path / "snapshots.h5"
        z_path = path / "z.h5"
        s_path = path / "s.h5"

        with h5py.File(snap_path, "r") as fs:
            frequencies = H5Serializer.load_dataset(fs["frequencies"]) if "frequencies" in fs else None
            
            # Load matrices into _fds_ref if available
            if _fds_ref is not None:
                mat_path = path / "matrices"
                if mat_path.exists():
                    # Load from modular folder
                    for mname in ["K", "M", "B"]:
                        mfile = mat_path / f"{mname}.h5"
                        if mfile.exists():
                            with h5py.File(mfile, "r") as f:
                                for domain in f:
                                    if mname in ["K", "M"]:
                                        _fds_ref.__dict__[mname][domain] = H5Serializer.load_sparse_csr(f[domain])
                                    else:
                                        _fds_ref.B[domain] = H5Serializer.load_dataset(f[domain])
                elif (path / "matrices.h5").exists():
                    # Legacy fallback
                    with h5py.File(path / "matrices.h5", "r") as fm:
                        for domain in fm:
                            if "K" in fm[domain]: _fds_ref.K[domain] = H5Serializer.load_sparse_csr(fm[f"{domain}/K"])
                            if "M" in fm[domain]: _fds_ref.M[domain] = H5Serializer.load_sparse_csr(fm[f"{domain}/M"])
                            if "B" in fm[domain]: _fds_ref.B[domain] = H5Serializer.load_dataset(fm[f"{domain}/B"])

            # Optional result files
            fz = h5py.File(z_path, "r") if z_path.exists() else {}
            fs_results = h5py.File(s_path, "r") if s_path.exists() else {}

            try:
                for solid_meta in metadata["solids"]:
                    domain = solid_meta["domain"]
                    
                    # Load Z
                    Z_matrix = None
                    if domain in fz:
                        Z_matrix = H5Serializer.load_dataset(fz[domain])
                    elif f"{domain}/Z_matrix" in fs: # Legacy
                        Z_matrix = H5Serializer.load_dataset(fs[f"{domain}/Z_matrix"])
                        
                    # Load S
                    S_matrix = None
                    if domain in fs_results:
                        S_matrix = H5Serializer.load_dataset(fs_results[domain])
                    elif f"{domain}/S_matrix" in fs: # Legacy
                        S_matrix = H5Serializer.load_dataset(fs[f"{domain}/S_matrix"])

                    residual_data = H5Serializer.load_dataset(fs[f"{domain}/residual_data"]) if f"{domain}/residual_data" in fs else None
                    
                    fom = FOMResult(
                        domain=domain,
                        frequencies=frequencies,
                        Z_matrix=Z_matrix,
                        S_matrix=S_matrix,
                        Z_dict=None,
                        S_dict=None,
                        n_ports=solid_meta["n_ports"],
                        ports=solid_meta["ports"],
                        n_modes_per_port=solid_meta.get("n_modes_per_port", 1),
                        residual_data=residual_data,
                        _solver_ref=_fds_ref
                    )
                    
                    # Populate solver dictionaries for consistency using reconstructed maps
                    if _fds_ref is not None:
                        if not hasattr(_fds_ref, '_residuals') or _fds_ref._residuals is None:
                            _fds_ref._residuals = {}
                        if residual_data is not None:
                            _fds_ref._residuals[domain] = residual_data
                        if Z_matrix is not None:
                            _fds_ref._Z_per_domain[domain] = fom.Z_dict
                        if S_matrix is not None:
                            _fds_ref._S_per_domain[domain] = fom.S_dict

                    fom_list.append(fom)
            finally:
                if hasattr(fz, 'close'): fz.close()
                if hasattr(fs_results, 'close'): fs_results.close()
        
        return cls(fom_list, _fds_ref=_fds_ref)

    def __repr__(self) -> str:
        return (f"FOMCollection([{', '.join(f.domain for f in self._foms)}])")

    def get_eigenmodes(self, **kwargs):
        """
        Compute or retrieve eigenmodes for all domains in the collection.
        """
        if self._fds_ref is not None and hasattr(self._fds_ref, 'get_eigenmodes'):
            res = self._fds_ref.get_eigenmodes(domain=None, **kwargs)
            
            # Hierarchical save
            self._auto_save_eigenmodes(res, **kwargs)
            
            return res
        raise RuntimeError("Eigenmodes not available for this FOMCollection.")

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        if self._fds_ref is None or not hasattr(self._fds_ref, '_project_path'):
            return
        
        save_path = Path(self._fds_ref._project_path) / "fds" / "foms" / "eigenmode"
        try:
            if hasattr(self._fds_ref, 'save_eigenmodes'):
                self._fds_ref.save_eigenmodes(save_path, domain=None, **kwargs)
        except Exception as e:
            print(f"Warning: Could not auto-save eigenmodes for collection: {e}")


# =============================================================================
# ROMCollection
# =============================================================================

class ROMCollection(PlotMixin):
    """
    Collection of per-domain reduced-order models.

    Created by :meth:`FOMCollection.reduce`. Each call to ``reduce()``
    produces a new, independent ``ROMCollection``.

    Methods
    -------
    concatenate() -> ConcatenatedSystem
        Concatenate all per-domain ROMs via Kirchhoff coupling.
    """

    def __init__(
        self,
        *,
        _fds_ref=None,
        _mor_ref=None,   # underlying ModelOrderReduction
    ):
        if _mor_ref is None:
            raise ValueError("ROMCollection requires a ModelOrderReduction reference.")
        self._fds_ref = _fds_ref
        self._mor_ref = _mor_ref
        
        # Initialize concatenation cache from MOR if available
        self._concat_cache = getattr(_mor_ref, '_concatenated', None)

    # ------------------------------------------------------------------
    # Sequence interface (delegates to MOR domains)
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        """Access per-domain reduced data by index."""
        domain = self._mor_ref.domains[idx]
        return self._mor_ref.get_reduced_structure(domain)

    def __len__(self) -> int:
        return self._mor_ref.n_domains

    def __iter__(self):
        for domain in self._mor_ref.domains:
            yield self._mor_ref.get_reduced_structure(domain)

    # ------------------------------------------------------------------
    # PlotMixin — aggregate
    # ------------------------------------------------------------------

    @property
    def frequencies(self) -> np.ndarray:
        return self._mor_ref.frequencies if hasattr(self._mor_ref, 'frequencies') and self._mor_ref.frequencies is not None else np.array([])

    @property
    def Z_dict(self) -> Optional[Dict]:
        return self._mor_ref.Z_dict if hasattr(self._mor_ref, 'Z_dict') else None

    @property
    def S_dict(self) -> Optional[Dict]:
        return self._mor_ref.S_dict if hasattr(self._mor_ref, 'S_dict') else None

    # ------------------------------------------------------------------
    # Backward-compatible concat accessor
    # ------------------------------------------------------------------

    @property
    def concat(self):
        """
        Access the cached concatenated system.

        Returns the concatenated system only if it has already been
        computed via ``roms.concatenate()`` or loaded from disk.
        Does **not** trigger concatenation automatically.

        Raises
        ------
        RuntimeError
            If no concatenated system has been computed yet.
        """
        if not hasattr(self, '_concat_cache') or self._concat_cache is None:
            raise RuntimeError(
                "No concatenated system available. "
                "Call roms.concatenate() first."
            )
        return self._concat_cache

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------

    def concatenate(self):
        """
        Concatenate all per-domain ROMs via Kirchhoff coupling.

        Returns
        -------
        ConcatenatedSystem
            Coupled reduced system with ``.solve()`` and ``.reduce()`` methods.
        """
        if self._mor_ref is None:
            raise RuntimeError(
                "ROMCollection has no reference to ModelOrderReduction. "
                "Access via fds.foms.reduce() to get a properly wired collection."
            )
        res = self._mor_ref.concatenate()
        self._concat_cache = res
        return res

    @property
    def mor(self):
        """Access the underlying ModelOrderReduction object."""
        return self._mor_ref

    def get_eigenvalues(self, domain: str = None, **kwargs):
        """Compute eigenvalues from per-domain A_r matrices."""
        return self._mor_ref.get_eigenvalues(domain=domain, **kwargs)

    def get_resonant_frequencies(self, **kwargs):
        return self._mor_ref.get_resonant_frequencies(**kwargs)

    def __repr__(self) -> str:
        domains = ', '.join(self._mor_ref.domains)
        return f"ROMCollection([{domains}])"

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """
        Standardized API for retrieving eigenvalues and eigenvectors for all FOMs.
        """
        if self._mor_ref is not None and hasattr(self._mor_ref, 'get_eigenmodes'):
            res = self._mor_ref.get_eigenmodes(**kwargs)
            
            # Hierarchical save
            if _auto_save:
                self._auto_save_eigenmodes(res, **kwargs)
                
            return res
        raise RuntimeError("Eigenmodes not available for this ROMCollection.")

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        if self._fds_ref is None or not hasattr(self._fds_ref, '_project_path'):
            return
        
        # Mirror: fds/foms/roms -> eigenmode/foms/roms
        save_path = Path(self._fds_ref._project_path) / "eigenmode" / "foms" / "roms"
        try:
            if hasattr(self._mor_ref, 'save_eigenmodes'):
                self._mor_ref.save_eigenmodes(save_path, **kwargs)
        except Exception as e:
            print(f"Warning: Could not auto-save eigenmodes for ROMCollection: {e}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Save ROMCollection (delegates to ModelOrderReduction)."""
        self._mor_ref.save(path)

    @classmethod
    def load(cls, path: Union[str, Path], _fds_ref=None) -> ROMCollection:
        """Load ROMCollection from disk."""
        from rom.reduction import ModelOrderReduction
        mor = ModelOrderReduction.load(path, solver=_fds_ref)
        return cls(_fds_ref=_fds_ref, _mor_ref=mor)


# =============================================================================
# Factory helpers (used by FrequencyDomainSolver to build the wrappers)
# =============================================================================

def build_fom_result(fds, domain: str = 'global') -> FOMResult:
    """
    Build a FOMResult for the given domain (or 'global') from a solved FDS.
    """
    if domain == 'global':
        Z_mat = fds._Z_matrix
        S_mat = fds._S_matrix
        z_dict = fds.Z_dict
        s_dict = fds.S_dict
        ports = fds.ports
        n_ports = fds.n_ports
    else:
        # Per-domain
        z_domain = fds._Z_per_domain.get(domain)
        s_domain = fds._S_per_domain.get(domain)
        domain_ports = fds.domain_port_map.get(domain, [])
        n_modes = fds._n_modes_per_port or 1

        # Build small dense Z/S matrices for this domain
        n_p = len(domain_ports) * n_modes
        n_freq = len(fds.frequencies)
        Z_mat = np.zeros((n_freq, n_p, n_p), dtype=complex) if z_domain else None
        S_mat = np.zeros((n_freq, n_p, n_p), dtype=complex) if s_domain else None

        z_dict = None
        s_dict = None

        if z_domain:
            # Reconstruct matrix from dict keys
            z_dict = {'frequencies': fds.frequencies}
            s_dict = {'frequencies': fds.frequencies} if s_domain else None
            for row in range(n_p):
                prow = row // n_modes + 1
                mrow = row % n_modes + 1
                for col in range(n_p):
                    pcol = col // n_modes + 1
                    mcol = col % n_modes + 1
                    key = f'{pcol}({mcol}){prow}({mrow})'
                    if key in z_domain:
                        Z_mat[:, row, col] = z_domain[key]
                        z_dict[key] = z_domain[key]
                    if s_domain and key in s_domain:
                        S_mat[:, row, col] = s_domain[key]
                        if s_dict is not None:
                            s_dict[key] = s_domain[key]

        ports = domain_ports
        n_ports = len(domain_ports)

    return FOMResult(
        domain=domain,
        frequencies=fds.frequencies,
        Z_matrix=Z_mat,
        S_matrix=S_mat,
        Z_dict=z_dict,
        S_dict=s_dict,
        n_ports=n_ports,
        ports=list(ports),
        n_modes_per_port=fds._n_modes_per_port or 1,
        residual_data=getattr(fds, '_residuals', {}).get(domain),
        _solver_ref=fds,
    )


def build_fom_collection(fds) -> FOMCollection:
    """Build a FOMCollection of per-domain FOMResults from a solved FDS."""
    if not fds.is_compound:
        raise RuntimeError(
            "fds.foms is only available for multi-solid (compound) structures. "
            "For single-solid, use fds.fom."
        )
    if not fds._Z_per_domain:
        raise RuntimeError(
            "No per-domain results found. "
            "Call fds.solve(..., per_domain=True, store_snapshots=True) first."
        )

    fom_list = [build_fom_result(fds, domain=d) for d in fds.domains]
    return FOMCollection(fom_list, _fds_ref=fds)
