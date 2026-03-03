"""
Result wrapper objects for the cavsim3d computation graph.

These lightweight objects wrap solver results and make the computation
graph navigable via attribute access.  Each node in the graph inherits
from PlotMixin so it can plot itself directly.

Graph overview
--------------

Single-solid
  fds.fom             -> FOMResult
  fds.fom.rom         -> ROMResult

Multi-solid
  fds.fom             -> FOMResult          (global / coupled)
  fds.fom.rom         -> ROMResult
  fds.foms            -> FOMCollection      (per-domain list)
  fds.foms[0]         -> FOMResult          (first domain)
  fds.foms.concat     -> ConcatResult       (concatenated FOM)
  fds.foms.concat.rom -> ROMResult
  fds.foms.roms       -> ROMCollection      (per-domain ROMs)
  fds.foms.roms[0]    -> ROMResult
  fds.foms.roms.concat          -> ConcatResult
  fds.foms.roms.concat.rom      -> ROMResult  (chain 1-2-3-4-5 from diagram)
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, Union
import numpy as np

from utils.plot_mixin import PlotMixin


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
        # Back-reference to the FDS and ModelOrderReduction helpers
        _solver_ref=None,
        _rom_factory=None,         # callable() -> ROMResult
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
        self._rom_factory = _rom_factory

        # Lazy cache
        self._rom_cache: Optional[ROMResult] = None

    # ------------------------------------------------------------------
    # PlotMixin requirements
    # ------------------------------------------------------------------

    @property
    def Z_dict(self) -> Optional[Dict]:
        return self._Z_dict

    @property
    def S_dict(self) -> Optional[Dict]:
        return self._S_dict

    # ------------------------------------------------------------------
    # ROM navigation
    # ------------------------------------------------------------------

    @property
    def rom(self) -> ROMResult:
        """
        Reduced-order model of this FOM.

        For single-domain or global results this triggers a
        ``ModelOrderReduction`` on the underlying solver.
        """
        if self._rom_cache is None:
            if self._rom_factory is None:
                raise RuntimeError(
                    "ROM factory not configured for this FOMResult. "
                    "Ensure the FOMResult was created by FrequencyDomainSolver."
                )
            self._rom_cache = self._rom_factory()
        return self._rom_cache

    def _invalidate_rom(self):
        self._rom_cache = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def get_eigenvalues(self, **kwargs):
        """Delegate to solver if available."""
        if self._solver_ref is not None and hasattr(self._solver_ref, 'get_eigenvalues'):
            return self._solver_ref.get_eigenvalues(**kwargs)
        raise RuntimeError("Eigenvalues not available for this FOMResult.")

    def get_resonant_frequencies(self, **kwargs):
        if self._solver_ref is not None and hasattr(self._solver_ref, 'get_resonant_frequencies'):
            return self._solver_ref.get_resonant_frequencies(**kwargs)
        raise RuntimeError("Resonant frequencies not available for this FOMResult.")

    def __repr__(self) -> str:
        n_freq = len(self.frequencies) if self.frequencies is not None else 0
        return (f"FOMResult(domain='{self.domain}', "
                f"n_ports={self.n_ports}, n_freq={n_freq})")


# =============================================================================
# ROMResult
# =============================================================================

class ROMResult(PlotMixin):
    """
    Wrapper around a solved ROM (single domain or global concatenated).

    Attributes
    ----------
    domain : str
    frequencies : np.ndarray  (Hz)
    Z_dict, S_dict : dict
    n_ports : int
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
        _rom_ref=None,           # underlying ModelOrderReduction or ConcatenatedSystem
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
        self._rom_ref = _rom_ref

    # ------------------------------------------------------------------
    # PlotMixin requirements
    # ------------------------------------------------------------------

    @property
    def Z_dict(self) -> Optional[Dict]:
        return self._Z_dict

    @property
    def S_dict(self) -> Optional[Dict]:
        return self._S_dict

    # ------------------------------------------------------------------
    # Eigenvalue access
    # ------------------------------------------------------------------

    def get_eigenvalues(self, **kwargs):
        if self._rom_ref is not None and hasattr(self._rom_ref, 'get_eigenvalues'):
            return self._rom_ref.get_eigenvalues(**kwargs)
        raise RuntimeError("Eigenvalues not available for this ROMResult.")

    def get_resonant_frequencies(self, **kwargs):
        if self._rom_ref is not None and hasattr(self._rom_ref, 'get_resonant_frequencies'):
            return self._rom_ref.get_resonant_frequencies(**kwargs)
        raise RuntimeError("Resonant frequencies not available for this ROMResult.")

    def __repr__(self) -> str:
        n_freq = len(self.frequencies) if self.frequencies is not None else 0
        return (f"ROMResult(domain='{self.domain}', "
                f"n_ports={self.n_ports}, n_freq={n_freq})")


# =============================================================================
# ConcatResult
# =============================================================================

class ConcatResult(PlotMixin):
    """
    Wrapper around a ``ConcatenatedSystem`` (FOM or ROM concatenation).

    Attributes
    ----------
    frequencies : np.ndarray (Hz) — available after .solve() is called internally
    Z_dict, S_dict : dict
    n_ports : int

    Properties
    ----------
    rom : ROMResult
        Further reduces via ``reduce_concatenated_system()`` (lazy, cached).
    """

    def __init__(
        self,
        *,
        concat_system,          # ConcatenatedSystem instance (already coupled+solved)
        frequencies: np.ndarray,
        Z_matrix: Optional[np.ndarray],
        S_matrix: Optional[np.ndarray],
        Z_dict: Optional[Dict],
        S_dict: Optional[Dict],
        n_ports: int,
        ports: List[str],
        n_modes_per_port: int = 1,
        residual_data: Optional[Dict] = None,
    ):
        self._concat_system = concat_system
        self.frequencies = frequencies
        self._Z_matrix = Z_matrix
        self._S_matrix = S_matrix
        self._Z_dict = Z_dict
        self._S_dict = S_dict
        self.n_ports = n_ports
        self.ports = ports
        self._n_modes_per_port = n_modes_per_port
        self._residual_data = residual_data

        self._rom_cache: Optional[ROMResult] = None

    # ------------------------------------------------------------------
    # PlotMixin requirements
    # ------------------------------------------------------------------

    @property
    def Z_dict(self) -> Optional[Dict]:
        return self._Z_dict

    @property
    def S_dict(self) -> Optional[Dict]:
        return self._S_dict

    # ------------------------------------------------------------------
    # ROM navigation
    # ------------------------------------------------------------------

    @property
    def rom(self) -> ROMResult:
        """Further-reduce this concatenated system via POD (lazy, cached)."""
        if self._rom_cache is None:
            self._rom_cache = self._build_rom()
        return self._rom_cache

    def _build_rom(self, tol: float = 1e-6, max_rank: Optional[int] = None) -> ROMResult:
        from solvers.concatenation import reduce_concatenated_system
        reduced = reduce_concatenated_system(
            self._concat_system, tol=tol, max_rank=max_rank
        )
        # Solve the reduced system over the same frequency range
        fmin_ghz = self.frequencies[0] / 1e9
        fmax_ghz = self.frequencies[-1] / 1e9
        nsamples = len(self.frequencies)
        result = reduced.solve(fmin_ghz, fmax_ghz, nsamples)

        return ROMResult(
            domain='concat_rom',
            frequencies=result['frequencies'],
            Z_matrix=result['Z'],
            S_matrix=result['S'],
            Z_dict=result['Z_dict'],
            S_dict=result['S_dict'],
            n_ports=reduced.n_ports,
            ports=list(reduced.ports),
            n_modes_per_port=reduced._n_modes_per_port,
            _rom_ref=reduced,
        )

    def get_eigenvalues(self, **kwargs):
        return self._concat_system.get_eigenvalues(**kwargs)

    def get_resonant_frequencies(self, **kwargs):
        return self._concat_system.get_resonant_frequencies(**kwargs)

    def __repr__(self) -> str:
        n_freq = len(self.frequencies) if self.frequencies is not None else 0
        return (f"ConcatResult(n_ports={self.n_ports}, n_freq={n_freq})")


# =============================================================================
# FOMCollection
# =============================================================================

class FOMCollection(PlotMixin):
    """
    Ordered collection of per-domain :class:`FOMResult` objects.

    Supports indexing (``fds.foms[0]``), iteration, and length.

    Properties
    ----------
    roms : ROMCollection
        Per-domain ROMs (lazy, cached).
    concat : ConcatResult
        Concatenated FOM result (lazy, cached from FDS concatenate method).
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

        # Lazy caches
        self._roms_cache: Optional[ROMCollection] = None
        self._concat_cache: Optional[ConcatResult] = None

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
    # ROM and Concat navigation
    # ------------------------------------------------------------------

    @property
    def roms(self) -> ROMCollection:
        """Per-domain ROMs (lazy, cached)."""
        if self._roms_cache is None:
            self._roms_cache = self._build_roms()
        return self._roms_cache

    @property
    def concat(self) -> ConcatResult:
        """Concatenate the per-domain FOMs via Kirchhoff coupling (lazy, cached)."""
        if self._concat_cache is None:
            self._concat_cache = self._build_concat()
        return self._concat_cache

    def _build_roms(self, tol: float = 1e-6, max_rank: Optional[int] = None) -> ROMCollection:
        """Build a ROMCollection by reducing each domain's FOM.

        Creates a single ModelOrderReduction that covers all domains so
        that ``ROMCollection.concat`` can call ``mor.concatenate()``.
        """
        if self._fds_ref is None:
            raise RuntimeError("FOMCollection has no reference to the FrequencyDomainSolver.")

        from rom.reduction import ModelOrderReduction

        fds = self._fds_ref
        mor = ModelOrderReduction(fds)
        mor.reduce(tol=tol, max_rank=max_rank)

        # Solve per-domain to get per-domain Z/S
        freqs = fds.frequencies
        fmin_ghz = freqs[0] / 1e9
        fmax_ghz = freqs[-1] / 1e9
        nsamples = len(freqs)
        per_domain_results = mor.solve_per_domain(fmin_ghz, fmax_ghz, nsamples)

        rom_results = []
        for domain in fds.domains:
            dr = per_domain_results[domain]
            rom_results.append(ROMResult(
                domain=domain,
                frequencies=dr['frequencies'],
                Z_matrix=dr['Z'],
                S_matrix=dr['S'],
                Z_dict=dr['Z_dict'],
                S_dict=dr['S_dict'],
                n_ports=len(dr['ports']),
                ports=list(dr['ports']),
                n_modes_per_port=fds._n_modes_per_port or 1,
                _rom_ref=mor,
            ))

        return ROMCollection(rom_results, _fds_ref=self._fds_ref, _mor_ref=mor)

    def _build_concat(self) -> ConcatResult:
        """
        Concatenate per-domain FOM Z-matrices via the solver's built-in
        concatenation (global_method='concatenate').
        """
        if self._fds_ref is None:
            raise RuntimeError("FOMCollection has no reference to the FrequencyDomainSolver.")

        fds = self._fds_ref
        # Trigger concatenation solve if not already done
        if fds._Z_global_concatenate is None:
            # Re-run using concatenate method over the same freq range
            freqs = fds.frequencies
            fmin_ghz = freqs[0] / 1e9
            fmax_ghz = freqs[-1] / 1e9
            nsamples = len(freqs)
            fds.solve(fmin_ghz, fmax_ghz, nsamples,
                      per_domain=True, global_method='concatenate',
                      store_snapshots=False)

        return _make_concat_result_from_fds_z(fds)

    def _invalidate(self):
        self._roms_cache = None
        self._concat_cache = None

    def __repr__(self) -> str:
        return (f"FOMCollection([{', '.join(f.domain for f in self._foms)}])")


# =============================================================================
# ROMCollection
# =============================================================================

class ROMCollection(PlotMixin):
    """
    Ordered collection of per-domain :class:`ROMResult` objects.

    Properties
    ----------
    concat : ConcatResult
        Concatenated ROM result (lazy, cached).
    """

    def __init__(
        self,
        rom_list: List[ROMResult],
        *,
        _fds_ref=None,
        _mor_ref=None,   # underlying ModelOrderReduction (if available)
    ):
        if not rom_list:
            raise ValueError("ROMCollection requires at least one ROMResult.")
        self._roms = rom_list
        self._fds_ref = _fds_ref
        self._mor_ref = _mor_ref

        self._concat_cache: Optional[ConcatResult] = None

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> ROMResult:
        return self._roms[idx]

    def __len__(self) -> int:
        return len(self._roms)

    def __iter__(self) -> Iterator[ROMResult]:
        return iter(self._roms)

    # ------------------------------------------------------------------
    # PlotMixin — aggregate
    # ------------------------------------------------------------------

    @property
    def frequencies(self) -> np.ndarray:
        return self._roms[0].frequencies

    @property
    def Z_dict(self) -> Optional[Dict]:
        return self._roms[0].Z_dict

    @property
    def S_dict(self) -> Optional[Dict]:
        return self._roms[0].S_dict

    def plot_s(self, params=None, plot_type='db', ax=None, label=None,
               title=None, show=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax)
        for rom in self._roms:
            lbl = f"{label or ''}{rom.domain}" if label else rom.domain
            fig, ax = rom.plot_s(params=params, plot_type=plot_type, ax=ax,
                                  label=lbl, title=title, **kwargs)
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def plot_z(self, params=None, plot_type='db', ax=None, label=None,
               title=None, show=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax)
        for rom in self._roms:
            lbl = f"{label or ''}{rom.domain}" if label else rom.domain
            fig, ax = rom.plot_z(params=params, plot_type=plot_type, ax=ax,
                                  label=lbl, title=title, **kwargs)
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    def plot_eigenvalues(self, n_modes=30, ax=None, label=None,
                         title=None, show=False, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = self._ensure_ax(ax, figsize=(10, 3))
        for rom in self._roms:
            lbl = f"{label or ''}{rom.domain}" if label else rom.domain
            try:
                fig, ax = rom.plot_eigenvalues(n_modes=n_modes, ax=ax,
                                               label=lbl, **kwargs)
            except RuntimeError:
                pass
        if title:
            ax.set_title(title)
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Concat navigation
    # ------------------------------------------------------------------

    @property
    def concat(self) -> ConcatResult:
        """Concatenate all per-domain ROMs via Kirchhoff coupling (lazy, cached)."""
        if self._concat_cache is None:
            self._concat_cache = self._build_concat()
        return self._concat_cache

    def _build_concat(self) -> ConcatResult:
        """
        Concatenate per-domain ROMs using ModelOrderReduction.concatenate().
        """
        if self._mor_ref is None:
            raise RuntimeError(
                "ROMCollection has no reference to ModelOrderReduction. "
                "Access via fds.foms.roms to get a properly wired collection."
            )
        mor = self._mor_ref
        concat_sys = mor.concatenate()

        # Solve the concatenated system over the same frequency range
        freqs = self._roms[0].frequencies
        fmin_ghz = freqs[0] / 1e9
        fmax_ghz = freqs[-1] / 1e9
        nsamples = len(freqs)
        result = concat_sys.solve(fmin_ghz, fmax_ghz, nsamples)

        return ConcatResult(
            concat_system=concat_sys,
            frequencies=result['frequencies'],
            Z_matrix=result['Z'],
            S_matrix=result['S'],
            Z_dict=result['Z_dict'],
            S_dict=result['S_dict'],
            n_ports=concat_sys.n_ports,
            ports=list(concat_sys.ports),
            n_modes_per_port=concat_sys._n_modes_per_port,
        )

    def _invalidate(self):
        self._concat_cache = None

    def __repr__(self) -> str:
        return (f"ROMCollection([{', '.join(r.domain for r in self._roms)}])")


# =============================================================================
# Helper: build a ConcatResult from the FDS concatenate Z matrices
# =============================================================================

def _make_concat_result_from_fds_z(fds) -> ConcatResult:
    """
    Build a lightweight ConcatResult from FrequencyDomainSolver concatenate results.

    This does NOT require a ConcatenatedSystem object — it wraps the Z/S matrices
    that were computed by fds._build_concatenated_z_matrices().
    """
    if fds._Z_global_concatenate is None:
        raise RuntimeError(
            "No concatenated Z-matrix found. "
            "Call fds.solve(..., global_method='concatenate') first."
        )

    # Build Z_dict / S_dict from the matrices
    Z_mat = fds._Z_global_concatenate
    S_mat = fds._S_global_concatenate

    n_freq, n_p, _ = Z_mat.shape
    n_modes = fds._n_modes_per_port or 1
    Z_dict = {'frequencies': fds.frequencies}
    S_dict = {'frequencies': fds.frequencies} if S_mat is not None else None

    for row in range(n_p):
        port_row = row // n_modes + 1
        mode_row = row % n_modes + 1
        for col in range(n_p):
            port_col = col // n_modes + 1
            mode_col = col % n_modes + 1
            key = f'{port_col}({mode_col}){port_row}({mode_row})'
            Z_dict[key] = Z_mat[:, row, col]
            if S_dict is not None and S_mat is not None:
                S_dict[key] = S_mat[:, row, col]

    # Use a mock concat system that at least supports eigenvalues via the fds
    class _FDSConcatProxy:
        """Minimal proxy exposing ConcatenatedSystem-like interface for FOM concat."""
        def __init__(self, solver):
            self._fds = solver
        def get_eigenvalues(self, **kwargs):
            return self._fds.get_eigenvalues(**kwargs)
        def get_resonant_frequencies(self, **kwargs):
            return self._fds.get_resonant_frequencies(**kwargs)

    return ConcatResult(
        concat_system=_FDSConcatProxy(fds),
        frequencies=fds.frequencies,
        Z_matrix=Z_mat,
        S_matrix=S_mat,
        Z_dict=Z_dict,
        S_dict=S_dict,
        n_ports=n_p // n_modes,
        ports=fds.external_ports,
        n_modes_per_port=n_modes,
        residual_data=getattr(fds, '_residuals', {}).get('global'),
    )


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

    def _rom_factory():
        return _build_rom_result_from_fds(fds, domain=domain)

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
        _rom_factory=_rom_factory,
    )


def _build_rom_result_from_fds(fds, domain: str = 'global') -> ROMResult:
    """
    Build a ROMResult by running ModelOrderReduction on fds.

    For ``domain='global'`` reduces the full coupled mesh.
    For a specific domain, reduces that domain's per-domain matrices.
    """
    from rom.reduction import ModelOrderReduction

    mor = ModelOrderReduction(fds)
    mor.reduce()
    mor.solve(
        fds.frequencies[0] / 1e9,
        fds.frequencies[-1] / 1e9,
        len(fds.frequencies),
    )

    return ROMResult(
        domain=domain,
        frequencies=mor.frequencies,
        Z_matrix=mor._Z_matrix,
        S_matrix=mor._S_matrix,
        Z_dict=mor.Z_dict,
        S_dict=mor.S_dict,
        n_ports=mor.n_ports,
        ports=list(mor.ports),
        n_modes_per_port=mor._n_modes_per_port or 1,
        _rom_ref=mor,
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
