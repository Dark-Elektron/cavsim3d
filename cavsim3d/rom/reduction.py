"""Model Order Reduction using Proper Orthogonal Decomposition."""


from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional, Dict, List, Union, Literal
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cavsim3d.solvers.eigen_mixin import ROMEigenMixin
from cavsim3d.solvers.base import BaseEMSolver, ParameterConverter
from cavsim3d.utils.plot_mixin import PlotMixin
from cavsim3d.rom.structures import ReducedStructure
from ngsolve import GridFunction, Norm, curl, BoundaryFromVolumeCF, HCurl
from cavsim3d.solvers.nedelec import hcurl_flags
from ngsolve.webgui import Draw
from cavsim3d.core.constants import mu0, MIN_EIGENVALUE
from cavsim3d.core.persistence import H5Serializer, ProjectManager
from cavsim3d.solvers.options import (FOM_SOLVE_OPTIONS, REDUCED_SOLVE_OPTIONS,
                                      check_solve_options, reduced_sweep)
import h5py
import json
from pathlib import Path
from datetime import datetime
import cavsim3d.utils.printing as pr
from cavsim3d.utils.printing import read_log
from cavsim3d.utils.threads import small_dense_blas
import warnings
import time
import matplotlib.pyplot as plt
from cavsim3d.geometry.base import _display_webgui_fallback
from cavsim3d.solvers.beam import BeamResultMixin
from cavsim3d.rom import beam_reduction as _brom

if TYPE_CHECKING:
    from cavsim3d.solvers.concatenation import ConcatenatedSystem




def _lossy_reduced_solve(A, C, D, B, omegas, keep_states: bool = True):
    """Solve (A + jwC - w^2 (I - jD)) X = w B at every w; returns (Z, X list).

    A, C, D are the mass-normalised reduced operators (C, D may be None).
    Losses break the single eigendecomposition used for the lossless case,
    so each frequency is a small dense solve.  Z = j B^T X (bilinear: the
    lossy system is complex SYMMETRIC, not Hermitian).  ``keep_states=False``
    returns None for the X list.
    """
    r = A.shape[0]
    I = np.eye(r)
    Cm = np.zeros((r, r)) if C is None else C
    Dm = np.zeros((r, r)) if D is None else D
    Z, X_all = [], []
    for w in omegas:
        lhs = A + 1j * w * Cm - w ** 2 * (I - 1j * Dm)
        X = np.linalg.solve(lhs, w * B)
        if keep_states:
            X_all.append(X)
        Z.append(1j * (B.T @ X))
    return np.array(Z), (X_all if keep_states else None)


def real_snapshot_matrix(snapshots: np.ndarray) -> np.ndarray:
    """A real, Fortran-ordered copy of the snapshots that the POD may overwrite.

    Complex (lossy) snapshots are split into [Re X, Im X]: a real basis keeps
    the projected operators real and the reduced system complex-symmetric,
    exactly like the full-order one.
    """
    X = np.asarray(snapshots)
    if not np.iscomplexobj(X):
        return np.array(X, dtype=float, order='F')
    n, m = X.shape
    out = np.empty((n, 2 * m), order='F')
    out[:, :m] = X.real
    out[:, m:] = X.imag
    return out


def _real_pod_basis(snapshots: np.ndarray, overwrite: bool = False):
    """``(Q, U_R, S)``: the real POD basis of rank r is ``Q @ U_R[:, :r]``.

    The snapshot matrix is tall and thin (n DOFs x a few hundred or thousand
    columns). It is factored X = QR in place, and only the small R is passed
    to the SVD: about one copy of X instead of LAPACK gesdd's several, whose
    workspace allocation can fail on a large model. ``U_R`` is None when X has
    no more rows than columns (Q is then the SVD's own U).

    ``overwrite=True`` lets the factorisation use ``snapshots`` itself when it
    is already real and Fortran-ordered (as :func:`real_snapshot_matrix` and
    the beam's ``pod_snapshots`` return it).

    Raises ``FloatingPointError`` when the snapshots or the singular values
    are not finite, or the snapshots are all zero.
    """
    X = np.asarray(snapshots)
    if not (overwrite and X.dtype == np.float64 and X.flags.f_contiguous):
        X = real_snapshot_matrix(X)
    n, m = X.shape
    where = f"the snapshot matrix ({n} x {m}, {X.nbytes / 1e9:.2f} GB)"
    if not np.isfinite(X).all():
        raise FloatingPointError(f"POD: {where} holds NaN or infinite values; "
                                 "the full-order solve that made them failed.")
    if n > m:
        Q, R = sl.qr(X, mode='economic', overwrite_a=True, check_finite=False)
        U_R, S, _ = np.linalg.svd(R)
    else:
        Q, S, _ = np.linalg.svd(X, full_matrices=False)
        U_R = None
    if not np.isfinite(S).all() or not len(S) or not S[0] > 0:
        raise FloatingPointError(
            f"POD: the SVD of {where} returned "
            + ("no singular values" if not len(S) else
               "an all-zero spectrum" if np.isfinite(S).all() else
               "NaN singular values (LAPACK could not allocate its workspace: "
               "free memory, e.g. by running one reduction at a time)")
            + ". The stored reduced model is left as it was.")
    return Q, U_R, S


def check_reduce_args(tol, max_rank=None) -> None:
    """Reject a truncation tolerance / rank that cannot define a reduced model."""
    if (isinstance(tol, bool) or not isinstance(tol, (int, float, np.number))
            or not np.isfinite(tol) or tol < 0):
        raise ValueError(f"tol must be a finite number >= 0, got {tol!r}")
    if max_rank is not None:
        if (isinstance(max_rank, bool) or not isinstance(max_rank, (int, np.integer))
                or max_rank < 1):
            raise ValueError(
                f"max_rank must be a whole number >= 1 (or None), got {max_rank!r}")


def pod_reduce(K, M, B, snapshots, C=None, D=None, tol: float = 1e-6,
               max_rank: Optional[int] = None, overwrite_snapshots: bool = False) -> Dict:
    """POD reduction of one domain's system ``(K - w^2 M) x = w B u``.

    The snapshots span the basis ``W``; the projected mass matrix is
    normalised to the identity, so the reduced system is
    ``(A_r - w^2 I) x_r = w B_r u`` (with ``C_r``, ``D_r`` for lossy media).
    ``overwrite_snapshots=True`` lets the POD factor a real Fortran-ordered
    snapshot matrix in place (it is destroyed).

    Returns
    -------
    dict
        ``W``, ``S`` (singular values), ``r_pod`` (rank kept by the SVD
        truncation), ``r`` (after dropping directions of ~zero mass),
        ``Q_L_inv``, ``A_r``, ``B_r``, ``C_r``, ``D_r`` (None if lossless) and
        ``n_filtered``.

    Raises
    ------
    FloatingPointError
        If the SVD fails (NaN singular values, e.g. when LAPACK cannot get its
        workspace) or leaves no direction of positive mass.
    """
    check_reduce_args(tol, max_rank)
    Q, U_R, S = _real_pod_basis(snapshots, overwrite=overwrite_snapshots)
    r_pod = max(int(np.sum(S > tol * S[0])), 1)
    if max_rank is not None:
        r_pod = min(r_pod, int(max_rank))
    W = np.ascontiguousarray(Q[:, :r_pod] if U_R is None else Q @ U_R[:, :r_pod])
    del Q

    M_r = W.T @ M @ W
    K_r = W.T @ K @ W
    M_r = (M_r + M_r.T) / 2
    K_r = (K_r + K_r.T) / 2

    # Mass-weighted transformation A_r = L^{-T} K_r L^{-1}; directions of
    # near-zero (or negative) mass are dropped to prevent numerical blow-up.
    lam_all, Q = sl.eigh(M_r)
    min_lam = np.finfo(float).eps * np.max(np.abs(lam_all))
    valid = lam_all > min_lam
    n_filtered = int(np.sum(~valid))
    lam, Q = lam_all[valid], Q[:, valid]
    if not len(lam):
        raise FloatingPointError(
            f"POD: none of the {r_pod} basis vector(s) has a positive mass (projected "
            f"mass eigenvalues {np.array2string(lam_all, precision=2)}); the reduced "
            "model would have 0 DOFs. The stored reduced model is left as it was.")
    Q_L_inv = Q @ np.diag(1.0 / np.sqrt(lam))

    A_r = Q_L_inv.T @ K_r @ Q_L_inv
    loss = {}
    for name, X in (("C_r", C), ("D_r", D)):
        if X is not None:
            X_r = Q_L_inv.T @ (W.T @ (X @ W)) @ Q_L_inv
            loss[name] = (X_r + X_r.T) / 2
    return {
        "W": W, "S": S, "r_pod": r_pod, "r": int(len(lam)), "n_filtered": n_filtered,
        "Q_L_inv": Q_L_inv, "A_r": (A_r + A_r.T) / 2, "B_r": Q_L_inv.T @ W.T @ B,
        "C_r": loss.get("C_r"), "D_r": loss.get("D_r"),
    }


def _same_grid(stored, requested) -> bool:
    """True if a stored frequency grid [Hz] equals the requested one."""
    return (stored is not None and len(stored) == len(requested)
            and np.allclose(stored, requested, rtol=1e-9, atol=0.0))


def _port_geometry_record(port_solver, ports) -> dict:
    """``{port: {center, normal, type, radius, inner_radius, a, b}}`` (JSON-able).

    Saved with a reduced model so a chain can join the ports that face each
    other, and check how many modes propagate at a join.
    """
    out = {}
    for p in ports:
        g = getattr(port_solver, 'port_geometries', {}).get(p)
        if g is None:
            continue
        out[p] = {
            "center": [float(v) for v in g.center],
            "normal": [float(v) for v in g.normal],
            "type": getattr(g.type, 'value', str(g.type)),
            "radius": getattr(g, 'radius', None),
            "inner_radius": getattr(g, 'inner_radius', None),
            "a": getattr(g, 'a', None),
            "b": getattr(g, 'b', None),
        }
    return out


def _port_impedance_record(port_solver, ports) -> Tuple[dict, dict]:
    """``(impedance, fingerprints)`` of ``ports`` from a port solver (JSON-able).

    ``impedance`` holds the analytic parameters a reloaded model rebuilds its
    port impedances from (cutoff, mode type, medium, TEM line impedance);
    ``fingerprints`` the per-mode identity checked at a join.
    """
    ps = port_solver
    imp = {"cutoff": {}, "mtype": {}, "eps": {}, "mu": {}, "zpv": {}}
    fingerprints = {}
    if ps is None:
        return imp, fingerprints
    ck = getattr(ps, 'port_cutoff_kc', {})
    mt = getattr(ps, 'port_mode_types', {})
    mi = getattr(ps, 'port_mode_indices', {})
    mp = getattr(ps, 'port_mode_polarizations', {})
    for p in ports:
        if p in ck:
            imp["cutoff"][p] = {int(m): float(ck[p][m]) for m in ck[p]}
            imp["mtype"][p] = {int(m): str(mt[p][m]) for m in mt[p]}
            fingerprints[p] = {
                int(m): {
                    "kc": float(ck[p][m]),
                    "type": str(mt[p].get(m, "")),
                    "indices": list(mi.get(p, {}).get(m, ())),
                    "pol": float(mp.get(p, {}).get(m, 0.0)),
                } for m in ck[p]}
        imp["eps"][p] = float(ps._port_media_eps_for(p)
                              if hasattr(ps, '_port_media_eps_for') else 1.0)
        imp["mu"][p] = float(ps._port_media_mu_for(p)
                             if hasattr(ps, '_port_media_mu_for') else 1.0)
        zli = (getattr(ps, 'port_line_impedance', {}) or {}).get(p)
        if zli:
            imp["zpv"][p] = {int(m): complex(zli[m]) for m in zli}
        else:
            # Analytic TEM (coax) ports have no entry in port_line_impedance;
            # compute and store it so a RELOADED model keeps the line reference.
            modes = imp["mtype"].get(p, {})
            tem = [m for m, mt_ in modes.items() if str(mt_) == 'TEM']
            try:
                zl = ps.get_port_line_impedance(p, tem[0]) if tem else None
            except Exception:
                zl = None
            if zl is not None and abs(zl) > 1e-9:
                # store a plain float: json cannot encode complex, and a TEM
                # line impedance is real
                imp["zpv"][p] = {int(m): float(np.real(zl))
                                 for m, mt_ in modes.items() if str(mt_) == 'TEM'}
                if not imp["zpv"][p]:
                    imp["zpv"].pop(p, None)
    return imp, fingerprints


def _band_record(frequencies) -> Optional[dict]:
    """Training band ``{fmin_GHz, fmax_GHz, n_snapshots}`` of a sweep [Hz]."""
    if frequencies is None or not len(frequencies):
        return None
    return {
        "fmin_GHz": float(np.min(frequencies)) / 1e9,
        "fmax_GHz": float(np.max(frequencies)) / 1e9,
        "n_snapshots": int(len(frequencies)),
    }


class ModelOrderReduction(BaseEMSolver, ROMEigenMixin, PlotMixin, BeamResultMixin):
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
    DEFAULT_MIN_EIGENVALUE = MIN_EIGENVALUE  # omega^2 of 1 MHz: below is static

    # Version of the saved Z/S results.  solve() reuses saved results only in
    # this format; older ones are re-solved (milliseconds).  2: every port mode
    # resolved on its own (ports with different mode counts; coax TE/TM modes
    # referred to their wave impedance).
    RESULTS_FORMAT = 2

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

        # For non-compound structures (single geometry, possibly multi-material),
        # treat the whole thing as one 'global' domain using global K/M/B/snapshots.
        if not getattr(solver, 'is_compound', False):
            self.domains = ['global']
            self.n_domains = 1
            self._all_ports = solver.all_ports
            self._external_ports = solver.external_ports
            self.domain_port_map = {'global': solver.external_ports}
            self._internal_ports = []
            self._port_domain_adjacency = {}
        else:
            self.domains = solver.domains
            self.n_domains = solver.n_domains
            self._all_ports = solver.all_ports
            self._external_ports = solver.external_ports
            self.domain_port_map = solver.domain_port_map
            self._internal_ports = list(getattr(solver, 'internal_ports', []))
            self._port_domain_adjacency = dict(
                getattr(solver, '_port_domain_adjacency', {}) or {})
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
        # Reference impedance for S/Z normalisation -- TEM/qTEM ports use the
        # line impedance (CST's convention), TE/TM the wave impedance.
        ps = solver.port_solver
        self._port_impedance_func = getattr(
            ps, 'get_port_reference_impedance', ps.get_port_wave_impedance)
        self._port_wave_impedance_func = ps.get_port_wave_impedance

        # Per-domain storage
        self._M: Dict[str, sp.csr_matrix] = {}
        self._K: Dict[str, sp.csr_matrix] = {}
        self._B: Dict[str, np.ndarray] = {}
        self._C: Dict[str, sp.csr_matrix] = {}      # loss (jw) matrices, if lossy
        self._D: Dict[str, sp.csr_matrix] = {}      # loss (jw^2) matrices, if lossy
        self._C_r: Dict[str, np.ndarray] = {}       # mass-normalised reduced C
        self._D_r: Dict[str, np.ndarray] = {}       # mass-normalised reduced D
        self._snapshots: Dict[str, np.ndarray] = {}
        self._band: Optional[dict] = None           # training band, set by reduce()
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

        # Beam (docs/theory/beam_reduction.md): the reduced beam column per
        # domain, the generalised matrices of the last sweep (single domain:
        # self._beam, read by BeamResultMixin; per domain) and the reduced
        # beam columns of that sweep
        self._reduced_beam: Dict[str, '_brom.ReducedBeam'] = {}
        self._beam: Optional[Dict] = None
        self._beam_per_domain: Dict[str, Dict] = {}
        self._beam_snapshots: Dict[str, np.ndarray] = {}

        # Load data from solver
        self._load_from_solver()

        # Track if reduced
        self._is_reduced = False

    def _load_from_solver(self) -> None:
        """Load matrices and snapshots from solver."""
        has_matrices = False

        # Load per-domain data
        for domain in self.domains:
            rom_data = self.solver.get_rom_data(domain)

            if rom_data['M'] is not None:
                self._M[domain] = rom_data['M']
                self._K[domain] = rom_data['K']
                self._B[domain] = rom_data['B']
                if rom_data.get('C') is not None:
                    self._C[domain] = rom_data['C']
                if rom_data.get('D') is not None:
                    self._D[domain] = rom_data['D']
                has_matrices = True

            if rom_data['W'] is not None:
                self._snapshots[domain] = rom_data['W']

        # Check for global snapshots
        if 'global' in self.solver.snapshots:
            global_rom_data = self.solver.get_rom_data('global')
            if global_rom_data['W'] is not None:
                self._snapshots['global'] = global_rom_data['W']

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

    @property
    def _port_mode_order(self):
        """(port, mode) of each column of a single-domain B_r, or None.

        The base class resolves Z/S labels and reference impedances through
        it; without it, it assumes every port carries the same number of
        modes, which mislabels and mis-scales ports with fewer modes.
        """
        if getattr(self, 'n_domains', 1) != 1 or not getattr(self, 'port_modes', None):
            return None
        try:
            return [(p, m) for (_i, p, m) in
                    self._domain_port_mode_pairs(self.domains[0], self._n_modes_per_port or 1)]
        except (KeyError, AttributeError):
            return None

    def _port_wave_impedance(self, port, mode: int, freq: float):
        """Wave impedance the reduced port basis inherited from the FOM."""
        wf = getattr(self, '_port_wave_impedance_func', None)
        return None if wf is None else wf(port, mode, freq)

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

        # by default only the modes near the training band (see
        # EigenMixinBase.TRAINING_BAND_MARGIN); an explicit min_eigenvalue
        # replaces that window
        window = (self._training_window()
                  if filter_static and min_eigenvalue is None else None)

        def process_modes(raw_eigs: np.ndarray, raw_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # Use FrequencyDomainSolver's filter helper
            from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
            eigs, vecs = FrequencyDomainSolver._filter_eigenvalues(
                raw_eigs, raw_vecs, filter_static, min_eigenvalue, None)
            if window is not None:
                keep = (np.real(eigs) >= window[0]) & (np.real(eigs) <= window[1])
                eigs, vecs = eigs[keep], vecs[:, keep]
            if n_modes is not None:
                eigs, vecs = eigs[:n_modes], vecs[:, :n_modes]
            return eigs, vecs

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
            filter_static: bool = True,
            fmax: float = None
    ) -> np.ndarray:
        """
        Get resonant frequencies from eigenvalues.

        A ROM is accurate near the band its snapshots cover; far from it the
        projection leaves spurious modes.  By default only the modes within
        10 % of the training band's edges are listed (``TRAINING_BAND_MARGIN``);
        the mode indices of :meth:`get_eigenmode`, :meth:`get_rq` and
        :meth:`get_figures_of_merit` count the same list.

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
        fmin, fmax : float, optional
            Band in GHz, in place of the training band: ``fmin=0`` lists
            every mode above the static ones.
        filter_static : bool
            If True (default), remove static modes (f ≈ 0).
            When fmin is specified, this is automatically True.

        Returns
        -------
        frequencies : ndarray
            Resonant frequencies in Hz, sorted ascending
        """
        def frequencies_of(modes):
            if isinstance(modes, dict):
                eigs = np.concatenate([v[0] for v in modes.values()])
            else:
                eigs = modes[0]
            eigs = eigs[eigs > 0]           # filtered already; positive for sqrt
            return np.sort(np.sqrt(eigs) / (2 * np.pi))

        explicit = fmin is not None or fmax is not None
        if fmin is not None:
            min_eigenvalue = (2 * np.pi * fmin * 1e9) ** 2
            filter_static = True
        elif explicit:
            min_eigenvalue = self.DEFAULT_MIN_EIGENVALUE if filter_static else None
        else:
            min_eigenvalue = None           # the training-band window applies

        freqs = frequencies_of(self.calculate_resonant_modes(
            domain=domain, source=source, filter_static=filter_static,
            min_eigenvalue=min_eigenvalue,
            n_modes=None  # Don't limit here, do it after freq conversion
        ))
        if fmax is not None:
            freqs = freqs[freqs <= fmax * 1e9]
        window = self._training_window() if filter_static and not explicit else None
        if window is not None:
            n_all = len(frequencies_of(self.calculate_resonant_modes(
                domain=domain, source=source, filter_static=True,
                min_eigenvalue=self.DEFAULT_MIN_EIGENVALUE, n_modes=None)))
            if n_all > len(freqs):
                lo, hi = (np.sqrt(w) / (2e9 * np.pi) for w in window)
                pr.info(f"{n_all - len(freqs)} resonance(s) outside {lo:.4g}-{hi:.4g} GHz "
                        "(the training band and 10 %) not listed; fmin/fmax list them.")

        if n_modes is not None:
            freqs = freqs[:n_modes]

        return freqs

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """
        Compute or retrieve eigenmodes for the reduced structure(s).
        """
        # accept (and ignore) options that only apply to the sparse FOM solve
        for k in ("return_eigenvalues", "sigma"):
            kwargs.pop(k, None)
        res = self.calculate_resonant_modes(**kwargs)
        
        # Hierarchical save
        if _auto_save:
            self._auto_save_eigenmodes(res, **kwargs)
        
        return res

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        try:
            self.save_eigenmodes(**kwargs)
        except (ValueError, Exception) as e:
            pr.warning(f"Could not auto-save eigenmodes for ModelOrderReduction: {e}")

    # =========================================================================
    # Model Reduction
    # =========================================================================

    def _beam_inputs(self, domain: str) -> Optional[Dict]:
        """The full-order beam column of ``domain`` that the reduction needs:
        its field snapshots, its beam data with the interpolation part, the
        (port, mode) of the columns of B and the metadata of its S~.  None
        without a beam; a beam whose sweep kept no field snapshots is reported
        and left out."""
        from cavsim3d.solvers import beam as _bm
        fds = self.solver
        tilde = (getattr(fds, '_beam_tilde', None) or {}).get(domain)
        if not tilde:
            return None
        raw = (getattr(fds, '_beam_raw', None) or {}).get(domain) or {}
        snaps = raw.get('snapshots')
        system = (getattr(fds, '_beam_systems', None) or {}).get(domain)
        data = (_bm.beam_data_of(system)
                if system is not None and system.affine is not None else None)
        # a project reduced read-only (an imported part) writes elsewhere but
        # keeps its files where they are
        root = (getattr(fds, '_read_root', None) or getattr(fds, '_project_path', None))
        if root is not None and (snaps is None or data is None):
            sub = "fom" if self.n_domains == 1 else "foms"
            tag = "global" if self.n_domains == 1 else domain.replace('/', '_')
            base = Path(root) / "fds" / sub
            if snaps is None:
                f = base / "snapshots_beam" / f"snapshots_beam_{tag}.h5"
                if f.exists():
                    with h5py.File(f, "r") as fh:
                        if "field_snapshots" in fh:
                            snaps = H5Serializer.load_dataset(fh["field_snapshots"])
            if data is None:
                data = _bm.load_beam_data(base / "matrices" / f"beam_{tag}.h5")
        if snaps is None or data is None or data.get('affine') is None:
            pr.warning(f"  Beam ({domain}): " + _brom.missing_beam_reason(snaps, data))
            return None
        if data.get('fingerprint') and data['fingerprint'] != tilde.get('fingerprint'):
            pr.warning(f"  Beam ({domain}): the stored beam data belong to another beam "
                       "definition; this reduced model carries no beam. Solve again.")
            return None
        port_modes = tilde.get('port_modes')
        if port_modes is None:
            port_modes = [(p, m) for (_i, p, m) in
                          self._domain_port_mode_pairs(domain, self._n_modes_per_port or 1)]
        return {'snapshots': snaps, 'data': data, 'port_modes': port_modes,
                'meta': {'names': tilde.get('names'), 'ports': tilde.get('ports'),
                         'fingerprints': tilde.get('fingerprints')}}

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
        check_reduce_args(tol, max_rank)
        for rank in (ranks or {}).values():
            check_reduce_args(tol, rank)

        # Start file logging
        _file_handler = None
        if self.solver and hasattr(self.solver, '_project_path') and self.solver._project_path:
            sub_folder = "fom/rom" if self.n_domains == 1 else "foms/roms"
            log_dir = Path(self.solver._project_path) / "fds" / sub_folder
            log_dir.mkdir(parents=True, exist_ok=True)
            self._reduce_log_path = str(log_dir / "reduce.log")
            _file_handler = pr.start_file_log(self._reduce_log_path)

        try:
            pr.running("\n" + "=" * 60)
            pr.running("Model Order Reduction")
            pr.running("=" * 60)

            # Validate snapshots
            self._validate_snapshots_for_reduction()

            _t_reduce_start = time.time()
            total_full = 0
            total_reduced = 0
            # Every domain is reduced before anything is replaced: a domain that
            # fails (pod_reduce raises) leaves the reduced model, in memory and
            # on disk, as it was.
            staged = {}

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

                # With a beam: one basis for the port and the beam columns
                # (docs/theory/beam_reduction.md §10.3)
                beam_in = self._beam_inputs(domain)
                if beam_in is not None:
                    snapshots = _brom.pod_snapshots(snapshots, beam_in['snapshots'],
                                                    beam_in['data']['affine']['free'])

                # POD basis (real, also for complex lossy snapshots) and the
                # mass-normalised reduced operators
                n_snapshots = snapshots.shape[1]
                try:
                    red = pod_reduce(K, M, B, snapshots,
                                     C=self._C.get(domain), D=self._D.get(domain),
                                     tol=tol, max_rank=domain_max_rank,
                                     overwrite_snapshots=beam_in is not None)
                except FloatingPointError as err:
                    raise FloatingPointError(f"Reducing domain '{domain}': {err}") from err
                del snapshots
                reduced_beam = None
                if beam_in is not None:
                    reduced_beam = _brom.reduce_beam(
                        red["W"] @ red["Q_L_inv"], K, M, beam_in['data'],
                        beam_in['port_modes'], C=self._C.get(domain),
                        D=self._D.get(domain), meta=beam_in['meta'])
                    pr.info(f"  Beam: {len(beam_in['data']['affine']['nodes'])} "
                            "interpolation frequencies of its phase")
                staged[domain] = (red, reduced_beam)
                S, r, r_pod = red["S"], red["r"], red["r_pod"]

                pr.info(f"  Full DOFs: {n}")
                pr.info(f"  Snapshots: {n_snapshots}")
                pr.info(f"  Reduced DOFs: {r}")
                pr.info(f"  Compression: {100*(1-r/n):.1f}%")
                pr.debug(f"  Singular value decay: {S[0]:.2e} → {S[min(r_pod, len(S)-1)]:.2e}")
                if red["n_filtered"]:
                    pr.debug(f"  Filtered {red['n_filtered']}/{r_pod} near-zero mass eigenvalue(s)")

                total_full += n
                total_reduced += r

            # results of an earlier reduction no longer apply
            self._beam, self._beam_per_domain, self._beam_snapshots = None, {}, {}
            for domain, (red, reduced_beam) in staged.items():
                self._reduced_beam.pop(domain, None)
                if reduced_beam is not None:
                    self._reduced_beam[domain] = reduced_beam
                self._singular_values[domain] = red["S"]
                self._W[domain] = red["W"]
                self._r[domain] = red["r"]
                self._Q_L_inv[domain] = red["Q_L_inv"]
                self._A_r[domain] = red["A_r"]
                self._B_r[domain] = red["B_r"]
                self._C_r.pop(domain, None)
                self._D_r.pop(domain, None)
                if red["C_r"] is not None:
                    self._C_r[domain] = red["C_r"]
                if red["D_r"] is not None:
                    self._D_r[domain] = red["D_r"]
            del staged

            _t_reduce = time.time() - _t_reduce_start
            pr.done(f"Reduction complete: {total_full} -> {total_reduced} DOFs ({100*(1-total_reduced/total_full):.1f}% compression)")
            pr.info("=" * 60)

            from cavsim3d.utils.timing import get_timing_registry
            get_timing_registry().record(
                "reduction", _t_reduce, category="ROM",
                full_dofs=int(total_full), reduced_dofs=int(total_reduced),
                n_domains=self.n_domains,
            )

            self._is_reduced = True
            # The band the snapshots cover: where the ROM is valid.  Kept apart
            # from self.frequencies, which a later rom.solve() replaces.
            self._band = _band_record(getattr(self.solver, 'frequencies', None))
            self._resonant_mode_cache = {}

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
        finally:
            if _file_handler:
                pr.stop_file_log(_file_handler)

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
            Individual solve parameters:

            - ``frequencies``: any array of frequencies [GHz] in place of
              ``fmin``, ``fmax``, ``nsamples`` (sorted, repeats dropped), e.g.
              points placed on narrow resonances.
            - ``store_snapshots`` (default True): keep the reduced solution of
              every frequency (n_f x r x port modes). With False it is not
              kept or saved; a field at one frequency is then solved again
              when asked for.
            - ``solver_type``, ``rerun``, ``verbose``, ``compute_s_params``.
        """
        # 1. Merge config and kwargs (a full-order config may be reused: its
        # full-order options are accepted and have no effect here)
        cfg = (config or {}).copy()
        cfg.update(kwargs)
        check_solve_options(cfg, REDUCED_SOLVE_OPTIONS, also_accepted=FOM_SOLVE_OPTIONS,
                            where="rom.solve()")

        # 2. The frequencies: a uniform grid or any array
        new_freqs, fmin, fmax, nsamples = reduced_sweep(fmin, fmax, nsamples, cfg, kwargs,
                                                        where="rom.solve()")

        # 3. Extract other options from merged cfg
        solver_type = cfg.get('solver_type', 'auto')
        rerun = cfg.get('rerun', None)   # None: auto, True: force, False: keep stored
        verbose = cfg.get('verbose')     # None: keep the console verbosity
        store_snapshots = bool(cfg.get('store_snapshots', True))

        # Start file logging
        _file_handler = None
        if self.solver and hasattr(self.solver, '_project_path') and self.solver._project_path:
            sub_folder = "fom/rom" if self.n_domains == 1 else "foms/roms"
            log_dir = Path(self.solver._project_path) / "fds" / sub_folder
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = str(log_dir / "solve.log")
            _file_handler = pr.start_file_log(self._log_path)

        _prev_verbosity = pr.push_verbosity(verbose)
        try:
            # Clear results if rerunning
            if rerun:
                self._Z_matrix = None
                self._S_matrix = None
                self._x_r_snapshots = None
                self._invalidate_cache()

            # --- Rerun protection ---
            has_results = (self._Z_matrix is not None)

            # For multi-domain, also check per-domain results
            if not has_results and self.n_domains > 1:
                per_domain = getattr(self, '_per_domain_results', None)
                has_results = per_domain is not None and len(per_domain) > 0

            # Check disk if in-memory is missing
            if not has_results and not rerun and self.solver and getattr(self.solver, '_project_path', None):
                sub_folder = "fom/rom" if self.n_domains == 1 else "foms/roms"
                rom_dir = Path(self.solver._project_path) / "fds" / sub_folder

                if self.n_domains == 1:
                    z_path = rom_dir / "z" / f"z_{self.domains[0]}.h5"
                else:
                    # Multi-domain: check for per-domain files
                    first_domain = self.domains[0].replace('/', '_')
                    z_path = rom_dir / "z" / f"z_{first_domain}.h5"
                try:
                    saved_format = json.loads(
                        (rom_dir / "metadata.json").read_text()).get("results_format")
                except (OSError, ValueError):
                    saved_format = None

                if z_path.exists() and saved_format == self.RESULTS_FORMAT:
                    try:
                        if self.n_domains == 1:
                            with h5py.File(z_path, "r") as f:
                                self._Z_matrix = H5Serializer.load_dataset(f["data"])
                            s_path = rom_dir / "s" / f"s_{self.domains[0]}.h5"
                            if s_path.exists():
                                with h5py.File(s_path, "r") as f:
                                    self._S_matrix = H5Serializer.load_dataset(f["data"])
                        else:
                            # Load per-domain results
                            self._per_domain_results = {}
                            for domain in self.domains:
                                safe_name = domain.replace('/', '_')
                                zp = rom_dir / "z" / f"z_{safe_name}.h5"
                                sp_path = rom_dir / "s" / f"s_{safe_name}.h5"
                                res = {}
                                if zp.exists():
                                    with h5py.File(zp, "r") as f:
                                        res['Z'] = H5Serializer.load_dataset(f["data"])
                                if sp_path.exists():
                                    with h5py.File(sp_path, "r") as f:
                                        res['S'] = H5Serializer.load_dataset(f["data"])
                                if res:
                                    self._per_domain_results[domain] = res

                        # Load frequencies
                        snap_path = rom_dir / "snapshots" / "snapshots.h5"
                        if not snap_path.exists(): snap_path = rom_dir / "snapshots.h5"
                        if snap_path.exists():
                            with h5py.File(snap_path, "r") as f:
                                self.frequencies = H5Serializer.load_dataset(f["frequencies"])

                        has_results = True
                        pr.milestone(f"  Loaded existing ROM results from {rom_dir}")
                    except Exception as e:
                        pr.warning(f"  Could not load existing ROM results: {e}")

            if has_results and not rerun:
                if rerun is False or _same_grid(self.frequencies, new_freqs):
                    pr.milestone("  Returning existing ROM results for this sweep. "
                                 "(Use rerun=True to force a re-solve)")
                    if self.n_domains == 1 and self._reduced_beam and not self._beam:
                        self._single_domain_beam()
                    return self._build_results_dict()
                # A reduced solve costs milliseconds: re-solve rather than hand
                # back results for a different band.
                pr.info("  Requested sweep differs from the stored ROM results; re-solving.")

            if not self._is_reduced:
                raise ValueError("Must call reduce() first")

            pr.running(f"\nROM Solve: {fmin:.4f} - {fmax:.4f} GHz, {nsamples} samples")

            self.frequencies = new_freqs

            _t_rom_solve = time.time()
            if self.n_domains == 1:
                result = self._solve_single_domain(solver_type=solver_type,
                                                   store_snapshots=store_snapshots)
            else:
                result = self._solve_multi_domain(solver_type=solver_type)

            from cavsim3d.utils.timing import get_timing_registry
            get_timing_registry().record(
                "ROM solve", time.time() - _t_rom_solve, category="ROM",
                n_samples=nsamples,
                reduced_dofs=int(sum(self._r.values())) if getattr(self, '_r', None) else None,
            )
            return result
        finally:
            pr.pop_verbosity(_prev_verbosity)
            if _file_handler:
                pr.stop_file_log(_file_handler)

    def print_reduce_log(self) -> None:
        """Print the log file from the last reduce() call, if it exists."""
        log_path = getattr(self, '_reduce_log_path', None)
        if log_path:
            print(read_log(Path(log_path)))
        else:
            print("No reduce log file available. Run reduce() first.")

    def _build_results_dict(self) -> Dict:
        """Build results dictionary for reduced solver."""
        if self.n_domains > 1:
            per_domain = getattr(self, '_per_domain_results', None)
            return {
                'frequencies': self.frequencies,
                'per_domain': per_domain,
                'Z_dict': self.Z_dict,
                'S_dict': self.S_dict,
                'x_r': getattr(self, '_x_r_snapshots', None),
            }
        out = {
            'frequencies': self.frequencies,
            'Z': self._Z_matrix,
            'S': self._S_matrix,
            'Z_dict': self.Z_dict,
            'S_dict': self.S_dict,
            'x_r': getattr(self, '_x_r_snapshots', None),
        }
        if getattr(self, '_beam', None):
            out['Z_tilde'] = self._beam.get('Z_tilde')
            out['S_tilde'] = self._beam.get('S_tilde')
        return out

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Union[str, Path], results_only: bool = False):
        """
        Save ModelOrderReduction data to disk.
        
        Saves reduced matrices (A_r, B_r, W, Q_L_inv) to separate files in matrices/
        and S/Z parameters, snapshots, and eigenmodes to their respective folders.

        ``results_only=True`` writes what a sweep produced -- S/Z, the reduced
        snapshots, the beam's S~/Z~ and the metadata -- and leaves the reduced
        matrices, the structure metadata and the eigenmodes, which change only
        with :meth:`reduce` (a folder without them is saved in full).
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

        # 1. Save reduced and projection matrices to modular files (they
        #    change only with reduce(); a sweep's save leaves them)
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        if results_only and not (mat_path / f"A_r_{self.domains[0]}.h5").exists():
            results_only = False                 # never saved in full: do it now
        if not results_only:
            self._save_matrices(mat_path)

        # 2. Save S and Z results
        if self.n_domains == 1:
            # Single-domain: save as z_<domain>.h5 / s_<domain>.h5
            if self._Z_matrix is not None:
                z_file = f"z_{self.domains[0]}.h5"
                with h5py.File(z_path_dir / z_file, "a") as f:
                    H5Serializer.save_dataset(f, "data", self._Z_matrix)
            if self._S_matrix is not None:
                s_file = f"s_{self.domains[0]}.h5"
                with h5py.File(s_path_dir / s_file, "a") as f:
                    H5Serializer.save_dataset(f, "data", self._S_matrix)
        else:
            # Multi-domain: only per-domain files (no global z.h5/s.h5)
            per_domain = getattr(self, '_per_domain_results', None)
            if per_domain:
                for domain, res in per_domain.items():
                    safe_name = domain.replace('/', '_')
                    if res.get('Z') is not None:
                        with h5py.File(z_path_dir / f"z_{safe_name}.h5", "a") as f:
                            H5Serializer.save_dataset(f, "data", res['Z'])
                    if res.get('S') is not None:
                        with h5py.File(s_path_dir / f"s_{safe_name}.h5", "a") as f:
                            H5Serializer.save_dataset(f, "data", res['S'])

        # 2b. Beam: the reduced beam column of every domain (matrices/), and
        #     S~, Z~ and the reduced beam columns of the last sweep
        self._save_beam(path, results_only=results_only)

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
            elif "x_r_snapshots" in f and self.frequencies is not None:
                del f["x_r_snapshots"]              # an earlier sweep's

        # 4. Save metadata
        metadata = {
            "domains": self.domains,
            "n_domains": self.n_domains,
            "is_reduced": self._is_reduced,
            "r": self._r,
            "n_ports_total": self._n_ports_total,
            "n_ports_external": self._n_ports_external,
            "n_modes_per_port": self._n_modes_per_port,
            # set only when this save wrote results: files left from an
            # earlier reduction are then not taken for this model's
            "results_format": (self.RESULTS_FORMAT
                               if getattr(self, '_Z_matrix', None) is not None
                               or getattr(self, '_per_domain_results', None) else None),
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)

        # 4b. Save per-structure metadata (ports, port-modes, sizes) + analytic
        #     port-impedance parameters, so the reduced model can be rebuilt as
        #     ReducedStructures and concatenated WITHOUT a live solver (import /
        #     reuse across projects).  Requires the solver at save time.
        try:
            solver = getattr(self, 'solver', None)
            if solver is not None and self._is_reduced and not results_only:
                ps = getattr(solver, 'port_solver', None)
                struct_meta = {"structures": []}
                imp = {"cutoff": {}, "mtype": {}, "eps": {}, "mu": {}, "zpv": {}}
                # Per (port, mode) fingerprint for the interface fit-check.
                fingerprints = {}
                for domain in self.domains:
                    if domain not in self._A_r:
                        continue
                    s = self.get_reduced_structure(domain)
                    struct_meta["structures"].append({
                        "domain": domain,
                        "ports": list(s.ports),
                        "port_modes": {p: [int(m) for m in s.port_modes[p]]
                                       for p in s.port_modes},
                        "r": int(s.r), "n_full": int(s.n_full),
                        "is_full_order": bool(s.is_full_order),
                    })
                    if ps is not None:
                        pg = _port_geometry_record(ps, s.ports)
                        if pg:
                            # where each port sits (joins pick facing ports)
                            struct_meta["structures"][-1]["port_geometry"] = pg
                        imp_d, fp_d = _port_impedance_record(ps, s.ports)
                        for key in imp:
                            imp[key].update(imp_d[key])
                        fingerprints.update(fp_d)
                struct_meta["impedance"] = imp
                struct_meta["fingerprints"] = fingerprints
                # Training frequency band (validity window of the ROM): the
                # snapshots' band, not the ROM's own last sweep.
                band = (getattr(self, '_band', None)
                        or _band_record(getattr(solver, 'frequencies', None)))
                if band is not None:
                    struct_meta["band"] = band
                # Serialise BEFORE opening the file. json.dump() streams, so a
                # non-encodable value (e.g. a complex impedance) raises partway
                # through and leaves a truncated file that later fails to load.
                _payload = json.dumps(
                    struct_meta, indent=2,
                    default=lambda o: float(np.real(o))
                    if isinstance(o, complex) else str(o))
                with open(path / "structures.json", "w") as fh:
                    fh.write(_payload)
        except Exception as e:
            warnings.warn(f"Could not save ROM structure metadata: {e}")

        # 5. Save eigenmodes
        if not results_only:
            try:
                self.save_eigenmodes(path=eig_path_dir)
            except Exception as e:
                warnings.warn(f"Could not save ROM eigenmodes to {eig_path_dir}: {e}")

        # 6. Save cached concatenation if available (a sweep's save: only if
        #    it was never saved)
        if self._concatenated is not None and not (
                results_only and (path / "concat" / "metadata.json").exists()):
            self._concatenated.save(path / "concat")

    def _save_matrices(self, mat_path: Path) -> None:
        """The reduced and projection matrices of every domain."""
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
                    for mname, mdict in [("A_r", self._A_r), ("B_r", self._B_r), ("W", self._W),
                                         ("Q_L_inv", self._Q_L_inv), ("C_r", self._C_r),
                                         ("D_r", self._D_r)]:
                        if mdict.get(domain) is None:
                            continue
                        with h5py.File(mat_path / f"{mname}_{domain}.h5", "a") as f_indiv:
                            H5Serializer.save_dataset(f_indiv, "data", mdict.get(domain))

    def _beam_files(self, path: Path, domain: str) -> Dict[str, Path]:
        tag = domain.replace('/', '_')
        return {'beam': path / "matrices" / f"beam_{tag}.h5",
                'Z': path / "z_tilde" / f"z_tilde_{tag}.h5",
                'S': path / "s_tilde" / f"s_tilde_{tag}.h5",
                'snapshots': path / "snapshots_beam" / f"snapshots_beam_{tag}.h5"}

    def _save_beam(self, path: Path, results_only: bool = False) -> None:
        """Write (or, without a beam, remove) the beam files of every domain;
        ``results_only``: not the reduced beam columns (they change with
        reduce() only)."""
        from cavsim3d.solvers import beam as _bm
        for domain in self.domains:
            files = self._beam_files(path, domain)
            rb = getattr(self, '_reduced_beam', {}).get(domain)
            tilde = (getattr(self, '_beam', None) if self.n_domains == 1
                     else getattr(self, '_beam_per_domain', {}).get(domain))
            if rb is not None and not (results_only and files['beam'].exists()):
                rb.save(files['beam'])
            if tilde and rb is not None:
                _bm.save_tilde(files['Z'], tilde, 'Z')
                _bm.save_tilde(files['S'], tilde, 'S')
                yb = getattr(self, '_beam_snapshots', {}).get(domain)
                if yb is not None:
                    files['snapshots'].parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(files['snapshots'], "w") as f:
                        H5Serializer.save_dataset(f, "frequencies", np.asarray(tilde['frequencies']))
                        H5Serializer.save_dataset(f, "reduced_snapshots", np.asarray(yb))
                        f.attrs["fingerprint"] = str(tilde.get('fingerprint', ''))
                continue
            for key, f in files.items():
                if f.exists() and (key != 'beam' or rb is None):
                    f.unlink()
        for folder in ("z_tilde", "s_tilde", "snapshots_beam"):
            d = path / folder
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    def _load_beam(self, path: Path) -> None:
        """Read the beam files :meth:`_save_beam` wrote."""
        from cavsim3d.solvers import beam as _bm
        self._reduced_beam, self._beam, self._beam_per_domain = {}, None, {}
        self._beam_snapshots = {}
        for domain in self.domains:
            files = self._beam_files(path, domain)
            rb = _brom.ReducedBeam.load(files['beam'])
            if rb is None:
                continue
            self._reduced_beam[domain] = rb
            z, s_ = _bm.load_tilde(files['Z']), _bm.load_tilde(files['S'])
            meta = z or s_
            if meta is None:
                continue
            tilde = {'Z_tilde': z['data'] if z else None, 'S_tilde': s_['data'] if s_ else None,
                     'rows': meta['rows'], 'cols': meta['cols'],
                     'frequencies': meta['frequencies'], 'names': meta['names'],
                     'setup': meta['setup'], 'fingerprint': meta['fingerprint'],
                     'summary': meta['summary'], 'port_modes': meta.get('port_modes'),
                     'ports': meta.get('ports'), 'fingerprints': meta.get('fingerprints'),
                     'zref': s_.get('zref') if s_ else None}
            if self.n_domains == 1:
                self._beam = tilde
            else:
                self._beam_per_domain[domain] = tilde

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
        # training band, as save() recorded it (older saves: none)
        rom._band = None
        if (path / "structures.json").exists():
            try:
                rom._band = json.loads((path / "structures.json").read_text()).get("band")
            except (OSError, ValueError):
                pass

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
            # as in __init__: a single (non-compound) structure is ONE
            # 'global' domain, whatever its mesh materials are called
            if getattr(solver, 'is_compound', False):
                rom.domain_port_map = solver.domain_port_map
            else:
                rom.domain_port_map = {'global': solver.external_ports}
            rom._port_impedance_func = solver._get_port_impedance
            ps = getattr(solver, 'port_solver', None)
            if ps is not None:
                rom._port_wave_impedance_func = ps.get_port_wave_impedance
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
        rom._C, rom._D = {}, {}
        rom._C_r, rom._D_r = {}, {}
        rom._singular_values = {}

        # 1. Load matrices from modular files or legacy matrices.h5
        mat_path = path / "matrices"
        if mat_path.exists():
             for mname in ["A_r", "B_r", "W", "Q_L_inv", "C_r", "D_r"]:
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

        # the global aliases reduce() sets: one domain stands for the model
        rom._A_r_global = rom._B_r_global = rom._W_r_global = rom._r_global = None
        if rom.n_domains == 1 and rom.domains[0] in rom._A_r:
            d = rom.domains[0]
            rom._A_r_global, rom._B_r_global = rom._A_r[d], rom._B_r.get(d)
            rom._W_r_global, rom._r_global = rom._W.get(d), rom._r.get(d)

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

        # 3b. Beam: the reduced beam columns and the last sweep's S~ / Z~
        rom._load_beam(path)

        # 4. Load eigenmodes
        rom.load_eigenmodes()

        # 5. Load concatenated system if it exists
        concat_path = path / "concat"
        rom._concatenated = None
        if concat_path.exists():
            try:
                from cavsim3d.solvers.concatenation import ConcatenatedSystem
                rom._concatenated = ConcatenatedSystem.load(concat_path, solver_ref=solver)
            except Exception as e:
                pr.warning(f"Could not load concatenated system from {concat_path}: {e}")

        # Restore log paths if they exist on disk
        if solver and hasattr(solver, '_project_path') and solver._project_path:
            sub_folder = "fom/rom" if rom.n_domains == 1 else "foms/roms"
            log_dir = Path(solver._project_path) / "fds" / sub_folder
            solve_log = log_dir / "solve.log"
            reduce_log = log_dir / "reduce.log"
            if solve_log.exists():
                rom._log_path = str(solve_log)
            if reduce_log.exists():
                rom._reduce_log_path = str(reduce_log)

        return rom

    def _solve_single_domain(self, solver_type: str = 'auto',
                             store_snapshots: bool = True) -> Dict:
        """Solve single-domain reduced system; ``store_snapshots=False`` keeps
        no reduced solutions (fields are then solved again on request)."""
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

        t0 = time.time()

        omegas = 2 * np.pi * self.frequencies  # (n_freq,)

        C_r = self._C_r.get(domain)
        D_r = self._D_r.get(domain)
        if C_r is not None or D_r is not None:
            # Lossy: no common eigenbasis -> one small dense solve per frequency
            self._Z_matrix, x_r_all = _lossy_reduced_solve(
                A_r, C_r, D_r, B_r, omegas, keep_states=store_snapshots)
        elif solver_type in ('auto', 'direct'):
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

            # Snapshots: x_r[k] = w V diag(d[k]) V^{-1} B
            x_r_all = ([omegas[k] * V @ (d[k, :, None] * Vinv_B)
                        for k in range(n_freq)] if store_snapshots else None)

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

                if store_snapshots:
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
        self._x_r_snapshots = {domain: np.array(x_r_all)} if store_snapshots else None

        self._compute_s_from_z()
        self._invalidate_cache()

        self._single_domain_beam()

        # Automatic save after simulation: this sweep's results only
        self._autosave_results()

        return self._build_results_dict()

    def _results_dir(self) -> Optional[Path]:
        """This reduced model's folder in the project (None without one)."""
        root = getattr(getattr(self, 'solver', None), '_project_path', None)
        if root is None:
            return None
        return Path(root) / "fds" / ("fom/rom" if self.n_domains == 1 else "foms/roms")

    def _autosave_results(self) -> None:
        """Write a sweep's results into this reduced model's folder, and the
        timing analysis -- not the whole project: the full-order results and
        the reduced matrices did not change.  Only the reduced model the
        project holds is written (as ``fds.fom.rom`` / ``fds.foms.roms``)."""
        fds = getattr(self, 'solver', None)
        ref = getattr(fds, '_project_ref', None)
        path = self._results_dir()
        if ref is None or path is None or getattr(ref, '_read_only', False):
            return
        if self.n_domains == 1:
            held = getattr(getattr(fds, '_fom_cache', None), '_rom_cache', None)
        else:
            roms = getattr(getattr(fds, '_foms_cache', None), '_roms_cache', None)
            held = getattr(roms, '_mor_ref', None)
        if held is not self:
            return
        self.save(path, results_only=True)
        ref.save_timing()

    def _single_domain_beam(self) -> None:
        """S~ and Z~ of the single domain at the sweep's frequencies, from its
        reduced beam column (docs/theory/beam_reduction.md §10.5)."""
        domain = self.domains[0]
        self._beam, self._beam_snapshots = None, {}
        rb = self._reduced_beam.get(domain)
        if rb is None or self.frequencies is None or domain not in self._A_r:
            return
        t0 = time.time()
        tilde = _brom.section_tilde(rb, self.frequencies, self._A_r[domain],
                                    self._B_r[domain], self._C_r.get(domain),
                                    self._D_r.get(domain), zref=self._get_port_impedance,
                                    zwave=self._port_wave_impedance)
        self._beam_snapshots = {domain: tilde.pop('y_b')}
        self._beam = tilde
        pr.done(f"  Beam columns: {time.time() - t0:.3f}s")

    def _solve_multi_domain(self, **kwargs) -> Dict:
        """Solve multi-domain system: per-domain S/Z only."""
        # Solve per-domain (individual S/Z for each domain)
        per_domain_results = self.solve_per_domain()
        self._per_domain_results = per_domain_results

        # Build concatenation for eigenmode reconstruction (but don't solve globally)
        self.concatenate()

        self._invalidate_cache()

        return per_domain_results

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

        struct = ReducedStructure(
            Ard=self._A_r[domain],
            Brd=self._B_r[domain],
            ports=domain_ports,
            port_modes=domain_port_modes,
            domain=domain,
            r=self._r[domain],
            # a reloaded ROM holds W (n_full x r) but not the full-order M
            n_full=(self._M[domain] if domain in self._M else self._W[domain]).shape[0],
            W=self._W[domain],
            Q_L_inv=self._Q_L_inv[domain],
            fes=fes,
            mesh=mesh,
            Crd=self._C_r.get(domain),
            Drd=self._D_r.get(domain),
        )
        # Attach interface fit-check metadata (per-mode fingerprint + training
        # band) so both live and imported structures validate the same way.
        ps = getattr(self.solver, 'port_solver', None)
        if ps is not None:
            ck = getattr(ps, 'port_cutoff_kc', {})
            mt = getattr(ps, 'port_mode_types', {})
            mi = getattr(ps, 'port_mode_indices', {})
            mp = getattr(ps, 'port_mode_polarizations', {})
            fp = {}
            for p in domain_ports:
                if p in ck:
                    fp[p] = {int(m): {
                        "kc": float(ck[p][m]), "type": str(mt[p].get(m, "")),
                        "indices": list(mi.get(p, {}).get(m, ())),
                        "pol": float(mp.get(p, {}).get(m, 0.0)),
                    } for m in ck[p]}
            struct.port_fingerprints = fp
            struct.port_geometry = _port_geometry_record(ps, domain_ports) or None
        band = (getattr(self, '_band', None)
                or _band_record(getattr(self.solver, 'frequencies', None)))
        if band is not None:
            struct.training_band = band
        return struct

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

        if not self._is_reduced:
            raise ValueError("Must call reduce() first")

        if others is None and self.n_domains > 1:
            structures = self.get_all_structures()
            connections = self._build_connections()
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
        from cavsim3d.solvers.concatenation import ConcatenatedSystem
        concat = ConcatenatedSystem(
            structures=structures,
            mesh=self.mesh,
            port_impedance_func=self._port_impedance_func,
            port_wave_impedance_func=getattr(
                self, '_port_wave_impedance_func', None),
            solver_ref=self,
        )
        concat.define_connections(connections)
        concat.couple()
        if others is None and self._reduced_beam and all(
                d in self._reduced_beam for d in self.domains):
            # the domains' S~ joined at every solve of the coupled system
            # (one beam frame: the parts are glued); docs/theory/beam_reduction.md §10.7
            from cavsim3d.solvers.beam import BeamSetup
            wave = getattr(self, '_port_wave_impedance_func', None)
            sections = [{'beam': self._reduced_beam[d], 'A': self._A_r[d], 'B': self._B_r[d],
                         'C': self._C_r.get(d), 'D': self._D_r.get(d),
                         'zref': self._get_port_impedance, 'zwave': wave, 'key': d}
                        for d in self.domains]
            setup = BeamSetup.from_dict(self._reduced_beam[self.domains[0]].setup)
            concat._beam_join = _brom.ReducedBeamJoin(
                concat, sections, setup, shifts=None,
                summary={'joined': list(self.domains)})

        self._concatenated = concat
        self._A_r_global = concat.A_coupled
        self._B_r_global = concat.B_coupled
        self._W_r_global = concat.W_coupled
        self._r_global = concat.A_coupled.shape[0]

        return concat

    def _build_connections(self) -> List[Tuple]:
        """Build concatenation connections from shared interface ports.

        Each internal port is shared by two (or more) domains; for every such
        port the corresponding domain structures are connected *at that port*.
        This handles arbitrary multiport topologies — a domain may carry any
        number of external ports plus its interface port(s) — rather than
        assuming a linear chain where each domain has exactly two ports.

        Falls back to the legacy sequential chain only when no adjacency
        information is available.
        """
        adj = getattr(self, '_port_domain_adjacency', {}) or {}
        internal = getattr(self, '_internal_ports', []) or []
        domain_index = {d: i for i, d in enumerate(self.domains)}

        connections: List[Tuple] = []
        for port in internal:
            doms = sorted(
                (d for d in adj.get(port, set()) if d in domain_index),
                key=lambda d: domain_index[d],
            )
            # Couple each further domain sharing this port back to the first,
            # so a port shared by N domains yields N-1 constraints.
            for other in doms[1:]:
                connections.append((
                    (domain_index[doms[0]], port),
                    (domain_index[other], port),
                ))

        if connections or adj:
            return connections

        # Legacy fallback: sequential chain (each domain's last port to the
        # next domain's first port).  Used only without adjacency data.
        for i in range(self.n_domains - 1):
            ports_i = self.domain_port_map[self.domains[i]]
            ports_next = self.domain_port_map[self.domains[i + 1]]
            connections.append((
                (i, ports_i[-1]),
                (i + 1, ports_next[0]),
            ))
        return connections

    # Backwards-compatible alias
    _build_sequential_connections = _build_connections

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

        # The reduced matrices are tiny; MKL's multithreaded LAPACK is roughly
        # 900x SLOWER than single-threaded at this size (thread setup dominates).
        with small_dense_blas():
            return self._solve_per_domain_inner(frequencies, n_modes, results)

    def _domain_port_mode_pairs(self, domain: str, n_modes: int):
        """Ordered ``(local_port_idx, port, mode)`` matching B_r's columns."""
        pairs = []
        for pidx, p in enumerate(self.domain_port_map[domain]):
            modes = (sorted(self.port_modes[p]) if self.port_modes and p in self.port_modes
                     else range(n_modes))
            pairs.extend((pidx, p, m) for m in modes)
        return pairs

    def _solve_per_domain_inner(self, frequencies, n_modes, results):
        for domain in self.domains:
            A_r = self._A_r[domain]
            B_r = self._B_r[domain]
            domain_ports = self.domain_port_map[domain]

            # B_r columns = n_ports_domain * n_modes  (all port-mode combos)
            n_pm = B_r.shape[1]

            n_freq = len(frequencies)
            Z_d = np.zeros((n_freq, n_pm, n_pm), dtype=complex)
            S_d = np.zeros((n_freq, n_pm, n_pm), dtype=complex)

            I_exc = np.eye(n_pm)
            # (port, mode) of each B_r column: ports in order, each with its
            # own sorted modes (ports may carry different mode counts).
            pm_order = self._domain_port_mode_pairs(domain, n_modes)
            if len(pm_order) != n_pm:
                raise ValueError(
                    f"Domain '{domain}': B_r has {n_pm} columns but its ports "
                    f"carry {len(pm_order)} modes {pm_order}.")

            # A_r is small (r ~ 100), so diagonalise ONCE and evaluate every
            # frequency with a diagonal solve:
            #     (A - w^2 I)^-1 = V diag(1/(lam - w^2)) V^-1
            # Re-factorising per frequency cost 1500 x r^3 and, because MKL's
            # threaded path is ~900x slower than single-threaded at this size,
            # turned a sub-second sweep into ~7 minutes.
            C_r = self._C_r.get(domain)
            D_r = self._D_r.get(domain)
            lossy = C_r is not None or D_r is not None
            if lossy:
                Z_lossy, _ = _lossy_reduced_solve(
                    A_r, C_r, D_r, B_r, 2 * np.pi * np.asarray(frequencies),
                    keep_states=False)
            elif np.allclose(A_r, A_r.T.conj(), atol=1e-10):
                lam, V = np.linalg.eigh(A_r)
                Vinv_B = V.T.conj() @ (B_r @ I_exc)
            else:
                lam, V = np.linalg.eig(A_r)
                Vinv_B = np.linalg.solve(V, B_r @ I_exc)

            for k, freq in enumerate(frequencies):
                omega = 2 * np.pi * freq

                if lossy:
                    Z_d[k] = Z_lossy[k]
                else:
                    # x_r = w V diag(1/(lam - w^2)) V^-1 B
                    x_r = omega * (V @ (Vinv_B / (lam - omega ** 2)[:, None]))
                    Z_d[k] = 1j * B_r.T @ x_r

                # Reference impedances -- complex, exactly as the FOM path uses
                # them.  Taking the real part zeroed the (purely reactive)
                # reference of every evanescent TE/TM mode.
                Z0_mat = np.diag([self._get_port_impedance(p, m, freq)
                                  for (_pi, p, m) in pm_order])
                # The reduced Z inherits the FOM's WAVE-impedance
                # normalisation. Where the reported reference differs (TEM
                # ports report the line impedance), rescale Z with it:
                # z_to_s(a*Z, a*Z0) == z_to_s(Z, Z0), so S is unchanged while
                # Z becomes physical ohms -- matching the FOM path.
                wf = getattr(self, '_port_wave_impedance_func', None)
                if wf is not None:
                    sc = []
                    for (_pi, p, m) in pm_order:
                        try:
                            zw = abs(wf(p, m, freq))
                            zt = abs(self._get_port_impedance(p, m, freq))
                            sc.append(zt / zw if zw > 1e-12 else 1.0)
                        except Exception:
                            sc.append(1.0)
                    sc = np.asarray(sc, dtype=float)
                    if not np.allclose(sc, 1.0):
                        Z_d[k] = Z_d[k] * np.sqrt(np.outer(sc, sc))
                S_d[k] = ParameterConverter.z_to_s(Z_d[k], Z0_mat)

            # Dicts keyed '<excitation>(mode)<response>(mode)' (column first),
            # the convention of every other result object.
            Z_dict = {}
            S_dict = {}
            for i, (pi, _p, mi) in enumerate(pm_order):
                for j, (pj, _q, mj) in enumerate(pm_order):
                    key = f'{pj + 1}({mj + 1}){pi + 1}({mi + 1})'
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

            # The domain's beam column (docs/theory/beam_reduction.md §10.5)
            self._beam_per_domain.pop(domain, None)
            rb = self._reduced_beam.get(domain)
            if rb is not None:
                tilde = _brom.section_tilde(
                    rb, frequencies, A_r, B_r, C_r, D_r, zref=self._get_port_impedance,
                    zwave=getattr(self, '_port_wave_impedance_func', None))
                self._beam_snapshots[domain] = tilde.pop('y_b')
                self._beam_per_domain[domain] = tilde

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
            fes = HCurl(self.mesh, order=order, complex=True, dirichlet=bc,
                        **hcurl_flags(getattr(self.solver, 'nedelec', 'second')))
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

        # Single domain case: the stored reduced solution, or solved again
        states = self._reduced_states(domain, freq_idx)

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

        # states shape: (r, n_port_modes)
        if col_idx >= states.shape[1]:
            raise ValueError(
                f"Excitation column {col_idx} out of range. "
                f"port_idx={port_idx}, mode={excitation_mode}"
            )

        return states[:, col_idx]

    def _reduced_states(self, domain: str, freq_idx: int) -> np.ndarray:
        """The reduced solution (r x port modes) of every port-mode excitation
        at ``frequencies[freq_idx]``: the stored one, or solved again when the
        sweep kept none (``solve(store_snapshots=False)``)."""
        n_f = 0 if self.frequencies is None else len(self.frequencies)
        if not n_f:
            raise ValueError("No reduced solution: call solve() first.")
        if not 0 <= freq_idx < n_f:
            raise ValueError(f"freq_idx {freq_idx} out of range [0, {n_f - 1}]")
        stored = (self._x_r_snapshots or {}).get(domain)
        if stored is not None and len(stored) == n_f:
            return stored[freq_idx]
        w = 2 * np.pi * float(self.frequencies[freq_idx])
        _, X = _lossy_reduced_solve(self._A_r[domain], self._C_r.get(domain),
                                    self._D_r.get(domain), self._B_r[domain], [w])
        return X[0]

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

        # The space is complex exactly when the model is lossy.  A lossless
        # model's solution is real (the ports are driven on open circuits),
        # so a real space holds it; complex data in a real space would lose
        # the phase, so that is an error, not a silent cast.
        E_gf = GridFunction(fes)
        if not fes.is_complex:
            if np.abs(np.imag(x_full)).max() > 1e-12 * max(np.abs(x_full).max(), 1e-300):
                raise RuntimeError(
                    f"Complex field on the real FE space of '{domain}': "
                    "the imaginary part would be lost.")
            x_full = np.real(x_full)
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

        # a solution from solve(): stored, or solved again on request
        return self._x_r_snapshots is not None or self.frequencies is not None

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
            # Faraday, e^{+jwt}: curl E = -j w mu0 H  =>  H = j curl(E) / (w mu0)
            field_cf = (1j / (omega * mu0)) * curl(E_gf)
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

        _display_webgui_fallback(Draw(BoundaryFromVolumeCF(cf_plot), self.mesh, plot_name, **draw_kwargs))
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
                pr.debug("\nGlobal system:")
                pr.debug(f"  Global reduced DOFs: {self._r_global}")
                pr.debug(f"  A_global shape: {self._A_r_global.shape}")
                pr.debug(f"  B_global shape: {self._B_r_global.shape}")
                if self._concatenated is not None:
                    pr.debug(f"  External ports: {self._concatenated.ports}")

            pr.debug(f"\nField reconstruction: {'Available' if self.can_reconstruct() else 'Not available'}")
        else:
            pr.info("\nNot yet reduced. Call reduce() first.")

        if self.frequencies is not None:
            pr.info("\nSolution available:")
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


# =============================================================================
# Standalone reduced-structure loading (import / reuse across projects)
# =============================================================================

def _load_matrix(fpath: Path):
    """Read a single-dataset matrix file ('data') written by ROM save."""
    with h5py.File(fpath, "r") as f:
        return H5Serializer.load_dataset(f["data"])


def load_reduced_structures(rom_dir, fes=None, mesh=None):
    """Rebuild a saved ROM's :class:`ReducedStructure` list WITHOUT a live solver.

    Reads the reduced operators (``matrices/{A_r,B_r,W,Q_L_inv}_{domain}.h5``)
    plus the ``structures.json`` metadata written by
    :meth:`ModelOrderReduction.save`, so a previously-run project's reduced
    model can be imported and concatenated (or further reduced).

    Parameters
    ----------
    rom_dir : path-like
        A saved ROM directory (e.g. ``<project>/fds/foms/roms``).
    fes, mesh : optional
        Attach an FE space / mesh to the structures (needed only for field
        reconstruction; not required for S-/Z-/eigenvalue concatenation).

    Returns
    -------
    (structures, impedance_func) : (list[ReducedStructure], callable | None)
        The reduced structures and a standalone port wave-impedance function
        rebuilt from the persisted analytic parameters (``None`` if absent).
    """
    from cavsim3d.solvers.ports import (make_analytic_port_impedance,
                                        make_analytic_port_wave_impedance)

    rom_dir = Path(rom_dir)
    meta_file = rom_dir / "structures.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"No structures.json in {rom_dir}. This ROM was saved without "
            "standalone structure metadata (re-run reduce()/save with a solver)."
        )
    with open(meta_file) as fh:
        meta = json.load(fh)

    mat = rom_dir / "matrices"
    # Fingerprints/band/impedance may be stored at the TOP level (multi-solid:
    # globally-unique port names) OR PER STRUCTURE (netlist: each section has
    # its own port1/port2, which would collide in a shared dict).  Prefer the
    # per-structure entry, fall back to the top level.
    top_fingerprints = meta.get("fingerprints", {})
    top_band = meta.get("band")
    top_imp = meta.get("impedance")
    structures = []
    impedance_func = make_analytic_port_impedance(top_imp) if top_imp else None
    wave_func = (make_analytic_port_wave_impedance(top_imp)
                 if top_imp else None)
    for sm in meta["structures"]:
        d = sm["domain"]
        mesh_source = None
        if sm.get("source_rom_dir"):
            # A REFERENCED section: its matrices stay in the source project,
            # under the source's own domain name.
            src_rom = Path(sm["source_rom_dir"])
            if not src_rom.is_absolute():
                src_rom = (rom_dir / src_rom).resolve()
            if not src_rom.exists():
                raise FileNotFoundError(
                    f"Section '{d}' is a reference to {src_rom}, which no longer "
                    "exists. Restore the source project, or re-import it.")
            smat, sd = src_rom / "matrices", sm.get("source_domain", d)

            def _mf(base, smat=smat, sd=sd):
                for name in (f"{base}_{sd}.h5", f"{base}.h5"):
                    if (smat / name).exists():
                        return smat / name
                return smat / f"{base}_{sd}.h5"
            if sm.get("source_mesh_dir"):
                ms = Path(sm["source_mesh_dir"])
                mesh_source = ms if ms.is_absolute() else (rom_dir / ms).resolve()
        else:
            def _mf(base, d=d):
                return mat / f"{base}_{d}.h5"
        W = None
        Q = None
        wf = _mf("W")
        qf = _mf("Q_L_inv")
        if wf.exists():
            W = _load_matrix(wf)
        if qf.exists():
            Q = _load_matrix(qf)
        cf_, df_ = _mf("C_r"), _mf("D_r")
        C_r = _load_matrix(cf_) if cf_.exists() else None
        D_r = _load_matrix(df_) if df_.exists() else None
        port_modes = {p: {int(m): None for m in sm["port_modes"][p]}
                      for p in sm["port_modes"]}
        struct = ReducedStructure(
            Ard=_load_matrix(_mf("A_r")),
            Brd=_load_matrix(_mf("B_r")),
            ports=list(sm["ports"]), port_modes=port_modes, domain=d,
            r=sm["r"], n_full=sm["n_full"],
            is_full_order=sm.get("is_full_order", False),
            W=W, Q_L_inv=Q, fes=fes, mesh=mesh, Crd=C_r, Drd=D_r,
        )
        # Interface fit-check metadata: per (port, mode) fingerprint {kc,type,
        # indices,pol} keyed by int mode, and the ROM training band.
        fingerprints = sm.get("fingerprints", top_fingerprints)
        struct.port_fingerprints = {
            p: {int(m): fp for m, fp in fingerprints[p].items()}
            for p in sm["ports"] if p in fingerprints}
        struct.training_band = sm.get("band", top_band)
        struct.port_geometry = sm.get("port_geometry")
        struct.mesh_source = mesh_source      # source project's mesh/ (references)
        # Per-section impedance (mixed-origin netlist) or shared top-level one.
        sm_imp = sm.get("impedance")
        media = sm_imp or top_imp or {}
        struct.port_media = {p: {"eps": float(media.get("eps", {}).get(p, 1.0)),
                                 "mu": float(media.get("mu", {}).get(p, 1.0))}
                             for p in sm["ports"]}
        struct.impedance_func = (make_analytic_port_impedance(sm_imp)
                                 if sm_imp else impedance_func)
        struct.wave_impedance_func = (make_analytic_port_wave_impedance(sm_imp)
                                      if sm_imp else wave_func)
        # the section's reduced beam column (None: reduced without a beam)
        struct.reduced_beam = _brom.ReducedBeam.load(_mf("beam"))
        structures.append(struct)

    return structures, impedance_func


def import_reduced_structures(project_path, fes=None, mesh=None):
    """Import a previously-run PROJECT's reduced structures (+ impedance).

    Locates the saved ROM inside a project directory (searching the usual
    ``fds/foms/roms`` / ``fds/fom/rom`` locations, then recursively for a
    ``structures.json``) and returns its :class:`ReducedStructure` list, ready
    to concatenate onto other (live or imported) sections.

    Parameters
    ----------
    project_path : path-like
        A project folder (or a ROM directory) produced by a previous run.
    fes, mesh : optional
        Attach an FE space / mesh (only needed for field reconstruction).

    Returns
    -------
    (structures, impedance_func)
    """
    project_path = Path(project_path)
    if (project_path / "structures.json").exists():
        return load_reduced_structures(project_path, fes=fes, mesh=mesh)
    prefer = [
        project_path / "fds" / "foms" / "roms",
        project_path / "fds" / "fom" / "rom",
        project_path / "foms" / "roms",
        project_path / "roms",
        project_path / "rom",
    ]
    for d in prefer:
        if (d / "structures.json").exists():
            return load_reduced_structures(d, fes=fes, mesh=mesh)
    hits = sorted(project_path.rglob("structures.json"))
    if hits:
        return load_reduced_structures(hits[0].parent, fes=fes, mesh=mesh)
    raise FileNotFoundError(
        f"No saved reduced model (structures.json) found under {project_path}. "
        "Run and reduce the project first (fds.solve() -> reduce())."
    )
