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


from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

from cavsim3d.utils.plot_mixin import PlotMixin
from cavsim3d.core.persistence import H5Serializer, ProjectManager
from cavsim3d.rom.reduction import ModelOrderReduction
from cavsim3d.rom.structures import ReducedStructure
from cavsim3d.solvers.concatenation import ConcatenatedSystem
from cavsim3d.solvers.beam import BeamResultMixin
from cavsim3d.solvers import beam as _beam


def _safe_filename(name: str) -> str:
    """Sanitize a domain name for use in file paths.

    Replaces characters that are invalid in Windows/POSIX paths
    (``/``, ``\\``, ``|``, ``:``, ``*``, ``?``, ``<``, ``>``, ``"``)
    with underscores.
    """
    import re
    return re.sub(r'[/\\|:*?"<>]', '_', name)


def _save_beam_files(path: Path, tag: str, beam: Optional[Dict], solver, domain: str) -> None:
    """Write (or, without a beam, remove) the beam files of one section:
    z_tilde/, s_tilde/, snapshots_beam/ and matrices/beam_<section>.h5."""
    files = [path / "z_tilde" / f"z_tilde_{tag}.h5",
             path / "s_tilde" / f"s_tilde_{tag}.h5",
             path / "snapshots_beam" / f"snapshots_beam_{tag}.h5",
             path / "matrices" / f"beam_{tag}.h5"]
    if not beam:
        for f in files:
            if f.exists():
                f.unlink()
        for folder in ("z_tilde", "s_tilde", "snapshots_beam"):
            d = path / folder
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        return
    _beam.save_tilde(files[0], beam, 'Z')
    _beam.save_tilde(files[1], beam, 'S')
    raw = getattr(solver, '_beam_raw', {}).get(domain) if solver is not None else None
    if raw is not None and raw.get('snapshots') is not None:
        files[2].parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(files[2], "w") as f:
            H5Serializer.save_dataset(f, "frequencies", beam['frequencies'])
            H5Serializer.save_dataset(f, "field_snapshots", raw['snapshots'])
            f.attrs["fingerprint"] = str(beam.get('fingerprint', ''))
    system = getattr(solver, '_beam_systems', {}).get(domain) if solver is not None else None
    if system is not None:
        _beam.save_beam_data(files[3], system)


def _join_domain_beams(concat, fds, blocks: List[Dict]) -> Dict:
    """S~ of the joined model from the per-domain S~ (one per structure of
    ``concat``, in the order of ``fds.domains``)."""
    modes, labels_ext = [], []
    for s_idx, domain in enumerate(fds.domains):
        modes.append([concat.port_mode_map[(s_idx, pn, m)]
                      for (_pi, pn, m) in fds._domain_port_mode_order(domain)])
    pairs = []
    for (sa, pa), (sb, pb) in concat.connections:
        n = concat.port_to_mode_range[(sa, pa)][1]
        pairs += [(concat.port_mode_map[(sa, pa, m)], concat.port_mode_map[(sb, pb, m)])
                  for m in range(n)]
    external = list(concat._external_port_modes)
    numbers: Dict[Tuple[int, str], int] = {}
    for g in external:
        s_idx, port, m = concat._global_to_local[g]
        n = numbers.setdefault((s_idx, port), len(numbers) + 1)
        labels_ext.append(f"{n}({m + 1})")
    St = _beam.join_s_tilde([b['S_tilde'] for b in blocks], modes, pairs, external)
    first = blocks[0]
    n_ext = len(external)
    rows = labels_ext + list(first['rows'][len(first['rows']) - (St.shape[1] - n_ext):])
    cols = labels_ext + list(first['cols'][len(first['cols']) - (St.shape[2] - n_ext):])
    return {'S_tilde': St, 'Z_tilde': None, 'rows': rows, 'cols': cols,
            'frequencies': np.asarray(first['frequencies']), 'names': first.get('names', {}),
            'setup': first.get('setup'), 'fingerprint': first.get('fingerprint'),
            'summary': {'joined': list(fds.domains)}}


def _load_beam_files(path: Path, tag: str) -> Optional[Dict]:
    """The beam results of one section, or None."""
    z = _beam.load_tilde(path / "z_tilde" / f"z_tilde_{tag}.h5")
    s_ = _beam.load_tilde(path / "s_tilde" / f"s_tilde_{tag}.h5")
    if z is None and s_ is None:
        return None
    meta = z or s_
    return {'Z_tilde': z['data'] if z else None, 'S_tilde': s_['data'] if s_ else None,
            'rows': meta['rows'], 'cols': meta['cols'], 'frequencies': meta['frequencies'],
            'names': meta['names'], 'setup': meta['setup'],
            'fingerprint': meta['fingerprint'], 'summary': meta['summary'],
            'port_modes': meta.get('port_modes'), 'ports': meta.get('ports'),
            'fingerprints': meta.get('fingerprints'),
            'zref': s_.get('zref') if s_ else None}


# =============================================================================
# FOMResult
# =============================================================================

class FOMResult(PlotMixin, BeamResultMixin):
    """
    Wrapper around a single solved FOM (one domain or a global coupled result).

    Solved with a beam (``proj.add_beam``), it also holds the generalised
    matrices ``s_tilde`` / ``z_tilde`` (see :class:`~cavsim3d.solvers.beam.BeamResultMixin`).

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
        # (port_number, mode_number), 1-based, for each matrix row/column.
        # Needed when ports carry different numbers of modes.
        mode_labels: Optional[List[Tuple[int, int]]] = None,
        # beam outputs: generalised matrices with labels (solvers/beam.py)
        beam: Optional[Dict] = None,
    ):
        self.domain = domain
        self.mode_labels = ([tuple(int(v) for v in lab) for lab in mode_labels]
                            if mode_labels else None)
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
        self._beam = beam or None

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
        # Key = '<excitation>(mode)<response>(mode)', i.e. column first
        for row, (prow, mrow) in enumerate(self._labels(matrix.shape[1])):
            for col, (pcol, mcol) in enumerate(self._labels(matrix.shape[1])):
                res_dict[f'{pcol}({mcol}){prow}({mrow})'] = matrix[:, row, col]
        return res_dict

    def _labels(self, n_p: int) -> List[Tuple[int, int]]:
        """(port, mode) label per matrix index; uniform fallback if unknown."""
        if self.mode_labels and len(self.mode_labels) == n_p:
            return list(self.mode_labels)
        n_modes = self._n_modes_per_port or 1
        return [(i // n_modes + 1, i % n_modes + 1) for i in range(n_p)]

    def _row_first_dict(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Matrix -> the solver's internal row-first per-domain dict."""
        labels = self._labels(matrix.shape[1])
        return {f'{rp}({rm}){cp}({cm})': matrix[:, r, c]
                for r, (rp, rm) in enumerate(labels)
                for c, (cp, cm) in enumerate(labels)}

    # ------------------------------------------------------------------
    # Beam field
    # ------------------------------------------------------------------

    def beam_field(self, freq_index: int, beam=None, total: bool = True):
        """The field of a beam (current 1 A) at one frequency sample.

        ``total=True``: E = E_s + E_free as a CoefficientFunction (E_free,
        the beam's own field, is singular on the beam line); ``total=False``:
        the scattered field E_s that the finite elements carry, as a
        GridFunction.  ``beam``: name or label (default: the first beam).
        """
        from ngsolve import HCurl, GridFunction, exp
        from cavsim3d.solvers.nedelec import hcurl_flags
        from cavsim3d.utils.names import region_pattern
        from cavsim3d.solvers.beam import BeamSetup, axis_index, free_field_profile
        fds = self._solver_ref
        if fds is None or not self.has_beam:
            raise RuntimeError("No beam results here: define a beam and solve.")
        label = self._beam_label(beam, sources=True)
        setup = BeamSetup.from_dict(self._beam_data().get('setup'))
        j = [f"b({i + 1})" for i in range(len(setup.sources))].index(label)
        line = setup.sources[j]
        snaps = (getattr(fds, '_beam_raw', {}).get(self.domain) or {}).get('snapshots')
        if snaps is None:
            snaps = self._stored_beam_snapshots()
        n_src = len(setup.sources)
        vec = np.asarray(snaps)[:, freq_index * n_src + j]
        if self.domain == 'global':
            region = {}
        else:
            mats = fds._get_domain_mesh_materials(self.domain) or [self.domain]
            region = {'definedon': fds.mesh.Materials(region_pattern(mats))}
        fes = HCurl(fds.mesh, order=fds.order, **hcurl_flags(fds.nedelec), complex=True,
                    dirichlet=fds.bc, **region)
        e_s = GridFunction(fes)
        e_s.vec.FV().NumPy()[:] = vec
        if not total:
            return e_s
        a = axis_index(setup.axis)
        from cavsim3d.core.constants import c0
        from ngsolve import x, y, z
        k = 2 * np.pi * float(self.frequencies[freq_index]) / (line.beta * c0)
        s = (x, y, z)[a]
        return e_s + free_field_profile(line.point, a) * exp(-1j * k * s)

    def _stored_beam_snapshots(self):
        fds = self._solver_ref
        root = getattr(fds, '_project_path', None)
        if root is None:
            raise RuntimeError("The beam field was not kept (no project to read it from).")
        sub = "fom" if self.domain == 'global' else "foms"
        tag = _safe_filename(self.domain) if self.domain else "global"
        path = Path(root) / "fds" / sub / "snapshots_beam" / f"snapshots_beam_{tag}.h5"
        if not path.exists():
            raise RuntimeError("The beam field was not stored (solve with store_snapshots=True).")
        with h5py.File(path, "r") as f:
            return H5Serializer.load_dataset(f["field_snapshots"])

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
    # Solve routing
    # ------------------------------------------------------------------

    def solve(self, fmin: float = None, fmax: float = None, nsamples: int = None,
              config: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Rerun simulation for this FOM.
        
        Delegates to the underlying FrequencyDomainSolver.
        """
        if self._solver_ref is None:
            raise RuntimeError("Cannot solve: no solver reference available.")
        
        # Clear children if rerunning
        if (config and config.get('rerun')) or kwargs.get('rerun'):
            self.clear_rom()
            
        return self._solver_ref.solve(fmin=fmin, fmax=fmax, nsamples=nsamples, 
                                     config=config, **kwargs)

    def print_log(self) -> None:
        """Print the log file from the last solve, if it exists."""
        if self._solver_ref is not None:
            self._solver_ref.print_log()
        else:
            print("No solver reference available.")

    def clear_rom(self) -> None:
        """Clear the cached ROM and delete its saved data from the project folder."""
        self._rom_cache = None
        if self._solver_ref and getattr(self._solver_ref, '_project_path', None):
            project_path = Path(self._solver_ref._project_path)
            # Paths where ROM data might be stored
            paths_to_delete = [
                project_path / "fds" / "fom" / "rom",
                project_path / "eigenmode" / "fom" / "rom"
            ]
            for p in paths_to_delete:
                if p.exists():
                    print(f"  Deleting stale ROM data at {p}")
                    shutil.rmtree(p)

    # ------------------------------------------------------------------
    # Logical Matrix Access
    # ------------------------------------------------------------------

    @property
    def K(self):
        """Access the full-order stiffness matrix for this domain."""
        if self._solver_ref is None:
            return None
        if self.domain == 'global':
            return getattr(self._solver_ref, 'K_global', None)
        return getattr(self._solver_ref, 'K', {}).get(self.domain)

    @property
    def M(self):
        """Access the full-order mass matrix for this domain."""
        if self._solver_ref is None:
            return None
        if self.domain == 'global':
            return getattr(self._solver_ref, 'M_global', None)
        return getattr(self._solver_ref, 'M', {}).get(self.domain)

    @property
    def B(self):
        """Access the full-order port excitation matrix for this domain."""
        if self._solver_ref is None:
            return None
        if self.domain == 'global':
            return getattr(self._solver_ref, 'B_global', None)
        return getattr(self._solver_ref, 'B', {}).get(self.domain)

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

        mor = ModelOrderReduction(self._solver_ref)
        mor.reduce(tol=tol, max_rank=max_rank)
        self._rom_cache = mor
        # Persist to the standard single-solid ROM location so the reduced
        # model can be reused / imported later (load-if-exists), mirroring the
        # multi-solid foms.reduce() save.
        pp = getattr(self._solver_ref, '_project_path', None)
        if pp:
            try:
                mor.save(Path(pp) / "fds" / "fom" / "rom")
            except Exception as e:
                warnings.warn(f"Could not save ROM to project: {e}",
                              UserWarning, stacklevel=2)
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
        if self._solver_ref is not None and hasattr(self._solver_ref, 'calculate_resonant_modes'):
            domain = self.domain if self.domain != 'global' else 'global'
            res = self._solver_ref.calculate_resonant_modes(domain=domain, **kwargs)
            if isinstance(res, dict):
                return {k: v[0] for k, v in res.items()}
            return res[0]
        raise RuntimeError("Eigenvalues not available for this FOMResult.")

    def get_resonant_frequencies(self, **kwargs):
        if self._solver_ref is not None and hasattr(self._solver_ref, 'get_resonant_frequencies'):
            return self._solver_ref.get_resonant_frequencies(**kwargs)
        raise RuntimeError("Resonant frequencies not available for this FOMResult.")

    def get_rq(self, mode_index: int, **kwargs):
        """R/Q of one eigenmode (see :meth:`FrequencyDomainSolver.get_rq`)."""
        kwargs.setdefault('domain', self.domain)
        return self._solver_ref.get_rq(mode_index, **kwargs)

    def get_figures_of_merit(self, mode_index: int, **kwargs):
        """Figures of merit of one eigenmode (see
        :meth:`FrequencyDomainSolver.get_figures_of_merit`)."""
        kwargs.setdefault('domain', self.domain)
        return self._solver_ref.get_figures_of_merit(mode_index, **kwargs)

    def get_cell_coupling(self, first: int, last: int, **kwargs):
        """Cell-to-cell coupling [%] (see :meth:`FrequencyDomainSolver.get_cell_coupling`)."""
        kwargs.setdefault('domain', self.domain)
        return self._solver_ref.get_cell_coupling(first, last, **kwargs)

    def get_eigenmodes(self, _auto_save=True, **kwargs):
        """
        Standardized API for retrieving eigenvalues and eigenvectors.
        """
        if self._solver_ref is not None and hasattr(self._solver_ref, 'calculate_resonant_modes'):
            # Delegate to solver, but filter for this domain if not global
            domain = self.domain if self.domain != 'global' else None
            res = self._solver_ref.calculate_resonant_modes(domain=domain, **kwargs)
            
            # Populate the solver's eigen caches so save_eigenmodes can find them
            if hasattr(self._solver_ref, '_init_eigen_cache'):
                self._solver_ref._init_eigen_cache()
                if isinstance(res, dict):
                    for d, (eigs, vecs) in res.items():
                        self._solver_ref._eigenvalues_cache[d] = eigs
                        self._solver_ref._eigenvectors_cache[d] = vecs
                else:
                    eigs, vecs = res
                    cache_key = domain or 'global'
                    self._solver_ref._eigenvalues_cache[cache_key] = eigs
                    self._solver_ref._eigenvectors_cache[cache_key] = vecs
            
            # Hierarchical save if possible
            if _auto_save:
                self._auto_save_eigenmodes(res, **kwargs)
            
            return res
        raise RuntimeError("Eigenmodes not available for this FOMResult.")

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        """Helper to save eigenmodes to the mirrored project structure."""
        if self._solver_ref is None or not hasattr(self._solver_ref, 'save_eigenmodes'):
            return
        try:
            domain = self.domain if self.domain != 'global' else None
            self._solver_ref.save_eigenmodes(domain=domain, **kwargs)
        except (ValueError, Exception) as e:
            print(f"Warning: Could not auto-save eigenmodes: {e}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """
        Save FOM results to disk.
        
        Saves matrices (K, M, B) to matrices/ folder, and S/Z parameters, 
        snapshots, and eigenmodes to their respective subfolders.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Subfolders
        s_path = path / "s"
        z_path = path / "z"
        snap_path = path / "snapshots"
        eig_path = path / "eigenmodes"
        for p in [s_path, z_path, snap_path, eig_path]:
            p.mkdir(parents=True, exist_ok=True)

        # 1. Save matrices (K, M, B) separately
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        
        if self._solver_ref is not None:
            if self.domain == 'global':
                K = getattr(self._solver_ref, 'K_global', None)
                M = getattr(self._solver_ref, 'M_global', None)
                B = getattr(self._solver_ref, 'B_global', None)
                C = getattr(self._solver_ref, 'C_global', None)
                D = getattr(self._solver_ref, 'D_global', None)
            else:
                K = getattr(self._solver_ref, 'K', {}).get(self.domain)
                M = getattr(self._solver_ref, 'M', {}).get(self.domain)
                B = getattr(self._solver_ref, 'B', {}).get(self.domain)
                C = getattr(self._solver_ref, 'C', {}).get(self.domain)
                D = getattr(self._solver_ref, 'D', {}).get(self.domain)
            for name, mat in (("C", C), ("D", D)):   # loss matrices (lossy only)
                if mat is not None:
                    with h5py.File(mat_path / f"{name}.h5", "a") as f:
                        H5Serializer.save_dataset(f, "data", mat)

            if K is not None:
                with h5py.File(mat_path / "K.h5", "a") as f:
                    H5Serializer.save_dataset(f, "data", K)
            if M is not None:
                with h5py.File(mat_path / "M.h5", "a") as f:
                    H5Serializer.save_dataset(f, "data", M)
            if B is not None:
                with h5py.File(mat_path / "B.h5", "a") as f:
                    H5Serializer.save_dataset(f, "data", B)

        # 2. Save S and Z results
        # Sanitised like load() expects (a domain may contain '/').
        tag = _safe_filename(self.domain) if self.domain else None
        if self._Z_matrix is not None:
            z_file = f"z_{tag}.h5" if tag else "z.h5"
            with h5py.File(z_path / z_file, "a") as f:
                H5Serializer.save_dataset(f, "data", self._Z_matrix)
        if self._S_matrix is not None:
            s_file = f"s_{tag}.h5" if tag else "s.h5"
            with h5py.File(s_path / s_file, "a") as f:
                H5Serializer.save_dataset(f, "data", self._S_matrix)

        # 3. Save snapshots and frequencies
        snap_file = f"snapshots_{tag}.h5" if tag else "snapshots.h5"
        with h5py.File(snap_path / snap_file, "a") as f:
            if self.frequencies is not None:
                H5Serializer.save_dataset(f, "frequencies", self.frequencies)
            if self._residual_data:
                H5Serializer.save_dataset(f, "residual_data", self._residual_data)
            
            # Save field snapshots if available in solver reference
            if self._solver_ref is not None and self.domain in getattr(self._solver_ref, 'snapshots', {}):
                H5Serializer.save_dataset(f, "field_snapshots", self._solver_ref.snapshots[self.domain])

        # 3b. Beam: generalised matrices, beam snapshots, beam data
        _save_beam_files(path, tag or "global", self._beam, self._solver_ref, self.domain)

        # 4. Save eigenmodes if available
        if self._solver_ref is not None:
            # We pass the domain to save_eigenmodes to keep it granular
            # The solver's save_eigenmodes knows how to handle paths
            try:
                self._solver_ref.save_eigenmodes(path=eig_path, domain=self.domain)
            except Exception as e:
                print(f"Note: Could not save FOM eigenmodes for {self.domain}: {e}")

        # 5. Save metadata
        metadata = {
            "domain": self.domain,
            "n_ports": self.n_ports,
            "ports": self.ports,
            "n_modes_per_port": self._n_modes_per_port,
            "mode_labels": self.mode_labels,
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)

    @classmethod
    def load(cls, path: Union[str, Path], _solver_ref=None) -> 'FOMResult':
        """Load FOM result from disk."""
        path = Path(path)
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        domain = metadata["domain"]
        
        # 1. Load frequencies from snapshots or legacy
        frequencies = None
        snap_path = path / "snapshots" / (f"snapshots_{_safe_filename(domain)}.h5" if domain else "snapshots.h5")
        if not snap_path.exists():
            snap_path = path / (f"snapshots_{_safe_filename(domain)}.h5" if domain else "snapshots.h5")
            
        if snap_path.exists():
            with h5py.File(snap_path, "r") as f:
                frequencies = H5Serializer.load_dataset(f["frequencies"]) if "frequencies" in f else None

        # 2. Load Z and S
        Z_matrix = None
        S_matrix = None
        
        z_path = path / "z" / (f"z_{_safe_filename(domain)}.h5" if domain else "z.h5")
        if not z_path.exists(): z_path = path / (f"z_{_safe_filename(domain)}.h5" if domain else "z.h5")
        if z_path.exists():
            with h5py.File(z_path, "r") as f:
                Z_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None
                
        s_path = path / "s" / (f"s_{_safe_filename(domain)}.h5" if domain else "s.h5")
        if not s_path.exists(): s_path = path / (f"s_{_safe_filename(domain)}.h5" if domain else "s.h5")
        if s_path.exists():
            with h5py.File(s_path, "r") as f:
                S_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None

        # 3. Load snapshots and residuals
        residual_data = None
        
        if snap_path.exists():
            with h5py.File(snap_path, "r") as f:
                frequencies = H5Serializer.load_dataset(f["frequencies"]) if "frequencies" in f else None
                residual_data = H5Serializer.load_dataset(f["residual_data"]) if "residual_data" in f else None
                field_snapshots = H5Serializer.load_dataset(f["field_snapshots"]) if "field_snapshots" in f else None
                
                # Support legacy loading if matrices were inside snapshots.h5
                if Z_matrix is None and "Z_matrix" in f:
                    Z_matrix = H5Serializer.load_dataset(f["Z_matrix"])
                if S_matrix is None and "S_matrix" in f:
                    S_matrix = H5Serializer.load_dataset(f["S_matrix"])

        # Restore results into _solver_ref if available
        if _solver_ref is not None:
            if not hasattr(_solver_ref, '_residuals') or _solver_ref._residuals is None:
                _solver_ref._residuals = {}
            if Z_matrix is not None: _solver_ref._Z_global_coupled = Z_matrix
            if S_matrix is not None: _solver_ref._S_global_coupled = S_matrix
            if frequencies is not None: _solver_ref.frequencies = frequencies
            if residual_data is not None: _solver_ref._residuals['global'] = residual_data
            if field_snapshots is not None and domain:
                _solver_ref.snapshots[domain] = field_snapshots

        # Restore matrices into _solver_ref if available
        if _solver_ref is not None:
            domain = metadata["domain"]
            mat_path = path / "matrices"
            if mat_path.exists():
                for mname in ["K", "M", "B", "C", "D"]:
                    mfile = mat_path / f"{mname}.h5"
                    if mfile.exists():
                        with h5py.File(mfile, "r") as f:
                            data = H5Serializer.load_sparse_csr(f["data"]) if mname != "B" else H5Serializer.load_dataset(f["data"])
                            if domain == 'global':
                                setattr(_solver_ref, f"{mname}_global", data)
                            else:
                                getattr(_solver_ref, mname)[domain] = data
            elif (path / "matrices.h5").exists():
                with h5py.File(path / "matrices.h5", "r") as f:
                    for mname in ["K", "M", "B"]:
                        if mname in f:
                            data = H5Serializer.load_sparse_csr(f[mname]) if mname in ["K", "M"] else H5Serializer.load_dataset(f[mname])
                            if domain == 'global':
                                setattr(_solver_ref, f"{mname}_global", data)
                            else:
                                getattr(_solver_ref, mname)[domain] = data

        if _solver_ref is not None:
            _solver_ref.load_eigenmodes(path / "eigenmodes")

        beam = _load_beam_files(path, _safe_filename(domain) if domain else "global")
        if beam is not None and _solver_ref is not None:
            _solver_ref._beam_tilde[domain] = beam

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
            _solver_ref=_solver_ref,
            mode_labels=metadata.get("mode_labels"),
            beam=beam,
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
        
        # Only ensure ax here if we are not doing a dual-axis plot, 
        # or if an axes was already provided.
        # Otherwise, let the first fom.plot_residual create the dual structure.
        fig = None
        if what != 'both' or ax is not None:
            fig, ax = self._ensure_ax(ax)

        for i, fom in enumerate(self._foms):
            lbl = f"{label or ''}{fom.domain}" if label else fom.domain
            # Set a generic title for sub-plots to avoid flickering titles in overlays
            sub_title = title if title else (f"{fom.domain} Convergence" if len(self._foms) == 1 else None)
            try:
                # Pass ax=None for the first call if what='both' and no ax was provided
                current_ax = ax if (i > 0 or ax is not None or what != 'both') else None
                fig, res = fom.plot_residual(what=what, ax=current_ax, label=lbl,
                                             title=sub_title, show=False, **kwargs)
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
        # if show:
        #     plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Backward-compatible ROM/Concat accessors
    # ------------------------------------------------------------------

    @property
    def roms(self) -> 'ROMCollection':
        """
        Access the cached per-domain ROMs.

        Returns the ROM collection only if it has already been computed
        via ``foms.reduce()`` or loaded from disk. Does **not** trigger
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
        if self._concat_cache is None and self._saved_scattering_join():
            # a join through the parts' S~ (beam) is rebuilt from their files
            self.concatenate()
        if self._concat_cache is None:
            raise RuntimeError(
                "No concatenated system available. "
                "Call foms.concatenate() first."
            )
        return self._concat_cache

    def _saved_scattering_join(self) -> bool:
        """True if the project holds a join through the parts' S~ (beam)."""
        fds = self._fds_ref
        root = getattr(fds, '_project_path', None)
        if root is None or not all(f.has_beam for f in self._foms):
            return False
        meta = Path(root) / "fds" / "foms" / "concat" / "metadata.json"
        try:
            return bool(json.loads(meta.read_text()).get("scattering_join"))
        except (OSError, ValueError):
            return False

    # ------------------------------------------------------------------
    # Solve routing
    # ------------------------------------------------------------------

    def solve(self, fmin: float = None, fmax: float = None, nsamples: int = None,
              config: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Rerun simulation for all domains in this collection.
        
        Delegates to the underlying FrequencyDomainSolver.
        """
        if self._fds_ref is None:
            raise RuntimeError("Cannot solve: no solver reference available.")
        
        # Clear children if rerunning
        if (config and config.get('rerun')) or kwargs.get('rerun'):
            self.clear_roms()
            
        return self._fds_ref.solve(fmin=fmin, fmax=fmax, nsamples=nsamples, 
                                  config=config, **kwargs)

    def print_log(self) -> None:
        """Print the log file from the last solve, if it exists."""
        if self._fds_ref is not None:
            self._fds_ref.print_log()
        else:
            print("No solver reference available.")

    def clear_roms(self) -> None:
        """Clear the cached ROMCollection and delete its saved data from the project folder."""
        self._roms_cache = None
        if self._fds_ref and getattr(self._fds_ref, '_project_path', None):
            project_path = Path(self._fds_ref._project_path)
            # Paths where ROM/Concat data might be stored
            paths_to_delete = [
                project_path / "fds" / "foms" / "roms",
                project_path / "fds" / "foms" / "concat",
                project_path / "eigenmode" / "foms" / "roms",
                project_path / "eigenmode" / "foms" / "concat",
            ]
            for p in paths_to_delete:
                if p.exists():
                    print(f"  Deleting stale ROM/Concat data at {p}")
                    shutil.rmtree(p)

    # ------------------------------------------------------------------
    # Logical Matrix Access (Aggregated)
    # ------------------------------------------------------------------

    @property
    def K(self) -> Dict[str, np.ndarray]:
        """Access per-domain stiffness matrices as a dictionary."""
        return {fom.domain: fom.K for fom in self._foms}

    @property
    def M(self) -> Dict[str, np.ndarray]:
        """Access per-domain mass matrices as a dictionary."""
        return {fom.domain: fom.M for fom in self._foms}

    @property
    def B(self) -> Dict[str, np.ndarray]:
        """Access per-domain port excitation matrices as a dictionary."""
        return {fom.domain: fom.B for fom in self._foms}

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


        fds = self._fds_ref
        mor = ModelOrderReduction(fds)
        mor.reduce(tol=tol, max_rank=max_rank)

        self._roms_cache = ROMCollection(_fds_ref=self._fds_ref, _mor_ref=mor)
        
        # Explicitly save the ROM hierarchy now that _roms_cache is set
        # (The auto-save inside mor.reduce() fires before _roms_cache is set,
        #  so it always skips ROM saving in the FDS save chain)
        if hasattr(self._fds_ref, '_project_path') and self._fds_ref._project_path:
            try:
                roms_path = Path(self._fds_ref._project_path) / "fds" / "foms" / "roms"
                self._roms_cache.save(roms_path)
            except Exception as e:
                warnings.warn(f"Could not save ROM hierarchy: {e}", UserWarning, stacklevel=2)
        
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


        fds = self._fds_ref
        blocks = [getattr(fds, '_beam_tilde', {}).get(d) for d in fds.domains]
        if blocks and all(b is not None and b.get('S_tilde') is not None for b in blocks):
            return self._concatenate_scattering(blocks)

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

            # ConcatenatedSystem.solve() expects the MASS-ORTHONORMAL form
            #     (A - ω²I) y = B u,
            # which is exactly what the ROM path produces in
            # rom/reduction.py: with M = Q Λ Qᵀ and T = Q Λ^(-1/2) we get
            # TᵀMT = I, so (K - ω²M)x = Bu becomes (TᵀKT - ω²I)y = TᵀBu,
            # with x = T y.
            #
            # Do NOT use A = M⁻¹K here: that matrix is not symmetric, so
            # symmetrizing it silently destroys the operator and the coupled
            # system degenerates to total reflection (|S11|=1, |S21|=0).
            # Mirroring the ROM transformation keeps FOM and ROM
            # concatenation on one convention.
            if hasattr(B_free, "toarray"):
                B_free = B_free.toarray()
            B_free = np.asarray(B_free)

            lam, Q = sl.eigh(M_free)

            # Drop near-zero/negative mass eigenvalues (same guard as the ROM)
            lam_max = np.max(np.abs(lam)) if lam.size else 0.0
            valid = lam > np.finfo(float).eps * lam_max
            if not np.all(valid):
                lam, Q = lam[valid], Q[:, valid]

            Q_L_inv = Q @ np.diag(1.0 / np.sqrt(lam))

            Ard = Q_L_inv.T @ K_free @ Q_L_inv
            Ard = 0.5 * (Ard + Ard.T)
            Brd = Q_L_inv.T @ B_free

            # Loss operators in the same coordinates (lossy domains only)
            loss = {}
            for name in ("C", "D"):
                X = getattr(fds, name, {}).get(domain)
                if X is not None:
                    X_free = X[np.ix_(free_dofs, free_dofs)].toarray()
                    X_r = Q_L_inv.T @ X_free @ Q_L_inv
                    loss[name] = 0.5 * (X_r + X_r.T)

            n_free = len(lam)

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
                Crd=loss.get("C"),
                Drd=loss.get("D"),
            ))

        concat = ConcatenatedSystem(
            structures=structures,
            port_impedance_func=fds._get_port_impedance,
            port_wave_impedance_func=fds._port_wave_impedance,
            solver_ref=fds,
        )

        # Connect the domains at their SHARED interface ports -- the same rule
        # the ROM path uses (ModelOrderReduction._build_connections), so FOM
        # and ROM concatenation agree for any topology, not only 2-port chains.
        domain_index = {d: i for i, d in enumerate(fds.domains)}
        connections = []
        for port in fds.internal_ports:
            doms = [d for d in fds.domains if port in fds.domain_port_map.get(d, [])]
            for other in doms[1:]:
                connections.append(((domain_index[doms[0]], port),
                                    (domain_index[other], port)))

        concat.define_connections(connections)
        concat.couple()

        self._concat_cache = concat
        return concat

    def _concatenate_scattering(self, blocks: List[Dict]):
        """Join the parts through their generalised scattering matrices.

        With a beam, every part's S~ (port modes and beams) is joined at the
        cuts (CSC-BEAM, docs/theory/beam.md §9.9): the joined S, Z and S~ at
        the full-order frequencies, without any full-order matrices.  Other
        frequencies need reduced models of the parts (not available with a
        beam yet).
        """
        fds = self._fds_ref
        structures = []
        for domain in fds.domains:
            ports = fds.domain_port_map[domain]
            port_modes_d = {p: fds.port_modes[p] for p in ports if p in fds.port_modes}
            n_pm = sum(len(m) for m in port_modes_d.values())
            # the port bookkeeping only: no operator (r = 0)
            structures.append(ReducedStructure(
                Ard=np.zeros((0, 0)), Brd=np.zeros((0, n_pm)), ports=ports,
                port_modes=port_modes_d, domain=domain, r=0, n_full=0,
                is_full_order=True))
        concat = ConcatenatedSystem(
            structures=structures,
            port_impedance_func=fds._get_port_impedance,
            port_wave_impedance_func=fds._port_wave_impedance,
            solver_ref=fds,
        )
        domain_index = {d: i for i, d in enumerate(fds.domains)}
        connections = []
        for port in fds.internal_ports:
            doms = [d for d in fds.domains if port in fds.domain_port_map.get(d, [])]
            for other in doms[1:]:
                connections.append(((domain_index[doms[0]], port),
                                    (domain_index[other], port)))
        concat.define_connections(connections)

        beam = _join_domain_beams(concat, fds, blocks)
        freqs = np.asarray(beam['frequencies'])
        ext = [concat._global_to_local[g] for g in concat._external_port_modes]
        Zref = np.array([np.diag([fds._get_port_impedance(p, m, f) for (_s, p, m) in ext])
                         for f in freqs])
        beam['Z_tilde'] = _beam.z_tilde_from_s_tilde(beam['S_tilde'], Zref)
        n = len(ext)
        concat.frequencies = freqs
        concat._S_matrix = beam['S_tilde'][:, :n, :n].copy()
        concat._Z_matrix = beam['Z_tilde'][:, :n, :n].copy()
        concat._beam = beam
        concat._scattering_join = True
        concat._invalidate_cache()
        import cavsim3d.utils.printing as _pr
        _pr.milestone(f"Joined {len(fds.domains)} parts through their generalised "
                      f"scattering matrices (beam): {n} external port mode(s), "
                      f"{len(freqs)} frequencies (those of the full-order solve).")
        self._concat_cache = concat
        if getattr(fds, '_project_path', None):
            concat.save(Path(fds._project_path) / "fds" / "foms" / "concat")
        return concat

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Save FOMCollection to multi-solid path."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Subfolders
        s_path = path / "s"
        z_path = path / "z"
        snap_path = path / "snapshots"
        eig_path = path / "eigenmodes"
        for p in [s_path, z_path, snap_path, eig_path]:
            p.mkdir(parents=True, exist_ok=True)

        # 1. Save matrices in dedicated files inside a matrices/ folder (prefixed by domain)
        mat_path = path / "matrices"
        mat_path.mkdir(parents=True, exist_ok=True)
        
        for fom in self._foms:
            if fom._solver_ref is not None:
                domain = fom.domain
                if domain in getattr(fom._solver_ref, 'K', {}):
                    with h5py.File(mat_path / f"K_{_safe_filename(domain)}.h5", "a") as fk:
                        H5Serializer.save_dataset(fk, "data", fom._solver_ref.K.get(domain))
                    with h5py.File(mat_path / f"M_{_safe_filename(domain)}.h5", "a") as fm:
                        H5Serializer.save_dataset(fm, "data", fom._solver_ref.M.get(domain))
                    with h5py.File(mat_path / f"B_{_safe_filename(domain)}.h5", "a") as fb:
                        H5Serializer.save_dataset(fb, "data", fom._solver_ref.B.get(domain))
                    for name in ("C", "D"):   # loss matrices (lossy domains only)
                        mat = getattr(fom._solver_ref, name, {}).get(domain)
                        if mat is not None:
                            with h5py.File(mat_path / f"{name}_{_safe_filename(domain)}.h5", "a") as fl:
                                H5Serializer.save_dataset(fl, "data", mat)

        # 2. Save S and Z results
        for fom in self._foms:
            domain = fom.domain
            if fom._Z_matrix is not None:
                with h5py.File(z_path / f"z_{_safe_filename(domain)}.h5", "a") as fz:
                    H5Serializer.save_dataset(fz, "data", fom._Z_matrix)
            if fom._S_matrix is not None:
                with h5py.File(s_path / f"s_{_safe_filename(domain)}.h5", "a") as fs:
                    H5Serializer.save_dataset(fs, "data", fom._S_matrix)

        # 3. Save frequencies, residual data, and field snapshots
        for fom in self._foms:
            domain = fom.domain
            snap_file = f"snapshots_{_safe_filename(domain)}.h5"
            with h5py.File(snap_path / snap_file, "a") as fsnap:
                if self.frequencies is not None:
                    H5Serializer.save_dataset(fsnap, "frequencies", self.frequencies)
                if fom._residual_data:
                    H5Serializer.save_dataset(fsnap, "residual_data", fom._residual_data)
                
                # Save field snapshots if available in solver reference
                if fom._solver_ref is not None and domain in getattr(fom._solver_ref, 'snapshots', {}):
                    H5Serializer.save_dataset(fsnap, "field_snapshots", fom._solver_ref.snapshots[domain])

        # 3b. Beam files per domain
        for fom in self._foms:
            _save_beam_files(path, _safe_filename(fom.domain), fom._beam, fom._solver_ref,
                             fom.domain)

        # 4. Save metadata
        metadata = {
            "n_solids": len(self._foms),
            "solids": [
                {
                    "domain": f.domain,
                    "n_ports": f.n_ports,
                    "ports": f.ports,
                    "n_modes_per_port": f._n_modes_per_port,
                    "mode_labels": f.mode_labels,
                } for f in self._foms
            ],
            "timestamp": datetime.now().isoformat()
        }
        ProjectManager.save_json(path, metadata)

        # 5. Save eigenmodes
        if self._fds_ref is not None:
            try:
                self._fds_ref.save_eigenmodes(eig_path)
            except Exception as e:
                warnings.warn(f"Could not save FOMCollection eigenmodes to {eig_path}: {e}")

        # 6. Save cached concatenation if available
        if self._concat_cache is not None:
            self._concat_cache.save(path / "concat")

    @classmethod
    def load(cls, path: Union[str, Path], _fds_ref=None) -> FOMCollection:
        """Load FOMCollection from disk."""
        path = Path(path)
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        fom_list = []
        
        # 1. Load frequencies
        frequencies = None
        for d_meta in metadata.get("solids", []):
            d = d_meta["domain"]
            snap_path = path / "snapshots" / f"snapshots_{_safe_filename(d)}.h5"
            if not snap_path.exists(): snap_path = path / f"snapshots_{_safe_filename(d)}.h5"
            
            if snap_path.exists():
                with h5py.File(snap_path, "r") as fs:
                    frequencies = H5Serializer.load_dataset(fs["frequencies"]) if "frequencies" in fs else None
                    if frequencies is not None: break
        
        if frequencies is None and (path / "snapshots" / "snapshots.h5").exists():
            with h5py.File(path / "snapshots" / "snapshots.h5", "r") as fs:
                frequencies = H5Serializer.load_dataset(fs["frequencies"]) if "frequencies" in fs else None
        elif frequencies is None and (path / "snapshots.h5").exists():
            with h5py.File(path / "snapshots.h5", "r") as fs:
                frequencies = H5Serializer.load_dataset(fs["frequencies"]) if "frequencies" in fs else None

        # 2. Load matrices into _fds_ref if available
        if _fds_ref is not None:
            mat_path = path / "matrices"
            if mat_path.exists():
                for d_meta in metadata.get("solids", []):
                    domain = d_meta["domain"]
                    for mname in ["K", "M", "B", "C", "D"]:
                        mfile = mat_path / f"{mname}_{_safe_filename(domain)}.h5"
                        if mfile.exists():
                            with h5py.File(mfile, "r") as f:
                                data = H5Serializer.load_sparse_csr(f["data"]) if mname != "B" else H5Serializer.load_dataset(f["data"])
                                getattr(_fds_ref, mname)[domain] = data
                        else:
                            mfile_leg = mat_path / f"{mname}.h5"
                            if mfile_leg.exists():
                                with h5py.File(mfile_leg, "r") as f:
                                    if domain in f:
                                        data = H5Serializer.load_sparse_csr(f[domain]) if mname in ["K", "M"] else H5Serializer.load_dataset(f[domain])
                                        if mname == "K": _fds_ref.K[domain] = data
                                        elif mname == "M": _fds_ref.M[domain] = data
                                        else: _fds_ref.B[domain] = data
            elif (path / "matrices.h5").exists():
                with h5py.File(path / "matrices.h5", "r") as fm:
                    for d_meta in metadata.get("solids", []):
                        domain = d_meta["domain"]
                        if domain in fm:
                            if "K" in fm[domain]: _fds_ref.K[domain] = H5Serializer.load_sparse_csr(fm[f"{domain}/K"])
                            if "M" in fm[domain]: _fds_ref.M[domain] = H5Serializer.load_sparse_csr(fm[f"{domain}/M"])
                            if "B" in fm[domain]: _fds_ref.B[domain] = H5Serializer.load_dataset(fm[f"{domain}/B"])
            # Restore eigenmodes into _fds_ref if available
            _fds_ref.load_eigenmodes(path / "eigenmodes")

        # 3. Iterate through domains to build FOMResult objects
        for solid_meta in metadata["solids"]:
            domain = solid_meta["domain"]
            Z_matrix = None
            S_matrix = None
            residual_data = None
            field_snapshots = None
            
            # Load Z
            z_path = path / "z" / f"z_{_safe_filename(domain)}.h5"
            if not z_path.exists(): z_path = path / f"z_{_safe_filename(domain)}.h5"
            if z_path.exists():
                with h5py.File(z_path, "r") as fz:
                    Z_matrix = H5Serializer.load_dataset(fz["data"])
            elif (path / "z.h5").exists():
                with h5py.File(path / "z.h5", "r") as fz:
                    if domain in fz: Z_matrix = H5Serializer.load_dataset(fz[domain])

            # Load S
            s_path = path / "s" / f"s_{_safe_filename(domain)}.h5"
            if not s_path.exists(): s_path = path / f"s_{_safe_filename(domain)}.h5"
            if s_path.exists():
                with h5py.File(s_path, "r") as fsr:
                    S_matrix = H5Serializer.load_dataset(fsr["data"])
            elif (path / "s.h5").exists():
                with h5py.File(path / "s.h5", "r") as fsr:
                    if domain in fsr: S_matrix = H5Serializer.load_dataset(fsr[domain])
            
            # Load snapshots and residuals
            snap_path = path / "snapshots" / f"snapshots_{_safe_filename(domain)}.h5"
            if not snap_path.exists(): snap_path = path / f"snapshots_{_safe_filename(domain)}.h5"
            if snap_path.exists():
                with h5py.File(snap_path, "r") as fs:
                    residual_data = H5Serializer.load_dataset(fs["residual_data"]) if "residual_data" in fs else None
                    field_snapshots = H5Serializer.load_dataset(fs["field_snapshots"]) if "field_snapshots" in fs else None

            beam = _load_beam_files(path, _safe_filename(domain))
            if beam is not None and _fds_ref is not None:
                _fds_ref._beam_tilde[domain] = beam
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
                _solver_ref=_fds_ref,
                mode_labels=solid_meta.get("mode_labels"),
                beam=beam,
            )

            # Update solver state.  The solver keeps per-domain results as
            # row-first '{row}({m}){col}({n})' dicts, not as matrices.
            if _fds_ref is not None:
                if not hasattr(_fds_ref, '_residuals') or _fds_ref._residuals is None:
                    _fds_ref._residuals = {}
                if residual_data is not None: _fds_ref._residuals[domain] = residual_data
                if field_snapshots is not None: _fds_ref.snapshots[domain] = field_snapshots
                if Z_matrix is not None:
                    _fds_ref._Z_per_domain[domain] = fom._row_first_dict(Z_matrix)
                if S_matrix is not None:
                    _fds_ref._S_per_domain[domain] = fom._row_first_dict(S_matrix)
                if frequencies is not None: _fds_ref.frequencies = frequencies

            fom_list.append(fom)
            
        return cls(fom_list, _fds_ref=_fds_ref)

    def __repr__(self) -> str:
        return (f"FOMCollection([{', '.join(f.domain for f in self._foms)}])")

    def get_eigenmodes(self, **kwargs):
        """
        Compute or retrieve eigenmodes for all domains in the collection.
        """
        if self._fds_ref is not None and hasattr(self._fds_ref, 'calculate_resonant_modes'):
            res = self._fds_ref.calculate_resonant_modes(domain=None, **kwargs)
            
            # Hierarchical save
            self._auto_save_eigenmodes(res, **kwargs)
            
            return res
        raise RuntimeError("Eigenmodes not available for this FOMCollection.")

    def get_eigenvalues(self, **kwargs):
        """
        Compute or retrieve eigenvalues for all domains in the collection.
        """
        if self._fds_ref is not None and hasattr(self._fds_ref, 'calculate_resonant_modes'):
            res = self._fds_ref.calculate_resonant_modes(domain=None, **kwargs)
            return {k: v[0] for k, v in res.items()}
        raise RuntimeError("Eigenvalues not available for this FOMCollection.")

    def _auto_save_eigenmodes(self, eigenmodes, **kwargs):
        if self._fds_ref is None or not hasattr(self._fds_ref, 'save_eigenmodes'):
            return
        try:
            self._fds_ref.save_eigenmodes(domain=None, **kwargs)
        except (ValueError, Exception) as e:
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

    def plot_s(self, params=None, plot_type='db', ax=None, label=None,
               title=None, show=False, **kwargs):
        """Overlay S-parameters for every domain on a single Axes."""
        fig, ax = self._ensure_ax(ax)
        per_domain = getattr(self._mor_ref, '_per_domain_results', None)
        
        if per_domain:
            for domain, res in per_domain.items():
                lbl = f"{label or ''}{domain}" if label else domain
                # Wrap dict results in FOMResult to use its plot_s
                # ROM per-domain results have same structure as FOM results
                fom = FOMResult(
                    domain=domain,
                    frequencies=res['frequencies'],
                    Z_matrix=res.get('Z'),
                    S_matrix=res.get('S'),
                    Z_dict=res.get('Z_dict'),
                    S_dict=res.get('S_dict'),
                    n_ports=len(res.get('ports', [])),
                    ports=res.get('ports', []),
                    n_modes_per_port=getattr(self._mor_ref, '_n_modes_per_port', 1),
                    _solver_ref=self._fds_ref
                )
                fig, ax = fom.plot_s(params=params, plot_type=plot_type, ax=ax,
                                     label=lbl, title=title, **kwargs)
        else:
            # Fallback for single-domain or global coupled results
            fig, ax = self._mor_ref.plot_s(params=params, plot_type=plot_type, ax=ax,
                                         label=label, title=title, show=False, **kwargs)
        
        if title:
            ax.set_title(title)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig, ax

    def plot_z(self, params=None, plot_type='db', ax=None, label=None,
               title=None, show=False, **kwargs):
        """Overlay Z-parameters for every domain on a single Axes."""
        fig, ax = self._ensure_ax(ax)
        per_domain = getattr(self._mor_ref, '_per_domain_results', None)
        
        if per_domain:
            for domain, res in per_domain.items():
                lbl = f"{label or ''}{domain}" if label else domain
                fom = FOMResult(
                    domain=domain,
                    frequencies=res['frequencies'],
                    Z_matrix=res.get('Z'),
                    S_matrix=res.get('S'),
                    Z_dict=res.get('Z_dict'),
                    S_dict=res.get('S_dict'),
                    n_ports=len(res.get('ports', [])),
                    ports=res.get('ports', []),
                    n_modes_per_port=getattr(self._mor_ref, '_n_modes_per_port', 1),
                    _solver_ref=self._fds_ref
                )
                fig, ax = fom.plot_z(params=params, plot_type=plot_type, ax=ax,
                                     label=lbl, title=title, **kwargs)
        else:
            fig, ax = self._mor_ref.plot_z(params=params, plot_type=plot_type, ax=ax,
                                         label=label, title=title, show=False, **kwargs)
        
        if title:
            ax.set_title(title)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        return fig, ax

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
    # Solve routing
    # ------------------------------------------------------------------

    def print_log(self) -> None:
        """Print the log file from the last solve, if it exists."""
        if self._mor_ref is not None:
            self._mor_ref.print_log()
        else:
            print("No MOR reference available.")

    def print_reduce_log(self) -> None:
        """Print the log file from the last reduce() call, if it exists."""
        if self._mor_ref is not None:
            self._mor_ref.print_reduce_log()
        else:
            print("No MOR reference available.")

    def solve(self, fmin: float = None, fmax: float = None, nsamples: int = None,
              config: Optional[Dict] = None, **kwargs) -> Dict:
        """
        Solve all reduced models in this collection.

        Delegates to the underlying ModelOrderReduction object.
        """
        if self._mor_ref is None:
            raise RuntimeError("Cannot solve: no MOR reference available.")
        result = self._mor_ref.solve(fmin=fmin, fmax=fmax, nsamples=nsamples,
                                   config=config, **kwargs)

        # Save the sweep's results (Z, S, snapshots); the reduced matrices are
        # those of reduce() and stay as saved
        if hasattr(self._fds_ref, '_project_path') and self._fds_ref._project_path:
            try:
                roms_path = Path(self._fds_ref._project_path) / "fds" / "foms" / "roms"
                self._mor_ref.save(roms_path, results_only=True)
                ref = getattr(self._fds_ref, '_project_ref', None)
                if ref is not None:
                    ref.save_timing()
            except Exception as e:
                warnings.warn(f"Could not save ROM results: {e}", UserWarning, stacklevel=2)
        
        return result

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
        
        # Explicitly save after concatenation (saves concat/ subfolder)
        if hasattr(self._fds_ref, '_project_path') and self._fds_ref._project_path:
            try:
                roms_path = Path(self._fds_ref._project_path) / "fds" / "foms" / "roms"
                self.save(roms_path)
            except Exception as e:
                warnings.warn(f"Could not save concat: {e}", UserWarning, stacklevel=2)
        
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

    def get_external_q(self, **kwargs):
        return self._mor_ref.get_external_q(**kwargs)

    def get_rq(self, mode_index: int, **kwargs):
        return self._mor_ref.get_rq(mode_index, **kwargs)

    def get_figures_of_merit(self, mode_index: int, **kwargs):
        return self._mor_ref.get_figures_of_merit(mode_index, **kwargs)

    def get_cell_coupling(self, first: int, last: int, **kwargs):
        return self._mor_ref.get_cell_coupling(first, last, **kwargs)

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
        if self._mor_ref is None or not hasattr(self._mor_ref, 'save_eigenmodes'):
            return
        try:
            self._mor_ref.save_eigenmodes(**kwargs)
        except (ValueError, Exception) as e:
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
        labels = (fds._matrix_index_labels(Z_mat.shape[1], fds._n_modes_per_port or 1)
                  if Z_mat is not None else None)
    else:
        # Per-domain: the solver's own (port, mode) ordering, which allows a
        # different number of modes per port.  Dicts are rebuilt from the
        # matrices in the standard excitation-first key convention.
        z_domain = fds._Z_per_domain.get(domain)
        s_domain = fds._S_per_domain.get(domain)
        domain_ports = fds.domain_port_map.get(domain, [])
        Z_mat = fds._domain_dict_to_matrix(domain, z_domain) if z_domain else None
        S_mat = fds._domain_dict_to_matrix(domain, s_domain) if s_domain else None
        z_dict = s_dict = None
        labels = [(pidx + 1, m + 1)
                  for (pidx, _p, m) in fds._domain_port_mode_order(domain)]
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
        mode_labels=labels,
        beam=getattr(fds, '_beam_tilde', {}).get(domain),
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


# =============================================================================
# Assembly-netlist collections (repeat-N sections, imported projects)
# =============================================================================

class NetlistSection:
    """One unique section of a netlist assembly.

    ``proj.fds.foms['cavity']`` returns this.  A section computed in this
    project (``kind == 'live'``) keeps its full-order results in the project's
    ``fds/foms`` tree; the usual result API reads them::

        sec = proj.fds.foms['cavity']
        sec.plot_s(['1(1)1(1)'])     # the section's full-order S-parameters
        sec.fom                      # its FOMResult

    Mapping access (``sec['kind']``) reads the section's record.
    """

    __slots__ = ('_name', '_rec', '_root', '_fom')

    def __init__(self, name: str, record: Dict, project_root=None):
        self._name = name
        self._rec = record
        self._root = Path(project_root) if project_root is not None else None
        self._fom = None

    # -- what the section IS ------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> Optional[str]:
        """'live' (computed in this project) or 'imported' (another project's)."""
        return self._rec.get('kind')

    @property
    def project(self):
        """Always None: a section is solved in a scratch project that is not kept.

        Its results are staged in this project (see :attr:`fom`).
        """
        return None

    @property
    def fom(self):
        """The section's full-order result (S, Z over the solve band)."""
        if self._fom is None:
            meta = self._rec.get('fom')
            if self.kind != 'live' or not meta or self._root is None:
                raise AttributeError(
                    f"Section {self._name!r} is imported from "
                    f"{self._rec.get('source')!r}: its full-order result is in "
                    f"that project.")
            from cavsim3d.solvers import netlist_persistence as npz
            self._fom = npz.load_staged_fom(self._root, self._name, meta)
        return self._fom

    # -- forward the usual result API ---------------------------------------
    @property
    def frequencies(self):
        return self.fom.frequencies

    @property
    def S_dict(self):
        return self.fom.S_dict

    @property
    def Z_dict(self):
        return self.fom.Z_dict

    @property
    def ports(self):
        return list(self.fom.ports)

    def plot_s(self, *args, **kwargs):
        return self.fom.plot_s(*args, **kwargs)

    def plot_z(self, *args, **kwargs):
        return self.fom.plot_z(*args, **kwargs)

    # -- the section's record -------------------------------------------------
    def __getitem__(self, key):
        return self._rec[key]

    def get(self, key, default=None):
        return self._rec.get(key, default)

    def __contains__(self, key):
        return key in self._rec

    def __repr__(self) -> str:
        return f"NetlistSection({self._name!r}, kind={self.kind!r})"


class NetlistFOMs:
    """Per-component FOM stage of an assembly NETLIST.

    Produced by ``fds.solve()`` when the project geometry is an assembly whose
    components carry repeat counts (``n > 1``) and/or reference already-run
    projects (and rebuilt from the project's files when it is reopened).
    Mirrors the standard fluent chain:

        proj.fds.solve(config=...)                # FOM per unique component
        roms  = proj.fds.foms.reduce(tol=...)     # ROM per unique component
        concat = roms.concatenate()               # coupled system (netlist expanded)
        concat.solve(...); concat.reduce(...)     # sweep / further reduction

    Each unique component is computed ONCE regardless of its repeat count;
    imported components are loaded, never recomputed.  ``reduce()`` may be
    called again (another ``tol``): it reduces from the staged full-order
    files.
    """

    def __init__(self, assembly, components: Dict[str, Dict], fds_ref, fom_config: Dict):
        self._assembly = assembly
        self._components = components          # base_name -> section record
        self._fds_ref = fds_ref
        self._config = fom_config
        self._roms_cache = None
        self._concat_cache = None               # joined with the beam (S~)

    @property
    def _root(self) -> Path:
        return Path(self._fds_ref._project_path)

    # -- introspection -----------------------------------------------------
    @property
    def keys(self) -> List[str]:
        return list(self._components.keys())

    def __len__(self) -> int:
        return len(self._components)

    def __getitem__(self, name: str) -> 'NetlistSection':
        if name not in self._components:
            raise KeyError(
                f"No section {name!r} in this netlist. Sections: "
                f"{list(self._components)}")
        return NetlistSection(name, self._components[name], self._root)

    def __repr__(self) -> str:
        parts = ", ".join(f"{b}({r.get('kind')})" for b, r in self._components.items())
        return f"NetlistFOMs([{parts}])"

    # -- stages ------------------------------------------------------------
    def reduce(self, tol: float = 1e-6, max_rank: Optional[int] = None) -> "NetlistROMs":
        """ROM stage: reduce each unique section once and stage its ROM into the
        single flat ``fds/foms/roms`` tree (``matrices/A_r_<domain>.h5`` …).

        Live sections are reduced from their staged full-order files (their
        FOM is never recomputed; a section already reduced with the same
        ``tol`` and ``max_rank`` is reused); imported sections are copied from
        (or referenced in) their already-run ROM.  The merged
        ``foms/roms/structures.json`` lists every section with its own
        fingerprints/band/impedance so the sections stay distinct.
        """
        import time
        from cavsim3d.rom.reduction import check_reduce_args
        from cavsim3d.solvers import netlist_persistence as npz
        from cavsim3d.utils.timing import get_timing_registry
        import cavsim3d.utils.printing as pr
        import shutil as _shutil

        check_reduce_args(tol, max_rank)
        project_root = self._root
        roms_dir = project_root / "fds" / "foms" / "roms"
        flat = roms_dir / "structures.json"
        existing = {}
        if flat.exists():
            existing = {e.get("domain"): e for e in
                        json.loads(flat.read_text()).get("structures", [])}
        t0 = time.time()
        entries, changed = [], False
        reduced_now = []                    # sections reduced by this call

        def announce():
            """Header before the first section this call reduces."""
            if not reduced_now:
                pr.running("\n" + "=" * 60)
                pr.running("Model Order Reduction")
                pr.running("=" * 60)

        for base, rec in self._components.items():
            if rec.get("kind") == "live":
                prev = existing.get(base)
                if (prev is not None and prev.get("tol") == float(tol)
                        and prev.get("max_rank") == max_rank
                        and (prev.get("reduction") or {}).get("beam") == rec.get("beam")
                        and (roms_dir / "matrices" / f"A_r_{base}.h5").exists()):
                    entries.append(prev)            # reduced so already: reuse
                    continue
                template = rec.get("rom_template")
                if template is None:
                    raise RuntimeError(
                        f"Section '{base}' has no recorded port data (solved by an "
                        "older version): solve the project again with rerun=True.")
                announce()
                entry = npz.reduce_staged_section(project_root, base, template,
                                                  tol, max_rank)
                pr.info(f"  {base}: {entry['n_full']} -> {entry['r']} DOFs")
                entries.append(entry)
                reduced_now.append(base)
                changed = True
                continue
            if rec.get("local") and base in existing \
                    and not existing[base].get("source_rom_dir"):
                entries.append(existing[base])      # copied earlier: reuse
                continue
            changed = True
            if rec["kind"] == "imported" and rec.get("reduce"):
                # Full-order results but no reduced model: reduce them here
                # (the source is read, never written) and keep the result.
                import tempfile as _tf
                announce()
                work = Path(_tf.mkdtemp(prefix="cavsim3d_reduce_"))
                try:
                    npz.reduce_source_into(Path(rec["source"]), work, tol, max_rank)
                    entries.append(npz.stage_rom(work, base, project_root))
                finally:
                    _shutil.rmtree(work, ignore_errors=True)
                reduced_now.append(base)
                continue
            src = Path(rec["source"])
            try:
                npz.find_rom_dir(src)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Imported section '{base}' has no saved reduced model "
                    f"under {src}. Reduce it in its own project first "
                    "(fds.fom.reduce / fds.foms.reduce).")
            if rec.get("mode") == "reference":
                # read in place from the source project; nothing copied
                entries.append(npz.reference_rom(src, base, project_root))
            else:
                entries.append(npz.stage_rom(src, base, project_root))
        if changed or [e.get("domain") for e in entries] != list(existing):
            npz.write_flat_structures(project_root, entries)
            # a joined model of the previous reduced models no longer applies
            _shutil.rmtree(roms_dir / "concat", ignore_errors=True)
        total_full = sum(int(e.get("n_full", 0)) for e in entries)
        total_r = sum(int(e.get("r", 0)) for e in entries)
        get_timing_registry().record(
            "reduction", time.time() - t0, category="ROM",
            full_dofs=total_full, reduced_dofs=total_r, n_domains=len(entries))
        # report what this call reduced; the other sections' ROMs were reused
        now = [e for e in entries if e.get("domain") in reduced_now]
        now_full = sum(int(e.get("n_full", 0)) for e in now)
        now_r = sum(int(e.get("r", 0)) for e in now)
        if now_full:
            pr.done(f"Reduction complete: {now_full} -> {now_r} DOFs "
                    f"({100 * (1 - now_r / now_full):.1f}% compression)")
        reused = [e.get("domain") for e in entries if e.get("domain") not in reduced_now]
        if reused:
            # nothing reduced: say so; otherwise the reuse is a detail
            (pr.info if reduced_now else pr.done)(
                f"Reduced models reused: {', '.join(reused)}")
        self._roms_cache = NetlistROMs(self._assembly, self._components, self._fds_ref,
                                       self._config, tol)
        return self._roms_cache

    @property
    def roms(self) -> "NetlistROMs":
        """The ROM stage from the last :meth:`reduce` (also after reopening)."""
        if self._roms_cache is None:
            flat = self._root / "fds" / "foms" / "roms" / "structures.json"
            entries = (json.loads(flat.read_text()).get("structures", [])
                       if flat.exists() else [])
            domains = {e.get("domain") for e in entries}
            if not entries or any(b not in domains for b in self._components):
                raise RuntimeError("No reduced models yet: call "
                                   "proj.fds.foms.reduce(tol) first.")
            tol = next((e.get("tol") for e in entries if e.get("tol") is not None), None)
            self._roms_cache = NetlistROMs(self._assembly, self._components,
                                           self._fds_ref, self._config, tol)
        return self._roms_cache

    def concatenate(self):
        """Join the parts at the full-order level -- with a beam only.

        With beams (``proj.add_beam``) the parts are joined through their
        generalised scattering matrices S~ (port modes and beams, see
        :meth:`_concatenate_scattering`): the joined S, Z and S~ at the
        full-order frequencies, without any matrices.  Without a beam the
        pipeline is FOM -> ROM -> Concatenation: ``fds.foms.reduce(tol).concatenate()``.
        """
        if self._fds_ref is not None and self._fds_ref.beam_setup is not None:
            return self._concatenate_scattering()
        raise NotImplementedError(
            "FOM-level concatenation of an assembly netlist is not supported "
            "(sections live on different meshes and would couple as dense "
            "full-order blocks). Reduce first: fds.foms.reduce(tol).concatenate().")

    @property
    def concat(self):
        """The parts joined with the beam (:meth:`concatenate`), also after
        reopening the project."""
        if self._concat_cache is None:
            meta = self._root / "fds" / "foms" / "concat" / "metadata.json"
            saved = False
            try:
                saved = bool(json.loads(meta.read_text()).get("scattering_join"))
            except (OSError, ValueError):
                pass
            if not saved or self._fds_ref.beam_setup is None:
                raise RuntimeError("No joined model yet: call proj.fds.foms.concatenate() "
                                   "(with a beam), or reduce first: "
                                   "proj.fds.foms.reduce(tol).concatenate().")
            self._concatenate_scattering(announce=False)
        return self._concat_cache

    def _section_tilde(self, base: str) -> Optional[Dict]:
        """S~ of section ``base``: from this project's flat tree, else (a part
        referenced in place) from its own project."""
        t = _beam.load_tilde(self._root / "fds" / "foms" / "s_tilde" / f"s_tilde_{base}.h5")
        rec = self._components.get(base, {})
        if t is None and rec.get("kind") == "imported" and rec.get("mode") == "reference":
            from cavsim3d.solvers import netlist_persistence as npz
            t = _beam.load_tilde(npz.source_tilde_file(Path(rec["source"])))
        return t

    def _concatenate_scattering(self, announce: bool = True):
        """Join the parts through their generalised scattering matrices (beam).

        Every copy of a part is joined with the S~ of its part, solved with
        the beams where they run through it (in the part's own frame).  A copy
        placed at z_i along the axis sees the beam's phase exp(-j k_b z_i):
        its beam columns get that factor and its path rows exp(+j k_b z_i)
        (docs/theory/beam.md §9.9); the beam voltages of the copies add.
        Ports are joined as for the reduced models (the faces that face each
        other, checked mode by mode).  The result is saved in
        ``fds/foms/concat/``.
        """
        from cavsim3d.solvers.concatenation import (ConcatenatedSystem, chain_placement,
                                                    netlist_instances, netlist_joins)
        import cavsim3d.utils.printing as pr
        fds = self._fds_ref
        setup = fds.beam_setup
        a = _beam.axis_index(setup.axis)
        tildes = {}
        for base in self._components:
            t = self._section_tilde(base)
            if (t is None or t.get('port_modes') is None or t.get('zref') is None
                    or not t.get('ports')):
                raise RuntimeError(
                    f"Part '{base}' has no beam results to join: solve the project with "
                    "the beam (proj.fds.solve()).")
            tildes[base] = t
        bases = list(tildes)
        freqs = np.asarray(tildes[bases[0]]['frequencies'])
        for b in bases[1:]:
            f = np.asarray(tildes[b]['frequencies'])
            if len(f) != len(freqs) or not np.allclose(f, freqs, rtol=1e-9, atol=0):
                raise RuntimeError(
                    f"Parts '{bases[0]}' and '{b}' were solved at different frequencies: "
                    "the join needs the same samples. Solve the project again.")

        instances = netlist_instances(self._assembly)
        structures = []
        for iname, _key, base in instances:
            t = tildes[base]
            modes: Dict[str, Dict[int, Any]] = {}
            for port, m in t['port_modes']:
                modes.setdefault(port, {})[int(m)] = None
            st = ReducedStructure(Ard=np.zeros((0, 0)), Brd=np.zeros((0, len(t['port_modes']))),
                                  ports=list(modes), port_modes=modes, domain=iname, r=0,
                                  n_full=0, is_full_order=True)
            st.port_geometry = t['ports']
            st.port_fingerprints = {p: {int(m): v for m, v in d.items()}
                                    for p, d in (t.get('fingerprints') or {}).items()}
            st.base_domain = base
            structures.append(st)
        connections = netlist_joins(self._assembly, structures,
                                    [k for _i, k, _b in instances])
        shifts = chain_placement(structures, connections)
        for (iname, _key, base), shift in zip(instances, shifts):
            solved = _beam.BeamSetup.from_dict(tildes[base]['setup'])
            if not solved.same_lines(setup.shifted(shift)):
                raise RuntimeError(
                    f"Part '{base}' was solved with the beams at other places than copy "
                    f"'{iname}' needs: solve the project again (proj.fds.solve()).")

        def zref_of(i):
            t = tildes[instances[i][2]]
            rows = {(p, int(m)): r for r, (p, m) in enumerate(t['port_modes'])}
            return lambda port, mode, f: complex(
                t['zref'][int(np.argmin(np.abs(freqs - f))), rows[(port, int(mode))]])

        lookups = [zref_of(i) for i in range(len(instances))]
        concat = ConcatenatedSystem(structures=structures, solver_ref=fds)
        concat.define_connections(connections)

        blocks, block_modes, col_phase, row_phase = [], [], [], []
        k_over_w = [1.0 / (l.beta * _beam.c0) for l in setup.sources]
        kp_over_w = [1.0 / (l.beta * _beam.c0) for l in setup.paths]
        w = 2 * np.pi * freqs
        for i, (iname, _key, base) in enumerate(instances):
            t = tildes[base]
            blocks.append(np.asarray(t['data']))
            block_modes.append([concat.port_mode_map[(i, p, int(m))] for p, m in t['port_modes']])
            z = float(shifts[i][a])
            col_phase.append(np.exp(-1j * np.outer(w, k_over_w) * z))
            row_phase.append(np.exp(1j * np.outer(w, kp_over_w) * z))
        pairs = []
        for (sa, pa), (sb, pb) in concat.connections:
            n = concat.port_to_mode_range[(sa, pa)][1]
            pairs += [(concat.port_mode_map[(sa, pa, m)], concat.port_mode_map[(sb, pb, m)])
                      for m in range(n)]
        external = list(concat._external_port_modes)
        St = _beam.join_s_tilde(blocks, block_modes, pairs, external, col_phase, row_phase)

        labels, numbers = [], {}
        ext = [concat._global_to_local[g] for g in external]
        for s_idx, port, m in ext:
            n = numbers.setdefault((s_idx, port), len(numbers) + 1)
            labels.append(f"{n}({m + 1})")
        Zref = np.array([np.diag([lookups[s](p, m, f) for (s, p, m) in ext]) for f in freqs])
        Zt = _beam.z_tilde_from_s_tilde(St, Zref)
        rows = labels + setup.path_labels
        cols = labels + setup.source_labels
        n_ext = len(ext)
        concat.frequencies = freqs
        concat._S_matrix = St[:, :n_ext, :n_ext].copy()
        concat._Z_matrix = Zt[:, :n_ext, :n_ext].copy()
        concat._beam = {
            'S_tilde': St, 'Z_tilde': Zt, 'rows': rows, 'cols': cols, 'frequencies': freqs,
            'names': {lab: line.name for lab, line in zip(setup.path_labels, setup.paths)},
            'setup': setup.to_dict(), 'fingerprint': setup.fingerprint(),
            'summary': {'joined': [iname for iname, _k, _b in instances],
                        'shift': {iname: [float(v) for v in s]
                                  for (iname, _k, _b), s in zip(instances, shifts)}}}
        concat._scattering_join = True
        concat._invalidate_cache()
        if announce:
            pr.milestone(f"Joined {len(instances)} part copies through their generalised "
                         f"scattering matrices (beam): {n_ext} external port mode(s), "
                         f"{len(freqs)} frequencies (those of the full-order solve).")
        self._concat_cache = concat
        concat.save(self._root / "fds" / "foms" / "concat")
        return concat


class NetlistROMs:
    """Per-component ROM stage of an assembly netlist (see NetlistFOMs)."""

    def __init__(self, assembly, components, fds_ref, fom_config, tol):
        self._assembly = assembly
        self._components = components
        self._fds_ref = fds_ref
        self._config = fom_config
        self._tol = tol
        self._concat_cache = None

    @property
    def keys(self) -> List[str]:
        return list(self._components.keys())

    def __len__(self) -> int:
        return len(self._components)

    def __repr__(self) -> str:
        return f"NetlistROMs([{', '.join(self._components.keys())}])"

    @property
    def _roms_dir(self) -> Path:
        return Path(self._fds_ref._project_path) / "fds" / "foms" / "roms"

    def concatenate(self):
        """Couple the netlist: expand repeat counts, load each component's ROM
        (from this project's ``fds/foms/roms`` or from its referenced project),
        validate the joins (port-mode counts, mode fingerprints, training
        bands) and return the coupled system (with .solve() / .reduce()).

        The coupled system is saved into the module project's standard
        location: ``<project>/fds/foms/roms/concat/`` (its sweep results too,
        when it is solved).
        """
        from cavsim3d.solvers.concatenation import ConcatenatedSystem
        concat_dir = self._roms_dir / "concat"
        # results of an earlier coupling must not pass for this one's
        shutil.rmtree(concat_dir, ignore_errors=True)
        self._concat_cache = ConcatenatedSystem.from_flat_roms(
            self._assembly, self._roms_dir)
        self._concat_cache._save_dir = concat_dir
        self._attach_beam(self._concat_cache)
        try:
            self._concat_cache.save(concat_dir)
        except Exception as e:
            warnings.warn(f"Could not save concatenated system: {e}",
                          UserWarning, stacklevel=2)
        return self._concat_cache

    @property
    def concat(self):
        """The concatenated system (also after reopening, with its saved sweep)."""
        if self._concat_cache is None:
            from cavsim3d.solvers.concatenation import ConcatenatedSystem
            concat_dir = self._roms_dir / "concat"
            if not (concat_dir / "metadata.json").exists():
                raise RuntimeError("No concatenated system yet: call "
                                   "roms.concatenate() first.")
            concat = ConcatenatedSystem.from_flat_roms(self._assembly, self._roms_dir)
            concat._save_dir = concat_dir
            concat.load_results(concat_dir)
            self._attach_beam(concat)
            concat._update_beam()
            self._concat_cache = concat
        return self._concat_cache

    def _attach_beam(self, concat) -> None:
        """With beams: join the parts' reduced beam columns at every solve of
        ``concat`` (docs/theory/beam_reduction.md §10.7).  Each copy of a part
        sees the beam's phase at its position along the axis."""
        fds = self._fds_ref
        setup = fds.beam_setup if fds is not None else None
        if setup is None:
            return
        from cavsim3d.solvers.concatenation import chain_placement
        from cavsim3d.rom.beam_reduction import ReducedBeamJoin
        structures = concat.structures
        missing = sorted({getattr(s, 'base_domain', s.domain) for s in structures
                          if getattr(s, 'reduced_beam', None) is None})
        if missing:
            import cavsim3d.utils.printing as pr
            pr.warning(
                f"Part(s) {', '.join(missing)} have no reduced beam column, so this joined "
                "model has the port results only. Solve the project with the beam and "
                "store_snapshots=True (an imported part: in its own project, then reduce it "
                "there), and reduce again (proj.fds.foms.reduce(tol)).")
            return
        a = _beam.axis_index(setup.axis)
        shifts = chain_placement(structures, concat.connections)
        for s, shift in zip(structures, shifts):
            solved = _beam.BeamSetup.from_dict(s.reduced_beam.setup)
            if not solved.same_lines(setup.shifted(shift)):
                raise RuntimeError(
                    f"Part '{getattr(s, 'base_domain', s.domain)}' was reduced with the beams "
                    f"at other places than copy '{s.domain}' needs: solve the project again "
                    "(proj.fds.solve()) and reduce again.")
        sections = [{'beam': s.reduced_beam, 'A': s.Ard, 'B': s.Brd, 'C': s.Crd,
                     'D': s.Drd, 'zref': getattr(s, 'impedance_func', None),
                     'zwave': getattr(s, 'wave_impedance_func', None),
                     'key': getattr(s, 'base_domain', s.domain)} for s in structures]
        concat._beam_join = ReducedBeamJoin(
            concat, sections, setup, shifts=[float(sh[a]) for sh in shifts],
            summary={'joined': [s.domain for s in structures],
                     'shift': {s.domain: [float(v) for v in sh]
                               for s, sh in zip(structures, shifts)}})
