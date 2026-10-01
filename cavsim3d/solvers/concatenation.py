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
from __future__ import annotations

from typing import List, Tuple, Dict, Optional, Callable, Union, Any, Literal
import time
import re
import numpy as np
import scipy.linalg as sl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cavsim3d.solvers.nedelec import hcurl_flags, kind_of

from ngsolve import (
    Norm, curl, BoundaryFromVolumeCF, GridFunction, HCurl, Mesh, VOL
)
from ngsolve.webgui import Draw

from cavsim3d.solvers.eigen_mixin import ConcatEigenMixin
from cavsim3d.core.constants import Z0, mu0, MIN_EIGENVALUE
from cavsim3d.solvers.base import BaseEMSolver
from cavsim3d.utils.plot_mixin import PlotMixin
from cavsim3d.rom.structures import ReducedStructure
from cavsim3d.core.persistence import H5Serializer, ProjectManager
from cavsim3d.utils.names import region_pattern
from cavsim3d.solvers.options import (FOM_SOLVE_OPTIONS, REDUCED_SOLVE_OPTIONS,
                                      check_solve_options, validate_sweep)
import h5py
import json
from pathlib import Path
from datetime import datetime
import cavsim3d.utils.printing as pr
from cavsim3d.geometry.base import _display_webgui_fallback


# Connection specification: ((struct_idx, port_name), (struct_idx, port_name))
Conn = Tuple[Tuple[int, str], Tuple[int, str]]
ConnSigns = Tuple[float, float]  # (signA, signB) e.g. (+1,-1)

_AXIS_IDX = {'X': 0, 'Y': 1, 'Z': 2}


def _facing_port(struct, axis_idx: int, direction: int) -> Optional[str]:
    """Port of ``struct`` whose outward normal points along ``direction`` x axis.

    Among several such ports the one furthest out along the axis is taken.
    None when the structure has no saved port positions.
    """
    best = None
    for p, g in (getattr(struct, 'port_geometry', None) or {}).items():
        if p not in struct.ports:
            continue
        try:
            facing = direction * float(g['normal'][axis_idx])
            reach = direction * float(g['center'][axis_idx])
        except (KeyError, TypeError, IndexError):
            continue
        if facing > 0.9 and (best is None or reach > best[1]):
            best = (p, reach)
    return best[0] if best else None


def _n_propagating(geom: dict, fmax_hz: float, eps: float = 1.0,
                   mu: float = 1.0) -> Optional[int]:
    """Waveguide modes above cutoff at ``fmax_hz`` for a circular, rectangular
    or coaxial port (coaxial higher modes approximated); None if unknown."""
    from scipy.special import jn_zeros, jnp_zeros
    from cavsim3d.core.constants import c0
    k = 2 * np.pi * fmax_hz * np.sqrt(eps * mu) / c0
    typ = str(geom.get('type') or '').lower()
    if typ == 'circular' and geom.get('radius'):
        R = float(geom['radius'])
        count = 0
        for m in range(40):
            te = jnp_zeros(m, 20) / R          # TE_mn (excludes the trivial zero)
            tm = jn_zeros(m, 20) / R           # TM_mn
            n_m = int(np.sum(te < k) + np.sum(tm < k))
            if n_m == 0 and m > 0:
                # From m = 1 on, the lowest cutoff of each order rises with m.
                # m = 0 is passed over: its lowest (TM01) is above TE11's.
                break
            count += n_m * (1 if m == 0 else 2)
        return count
    if typ == 'rectangular' and geom.get('a') and geom.get('b'):
        a, b = float(geom['a']), float(geom['b'])
        mmax, nmax = int(k * a / np.pi) + 1, int(k * b / np.pi) + 1
        count = 0
        for m in range(mmax + 1):
            for n in range(nmax + 1):
                if (m, n) == (0, 0):
                    continue
                if np.pi * np.hypot(m / a, n / b) < k:
                    count += 1 + (1 if m > 0 and n > 0 else 0)   # TE (+ TM)
        return count
    if typ == 'coaxial' and geom.get('radius') and geom.get('inner_radius'):
        ro, ri = float(geom['radius']), float(geom['inner_radius'])
        count = 1                                                  # TEM
        m = 1
        while 2 * m / (ro + ri) < k:                               # TE_m1 (approx.)
            count += 2
            m += 1
        if np.pi / (ro - ri) < k:                                  # TM_01 (approx.)
            count += 1
        return count
    return None


def _warn_unresolved_join_modes(structures, connections) -> None:
    """Warn when a join carries fewer modes than propagate at the band's top.

    A mode that is not carried sees a magnetic wall at the join and is fully
    reflected there, so the coupled result is wrong wherever it propagates.
    """
    import warnings
    seen = set()
    for (ia, pa), (ib, pb) in connections:
        for s, p in ((structures[ia], pa), (structures[ib], pb)):
            key = (getattr(s, 'base_domain', s.domain), p)
            if key in seen:
                continue
            seen.add(key)
            geom = (getattr(s, 'port_geometry', None) or {}).get(p)
            band = getattr(s, 'training_band', None)
            if not geom or not band:
                continue
            media = (getattr(s, 'port_media', None) or {}).get(p, {})
            n_prop = _n_propagating(geom, float(band['fmax_GHz']) * 1e9,
                                    media.get('eps', 1.0), media.get('mu', 1.0))
            n_have = len(s.port_modes.get(p, {}))
            if n_prop and n_have < n_prop:
                warnings.warn(
                    f"Join at '{key[0]}' {p}: {n_have} port mode(s) carried, but "
                    f"{n_prop} propagate below {band['fmax_GHz']:g} GHz. The others "
                    f"are reflected at the join. Solve that part with "
                    f"nportmodes={{'{p}': {n_prop}, ...}}.",
                    UserWarning, stacklevel=3)


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

    Instances are obtained from ``concatenate()`` on a FOM or ROM collection,
    which wires up the interface connections and couples the systems.  There is
    no need to construct one directly or to call :meth:`couple` afterwards.

    Examples
    --------
    >>> # Reduce each domain, then couple the reduced models
    >>> roms = proj.fds.foms.reduce(tol=1e-6)
    >>> concat = roms.concatenate()
    >>>
    >>> # Solve and visualise the unified field
    >>> concat.solve(fmin=1, fmax=10, nsamples=1000)
    >>> concat.plot_field(freq_idx=50)  # Shows entire structure

    >>> # Full-order equivalent (validation; builds dense matrices)
    >>> concat = proj.fds.foms.concatenate()
    """

    DEFAULT_MIN_EIGENVALUE = MIN_EIGENVALUE  # omega^2 of 1 MHz: below is static
    ITERATIVE_SIZE_THRESHOLD = 10000

    def __init__(
        self,
        structures: List[ReducedStructure],
        mesh: Optional[Mesh] = None,
        fes: Optional[HCurl] = None,
        port_impedance_func: Optional[Callable[[str, int, float], complex]] = None,
        port_wave_impedance_func: Optional[Callable[[str, int, float], complex]] = None,
        solver_ref: Any = None,
    ):
        super().__init__()

        self.structures = structures
        self.n_structures = len(structures)
        self._port_impedance_func = port_impedance_func or self._default_impedance
        # Wave impedance the port modes were normalised to. Needed so
        # _compute_s_from_z can rescale Z into the reported (line) reference;
        # without it the concat's Z stays in the wave normalisation.
        self._port_wave_impedance_func = port_wave_impedance_func
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
        # Loss operators of the coupled system (None when lossless):
        #   (A + j w C - w^2 (I - j D)) x = w B u
        self.C_coupled: Optional[np.ndarray] = None
        self.D_coupled: Optional[np.ndarray] = None

        # Caches
        self._resonant_mode_cache = {}
        self._interface_scale_cache = {}

        # DOF mapping caches (built on first use)
        self._domain_dofs: Optional[Dict[int, List[int]]] = None
        self._interface_dofs: Optional[set] = None
        self._interface_pairs: Optional[Dict[Tuple[int, int], set]] = None
        self._local_to_global_maps: Optional[Dict[int, Dict[int, int]]] = None

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

    # =========================================================================
    # Construction from a netlist's flat fds/foms/roms tree
    # =========================================================================

    @classmethod
    def from_flat_roms(cls, assembly, roms_dir,
                       from_port: str = "port2", to_port: str = "port1"):
        """Build the coupled system from a netlist's flat ``fds/foms/roms`` tree.

        Every unique section already lives as a domain in ``roms_dir``
        (``matrices/A_r_<domain>.h5`` + a merged ``structures.json``).  This
        loads them once, expands the assembly netlist (respecting repeat counts
        ``n`` and sub-assemblies), makes a lightweight per-instance copy that
        shares the reduced operators, wires consecutive connections, validates
        the joins and couples.
        """
        from cavsim3d.rom.reduction import load_reduced_structures

        roms_dir = Path(roms_dir)
        loaded, _ = load_reduced_structures(roms_dir)
        by_domain = {s.domain: s for s in loaded}

        # ---- flatten the netlist into an ordered instance list --------------
        instances = []          # (instance_name, component_key, base_name)

        def _flatten(asm, prefix=""):
            for key in asm._component_order:
                entry = asm._components[key]
                comp = entry.geometry
                if entry.metadata.get("flip"):
                    raise NotImplementedError(
                        f"'{key}' is flipped, but it is coupled through port modes "
                        "(imported or repeated). flip is supported for parts glued "
                        "into one mesh; a coupled part must be solved in the "
                        "orientation it is used.")
                n = int(entry.metadata.get("n", 1))
                for i in range(n):
                    suffix = f"_{i + 1}" if n > 1 else ""
                    iname = f"{prefix}{key}{suffix}"
                    if isinstance(comp, type(asm)):
                        _flatten(comp, prefix=iname + "/")
                    else:
                        instances.append((iname, key, entry.base_name))

        _flatten(assembly)
        if not instances:
            raise ValueError("Assembly contains no components.")

        def _instance_copy(s, domain):
            c = ReducedStructure(
                Ard=s.Ard, Brd=s.Brd, ports=list(s.ports),
                port_modes=s.port_modes, domain=domain,
                r=s.r, n_full=s.n_full, is_full_order=s.is_full_order,
                W=s.W, Q_L_inv=s.Q_L_inv, fes=s.fes, mesh=s.mesh,
                Crd=s.Crd, Drd=s.Drd)
            for attr in ("port_fingerprints", "training_band", "impedance_func",
                         "wave_impedance_func", "port_geometry", "port_media",
                         "mesh_source"):
                if hasattr(s, attr):
                    setattr(c, attr, getattr(s, attr))
            # Keep the SOURCE (base) domain so a per-section field can find its
            # saved mesh (mesh/mesh_<base>.pkl) for 3D reconstruction.
            c.base_domain = s.domain
            return c

        structures = []
        inst_keys = []
        for iname, key, base in instances:
            if base not in by_domain:
                raise KeyError(
                    f"Section '{base}' has no reduced model in {roms_dir}. "
                    "Did the ROM stage (foms.reduce) run for it?")
            structures.append(_instance_copy(by_domain[base], iname))
            inst_keys.append(key)

        # ---- consecutive connections ------------------------------------------
        # Parts follow the list order along the main axis, so part i joins
        # part i+1 through the port of i that faces +axis and the port of i+1
        # that faces -axis -- whatever those ports are called.  Ports named
        # explicitly (align_port) are used as given; without saved port
        # positions the default names (port2 -> port1) apply.
        axis_idx = _AXIS_IDX.get(str(getattr(assembly, "main_axis", "Z")).upper(), 2)
        conns = {(c.from_key, c.to_key): c for c in getattr(assembly, "_connections", [])}
        connections = []
        for i in range(len(structures) - 1):
            c = conns.get((inst_keys[i], inst_keys[i + 1]))
            fp = tp = None
            if c is not None and c.explicit:
                fp, tp = c.from_port, c.to_port
            else:
                fp = _facing_port(structures[i], axis_idx, +1)
                tp = _facing_port(structures[i + 1], axis_idx, -1)
                if fp is None or tp is None:
                    fp, tp = ((c.from_port, c.to_port) if c is not None
                              else (from_port, to_port))
            connections.append(((i, fp), (i + 1, tp)))
        _warn_unresolved_join_modes(structures, connections)

        concat = cls(structures=structures,
                     mesh=structures[0].mesh, fes=structures[0].fes)
        concat.define_connections(connections)
        concat.couple()
        # Where each section's saved mesh/FES live (project/mesh/{mesh,fes}_<base>.pkl),
        # so a 3D field can later be reconstructed per section for visualization.
        try:
            _root = Path(roms_dir).parents[2]
            concat._project_mesh_dir = _root / "mesh"
            # _project_path is a READ-ONLY property proxying the parent solver,
            # so it cannot be assigned here. Record the root separately; the
            # eigenmode auto-save falls back to it.
            concat._project_root = _root
        except Exception:
            concat._project_mesh_dir = None
            concat._project_root = None
        return concat

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
        kind = 'second'
        if self._solver_ref is not None:
            order = getattr(self._solver_ref, 'order', order)
            kind = getattr(self._solver_ref, 'nedelec', kind)
        
        # Try to get from first structure's fes
        for struct in self.structures:
            if struct.fes is not None:
                order = struct.fes.globalorder
                kind = kind_of(struct.fes)
                break

        # Determine Dirichlet BC label
        bc = 'default'
        if self._solver_ref is not None:
            bc = getattr(self._solver_ref, 'bc', bc)

        from ngsolve import HCurl
        self.fes = HCurl(self.mesh, order=order, complex=True, dirichlet=bc,
                         **hcurl_flags(kind))
        pr.debug(f"  Created unified FES: {self.fes.ndof} DOFs (order={order})")

    @property
    def project_sub_path(self) -> Path:
        """Relative path from project root for this concatenated system's data."""
        return self._solver_ref.project_sub_path / "concat"

    @property
    def _project_path(self):
        """Proxy project path from parent solver."""
        if self._solver_ref is not None:
            return getattr(self._solver_ref, '_project_path', None)
        return None

    def _validate_mode_counts(self) -> None:
        """Record a representative modes-per-port scalar.

        Per-port mode counts may now differ between ports and structures
        (e.g. a TEM port with one mode next to a TE port with several), so
        this no longer requires a uniform count.  ``_n_modes_per_port`` is
        kept only as a back-compat scalar (the most common per-port count);
        the authoritative per-port layout lives in ``port_to_mode_range`` and
        each structure's ``port_mode_pairs``.
        """
        if not self.structures:
            self._n_modes_per_port = 1
            return

        from collections import Counter
        counts = Counter()
        for struct in self.structures:
            for _port, modes in self._struct_modes_by_port(struct).items():
                counts[len(modes)] += 1
        # Most common per-port mode count (fallback to 1).
        self._n_modes_per_port = counts.most_common(1)[0][0] if counts else 1

    @staticmethod
    def _struct_modes_by_port(struct) -> Dict[str, list]:
        """{port: [mode_idx, ...]} for a structure, honouring per-port counts."""
        from collections import defaultdict
        by_port = defaultdict(list)
        for port, mode_idx in struct.port_mode_pairs:
            by_port[port].append(mode_idx)
        return by_port

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
            # Group the structure's (port, mode) pairs by port, preserving order,
            # so each port may carry a different number of modes.
            from collections import defaultdict
            modes_by_port: "dict[str, list]" = defaultdict(list)
            for port, mode_idx in struct.port_mode_pairs:
                modes_by_port[port].append(mode_idx)

            for port, mode_list in modes_by_port.items():
                start_idx = offset
                for mode_idx in mode_list:
                    self.port_mode_map[(struct_idx, port, mode_idx)] = offset
                    self._global_to_local[offset] = (struct_idx, port, mode_idx)
                    offset += 1
                self.port_to_mode_range[(struct_idx, port)] = (start_idx, len(mode_list))

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
    # DOF Mapping for Field Reconstruction
    # =========================================================================

    def _build_domain_dof_maps(self) -> None:
        """
        Build and cache DOF mappings for all domains.
        
        Creates:
        - _domain_dofs: dict mapping struct_idx -> list of global DOF indices
        - _interface_dofs: set of DOFs shared between multiple domains
        - _interface_pairs: dict mapping (i,j) -> set of shared DOFs
        - _local_to_global_maps: dict mapping struct_idx -> {local_dof: global_dof}
        """
        if self._domain_dofs is not None:
            return  # Already built
        
        if self.mesh is None or self.fes is None:
            raise ValueError("Cannot build DOF maps: mesh or fes not available")
        
        self._domain_dofs = {}
        self._local_to_global_maps = {}
        all_dofs_by_domain = {}
        
        # Resolve which mesh materials belong to each domain
        def _get_domain_mats(domain_name):
            """Get the set of mesh material names for a domain."""
            # Try via solver chain: MOR -> FDS -> _get_domain_mesh_materials
            solver = self._solver_ref
            if solver is not None:
                fds = getattr(solver, 'solver', solver)  # MOR.solver or FDS itself
                if hasattr(fds, '_get_domain_mesh_materials'):
                    mats = set(fds._get_domain_mesh_materials(domain_name))
                    if mats != {domain_name}:
                        return mats
            # Fallback: scan mesh for materials prefixed with domain_name/
            all_mats = set(self.mesh.GetMaterials())
            matched = {m for m in all_mats if m == domain_name or m.startswith(domain_name + '/')}
            return matched if matched else {domain_name}

        for struct_idx, struct in enumerate(self.structures):
            domain_name = struct.domain
            domain_mats = _get_domain_mats(domain_name)
            domain_mats_str = {str(m) for m in domain_mats}
            dofs = set()

            # Collect global DOFs belonging to this domain's elements
            for el in self.mesh.Elements(VOL):
                if str(el.mat) not in domain_mats_str:
                    continue
                for g_dof in self.fes.GetDofNrs(el):
                    if g_dof >= 0:
                        dofs.add(g_dof)

            self._domain_dofs[struct_idx] = sorted(dofs)
            all_dofs_by_domain[struct_idx] = dofs
            pr.debug(f"    Domain {struct_idx} ({domain_name}): {len(dofs)} global DOFs")
        
        # Find interface DOFs (shared between domains)
        self._interface_dofs = set()
        self._interface_pairs = {}
        
        for i in range(self.n_structures):
            for j in range(i + 1, self.n_structures):
                shared = all_dofs_by_domain[i] & all_dofs_by_domain[j]
                if shared:
                    self._interface_dofs.update(shared)
                    self._interface_pairs[(i, j)] = shared
                    self._interface_pairs[(j, i)] = shared
        
        pr.debug(f"  DOF mapping complete: {len(self._interface_dofs)} interface DOFs")

    def _get_local_to_global_dof_map(self, struct_idx: int) -> Dict[int, int]:
        """
        Get mapping from local (structure) DOF indices to global DOF indices.
        
        Returns dict: local_dof -> global_dof
        """
        self._build_domain_dof_maps()
        return self._local_to_global_maps[struct_idx]

    def _invalidate_dof_cache(self) -> None:
        """Invalidate DOF mapping caches."""
        self._domain_dofs = None
        self._interface_dofs = None
        self._interface_pairs = None
        self._local_to_global_maps = None
        self._interface_scale_cache = {}

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

            # Couple all modes of the connected (interface) ports; both sides
            # carry the same number of modes (checked in _validate_connection).
            n_modes_conn = self.port_to_mode_range[(sA_idx, pA)][1]
            for mode_idx in range(n_modes_conn):
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

        # Build external port naming.  Number the distinct external ports
        # sequentially (in their global order) so each external (struct, port)
        # gets one port number regardless of how many modes it has.
        self._external_port_mode_names = []
        self._external_port_mode_map = {}
        _port_number: Dict[Tuple[int, str], int] = {}
        _next_num = 1
        for global_idx in self._external_port_modes:
            struct_idx, orig_port, mode_idx = self._global_to_local[global_idx]
            key = (struct_idx, orig_port)
            if key not in _port_number:
                _port_number[key] = _next_num
                _next_num += 1
            new_name = f"port{_port_number[key]}({mode_idx + 1})"
            self._external_port_mode_names.append((orig_port, mode_idx))
            self._external_port_mode_map[new_name] = (struct_idx, orig_port, mode_idx)
        self._n_external_ports = len(_port_number)

        # Reverse map so the structure-qualified keys used by
        # _port_mode_order ("s0:port2") can be resolved back to the external
        # port name ("port2(1)").
        self._local_to_external = {
            loc: name for name, loc in self._external_port_mode_map.items()
        }

        # Ordered (port_key, mode_idx) for the coupled Z/S matrix columns, used
        # by the base _build_dicts to label parameters correctly when ports
        # have different numbers of modes.  The composite key keeps each
        # external (structure, port) distinct so it maps to one port number.
        self._port_mode_order = [
            (f"s{si}:{op}", mi)
            for gidx in self._external_port_modes
            for (si, op, mi) in (self._global_to_local[gidx],)
        ]

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
        domA = getattr(self.structures[sA_idx], 'domain', sA_idx)
        domB = getattr(self.structures[sB_idx], 'domain', sB_idx)
        if n_modes_A != n_modes_B:
            raise ValueError(
                "The number of port modes must match at connected interfaces. "
                f"Interface '{domA}'.{pA} has {n_modes_A} mode(s) but "
                f"'{domB}'.{pB} has {n_modes_B} mode(s). "
                "Re-run/import the connected sections with the same nportmodes "
                "on the interface ports."
            )
        # Per-mode fit-check: mode k on both interfaces must be the SAME physical
        # mode (type + cutoff kc + polarization).  Matching cross-section
        # dimensions give matching kc; polarization guards against coupling e.g.
        # a cos-oriented degenerate mode to its sin partner (numeric modes).
        fpA = getattr(self.structures[sA_idx], 'port_fingerprints', None)
        fpB = getattr(self.structures[sB_idx], 'port_fingerprints', None)
        if fpA and fpB and pA in fpA and pB in fpB:
            for m in range(n_modes_A):
                a = fpA[pA].get(m)
                b = fpB[pB].get(m)
                if not a or not b:
                    continue
                bad = None
                if str(a.get("type")) != str(b.get("type")):
                    bad = f"type {a.get('type')} vs {b.get('type')}"
                elif list(a.get("indices", [])) != list(b.get("indices", [])):
                    bad = f"indices {a.get('indices')} vs {b.get('indices')}"
                else:
                    ka, kb = float(a.get("kc", 0)), float(b.get("kc", 0))
                    if abs(ka - kb) > 1e-3 * max(abs(ka), abs(kb), 1e-30):
                        bad = f"cutoff kc {ka:.6g} vs {kb:.6g} (cross-sections differ)"
                    elif abs(float(a.get("pol", 0)) - float(b.get("pol", 0))) > 1e-3:
                        bad = (f"polarization {np.degrees(a.get('pol',0)):.1f}deg vs "
                               f"{np.degrees(b.get('pol',0)):.1f}deg")
                if bad:
                    raise ValueError(
                        "Interface port modes do not correspond at "
                        f"'{domA}'.{pA} <-> '{domB}'.{pB}, mode {m}: {bad}. "
                        "The connected cross-sections must share the same mode "
                        "basis (matching dimensions and, for degenerate modes, "
                        "the same polarization/orientation convention)."
                    )

    def _validate_bands(self) -> None:
        """Warn/error on incompatible ROM training bands across structures.

        Each ROM is only valid over the frequency window its snapshots covered;
        coupling is trustworthy only in the INTERSECTION of the sections'
        bands.  Empty intersection -> error; otherwise record it for solve() to
        warn when a sweep extrapolates beyond it.
        """
        bands = [getattr(s, 'training_band', None) for s in self.structures]
        bands = [b for b in bands if b]
        self._training_band = None
        if len(bands) < 2:
            if bands:
                self._training_band = (bands[0]["fmin_GHz"], bands[0]["fmax_GHz"])
            return
        lo = max(b["fmin_GHz"] for b in bands)
        hi = min(b["fmax_GHz"] for b in bands)
        if lo >= hi:
            ranges = ", ".join(f"[{b['fmin_GHz']:.4g}, {b['fmax_GHz']:.4g}] GHz"
                               for b in bands)
            raise ValueError(
                "Sections were reduced over disjoint frequency bands and cannot "
                f"be coupled: {ranges}. A ROM is only valid over its training "
                "band; reduce all sections over a common (overlapping) range."
            )
        self._training_band = (lo, hi)
        widest = max(b["fmax_GHz"] - b["fmin_GHz"] for b in bands)
        if (hi - lo) < 0.5 * widest:
            import warnings
            warnings.warn(
                f"Section training bands overlap only in [{lo:.4g}, {hi:.4g}] GHz, "
                "much narrower than the sections' individual bands. Coupled "
                "results outside this window may be inaccurate or wrong.",
                UserWarning, stacklevel=2)

    def _validate_connections(self) -> None:
        """Validate all connections."""
        for j, ((sA, pA), (sB, pB)) in enumerate(self.connections):
            if sA == sB and pA == pB:
                raise ValueError(f"Connection {j} connects port '{pA}' to itself")

    def _build_incidence_matrix(self) -> np.ndarray:
        """Build incidence matrix F for Kirchhoff coupling.

        One constraint column per connected (interface) port-mode; the number
        of modes can differ from connection to connection.
        """
        n_int = self._n_internal
        # Total mode-connections = sum of each connection's mode count.
        n_mode_connections = sum(
            self.port_to_mode_range[(sA_idx, pA)][1]
            for (sA_idx, pA), (_sB_idx, _pB) in self.connections
        )

        F = np.zeros((n_int, n_mode_connections))
        int_pos = {g: i for i, g in enumerate(self._internal_port_modes)}

        col = 0
        for ((sA_idx, pA), (sB_idx, pB)), (sgnA, sgnB) in zip(
            self.connections, self._connection_signs
        ):
            n_modes_conn = self.port_to_mode_range[(sA_idx, pA)][1]
            for mode_idx in range(n_modes_conn):
                gA = self.port_mode_map[(sA_idx, pA, mode_idx)]
                gB = self.port_mode_map[(sB_idx, pB, mode_idx)]
                F[int_pos[gA], col] = sgnA
                F[int_pos[gB], col] = sgnB
                col += 1

        return F

    # =========================================================================
    # Coupling
    # =========================================================================

    def couple(self, rcond_null: float = 1e-12) -> "ConcatenatedSystem":
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

        # Physical fit-check across sections: the ROMs must share a compatible
        # (overlapping) training frequency band, else coupling is meaningless.
        self._validate_bands()

        _t_couple = time.time()

        # Block-diagonal assembly of uncoupled structures
        A_blocks = [np.asarray(s.Ard) for s in self.structures]
        B_blocks = [np.asarray(s.Brd) for s in self.structures]

        A_blk = sl.block_diag(*A_blocks).astype(complex, copy=False)
        B_blk = sl.block_diag(*B_blocks).astype(complex, copy=False)

        # Loss operators (zero blocks for lossless sections)
        lossy = any(getattr(s, 'is_lossy', False) for s in self.structures)

        def _loss_blk(attr):
            blocks = [np.asarray(getattr(s, attr)) if getattr(s, attr, None) is not None
                      else np.zeros((s.r, s.r)) for s in self.structures]
            return sl.block_diag(*blocks)

        C_blk = _loss_blk('Crd') if lossy else None
        D_blk = _loss_blk('Drd') if lossy else None

        # Permute to [internal | external]
        B_perm = B_blk @ self._permutation.T
        B_int = B_perm[:, :self._n_internal]
        B_ext = B_perm[:, self._n_internal:]

        # Build and apply Kirchhoff constraints
        F = self._build_incidence_matrix()
        C = B_int @ F

        if C.shape[1] == 0:
            # No internal connections (e.g. a single-section netlist). There
            # are no Kirchhoff constraints, so the projector K and the null
            # space basis M are both the identity and KM leaves the
            # block-diagonal system untouched (verified: null_space of a (0,n)
            # matrix IS exactly I_n, and K reduces to I).
            #
            # Building them explicitly would hand 0-sized arrays to pinvh and
            # null_space. SciPy >= 1.15 tolerates that; every Python 3.9 build
            # (SciPy <= 1.13) fails deep inside LAPACK, first in eigh
            #   _flapack.error: (il>=1&&il<=n) failed ... zheevr:il=1
            # and then in the SVD workspace query
            #   ValueError: Internal work array size computation failed: -5
            KM = np.eye(A_blk.shape[0])
        else:
            # The constraint-satisfying subspace is simply a basis for
            # null(C^H). The orthogonal projector I - C(C^H C)^-1 C^H that used
            # to be applied here acts as the identity on that basis (C^H N = 0
            # => K_perp N = N, verified to ~5e-16), so building it cost a dense
            # n x n product and a pinvh for nothing.
            # C = B_int F is real, so its null space has a REAL orthonormal
            # basis.  Keeping KM real makes KM^H = KM^T: the projection is then
            # a Galerkin projection of the bilinear (complex-symmetric) lossy
            # system as well as of the Hermitian lossless one.
            C_real = np.real(C) if np.allclose(np.imag(C), 0.0) else C
            KM = sl.null_space(C_real.T.conj(), rcond=rcond_null)
            if KM.size == 0:
                raise RuntimeError("Null space empty; constraints overconstrained.")

        # Project system onto constraint-satisfying subspace
        self.A_coupled = KM.T.conj() @ A_blk @ KM
        self.B_coupled = KM.T.conj() @ B_ext
        self.W_coupled = KM.astype(complex, copy=False)

        # Ensure Hermitian symmetry
        self.A_coupled = 0.5 * (self.A_coupled + self.A_coupled.T.conj())

        if lossy:
            self.C_coupled = KM.T @ C_blk @ KM
            self.D_coupled = KM.T @ D_blk @ KM
        else:
            self.C_coupled = self.D_coupled = None

        pr.info(f"\nCoupled unified system: {A_blk.shape[0]} -> {self.A_coupled.shape[0]} DOFs")
        pr.debug(f"  External port-modes: {self._n_external}")
        pr.debug(f"  Internal port-modes (eliminated): {self._n_internal}")
        pr.debug(f"  Connections: {self.n_connections}")

        from cavsim3d.utils.timing import get_timing_registry
        get_timing_registry().record(
            "couple", time.time() - _t_couple, category="CONCAT",
            full_dofs=int(A_blk.shape[0]), reduced_dofs=int(self.A_coupled.shape[0]),
            n_connections=self.n_connections, n_external=self._n_external,
        )

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

        from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
        eigenvalues, eigenvectors = np.linalg.eigh(self.A_coupled)
        opts = dict(kwargs)
        n_modes = opts.pop('n_modes', None)
        eigs, vecs = FrequencyDomainSolver._filter_eigenvalues(eigenvalues, eigenvectors, **opts)
        # by default only the modes near the training band (an explicit
        # min_eigenvalue replaces that window)
        window = (self._training_window() if opts.get('filter_static', True)
                  and opts.get('min_eigenvalue') is None else None)
        if window is not None:
            keep = (np.real(eigs) >= window[0]) & (np.real(eigs) <= window[1])
            eigs, vecs = eigs[keep], vecs[:, keep]
        if n_modes is not None:
            eigs, vecs = eigs[:n_modes], vecs[:, :n_modes]
        res = (eigs, vecs)

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
        # one coupled system: domain/shift/return options do not apply
        for k in ("return_eigenvalues", "sigma", "domain", "source"):
            kwargs.pop(k, None)
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
        # A netlist concat proxies _project_path from a parent solver it does
        # not have; from_flat_roms records the project root instead.
        if not kwargs.get("path") and not self._project_path:
            root = getattr(self, "_project_root", None)
            if root is not None:
                kwargs["path"] = Path(root) / "fds" / "foms" / "roms" / "concat" / "eigenmodes"
        try:
            self.save_eigenmodes(**kwargs)
        except (ValueError, Exception) as e:
            pr.warning(f"Could not auto-save eigenmodes for ConcatenatedSystem: {e}")

    # =========================================================================
    # BaseEMSolver Interface
    # =========================================================================

    @property
    def n_ports(self) -> int:
        return self._n_external

    @property
    def n_external_ports(self) -> int:
        # Distinct external ports (counted once regardless of their mode count).
        n = getattr(self, '_n_external_ports', None)
        if n is not None:
            return n
        return self._n_external // max(self._n_modes_per_port, 1)

    @property
    def n_modes_per_port(self) -> int:
        return self._n_modes_per_port

    @property
    def ports(self) -> List[str]:
        return list(self._external_port_mode_map.keys())

    def port_map(self) -> List[Dict[str, Any]]:
        """Where each external port of the joined model comes from.

        The ports where parts join are gone; the others are numbered part by
        part, in each part's own port order.

        Returns
        -------
        list of dict
            ``{'port', 'part', 'part_port', 'modes'}`` per external port, in
            matrix order: its name here, the part it belongs to (a copy of a
            repeated part is ``'<name>_<copy>'``), that part's own name for the
            port, and the number of modes it carries.
        """
        rows: Dict[str, Dict[str, Any]] = {}
        for name, (s, port, _mode) in self._external_port_mode_map.items():
            here = name.split("(")[0]
            row = rows.setdefault(here, {"port": here, "part": self.structures[s].domain,
                                         "part_port": port, "modes": 0})
            row["modes"] += 1
        return list(rows.values())

    def print_port_map(self) -> None:
        """Print :meth:`port_map` as a table."""
        print(f'{"port":8s}{"part":16s}{"its port":10s}modes')
        for row in self.port_map():
            print(f'{row["port"]:8s}{row["part"]:16s}{row["part_port"]:10s}{row["modes"]}')

    def _impedance_for(self, struct_idx: int, port: str, mode: int, freq: float) -> complex:
        """Port wave impedance, preferring the structure's OWN impedance
        function (attached by import/load — sections from different projects
        may carry different media/cross-sections)."""
        func = getattr(self.structures[struct_idx], 'impedance_func', None)
        if func is not None:
            return func(port, mode, freq)
        return self._port_impedance_func(port, mode, freq)

    def _resolve_external_port(self, port, mode: int = 0):
        """Accept either an external name ("port2(1)") or the composite key
        used by _port_mode_order ("s0:port2"), and return the external name."""
        if port in self._external_port_mode_map:
            return port
        if f"{port}({int(mode) + 1})" in self._external_port_mode_map:
            return f"{port}({int(mode) + 1})"
        m = re.match(r'^s(\d+):(.+)$', str(port))
        if m:
            loc = (int(m.group(1)), m.group(2), int(mode))
            return getattr(self, '_local_to_external', {}).get(loc)
        return None

    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        key = self._resolve_external_port(port, mode)
        if key is None:
            raise KeyError(f"Port '{port}' not found. Available: {self.ports}")
        struct_idx, orig_port, orig_mode = self._external_port_mode_map[key]
        return self._impedance_for(struct_idx, orig_port, orig_mode, freq)

    def _port_wave_impedance(self, port, mode: int, freq: float):
        """Wave impedance the coupled port basis inherited from its section.

        Resolved through the same external-port map as the reference impedance,
        so a section imported from another project keeps its own medium.
        """
        key = self._resolve_external_port(port, mode)
        if key is None:
            return None
        struct_idx, orig_port, orig_mode = self._external_port_mode_map[key]
        # Prefer the SECTION's own function: a concat rebuilt by from_flat_roms
        # gets no global wave-impedance func, but each reloaded structure
        # carries one. Testing the global one first skipped the rescale
        # entirely on the netlist path.
        func = (getattr(self.structures[struct_idx], 'wave_impedance_func', None)
                or self._port_wave_impedance_func)
        if func is None:
            return None
        try:
            return func(orig_port, orig_mode, freq)
        except Exception:
            return None

    def _get_impedance_matrix(self, freq: float) -> np.ndarray:
        Z0_diag = []
        for global_idx in self._external_port_modes:
            struct_idx, port_name, mode_idx = self._global_to_local[global_idx]
            Zw = self._impedance_for(struct_idx, port_name, mode_idx, freq)
            Z0_diag.append(Zw)
        return np.diag(Z0_diag)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: Union[str, Path]):
        """
        Save ConcatenatedSystem data to disk.
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
        for name, mat in (("C", getattr(self, 'C_coupled', None)),
                          ("D", getattr(self, 'D_coupled', None))):
            if mat is not None:
                with h5py.File(mat_path / f"{name}.h5", "a") as fl:
                    H5Serializer.save_dataset(fl, "data", mat)

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
                if "coupled_snapshots" in f:
                    del f["coupled_snapshots"]
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
    def load(cls, path: Union[str, Path], solver_ref=None) -> "ConcatenatedSystem":
        """Load ConcatenatedSystem from disk."""
        path = Path(path)
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Create skeleton
        cs = cls.__new__(cls)
        cs._solver_ref = solver_ref
        cs.C_coupled = cs.D_coupled = None
        cs.n_structures = metadata["n_structures"]
        cs.domains = metadata["domains"]
        cs.n_connections = metadata["n_connections"]
        cs._n_internal = metadata["n_internal"]
        cs._n_external = metadata["n_external"]
        cs._n_modes_per_port = metadata["n_modes_per_port"]
        
        # Initialize caches
        cs._resonant_mode_cache = {}
        cs._interface_scale_cache = {}
        cs._domain_dofs = None
        cs._interface_dofs = None
        cs._interface_pairs = None
        cs._local_to_global_maps = None
        
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
            for name in ("C", "D"):
                fp = mat_path / f"{name}.h5"
                val = None
                if fp.exists():
                    with h5py.File(fp, "r") as f:
                        val = H5Serializer.load_dataset(f["data"])
                setattr(cs, f"{name}_coupled", val)
        elif (path / "matrices.h5").exists():
            with h5py.File(path / "matrices.h5", "r") as f:
                cs.A_coupled = H5Serializer.load_dataset(f["A_coupled"])
                cs.B_coupled = H5Serializer.load_dataset(f["B_coupled"])
                cs.W_coupled = H5Serializer.load_dataset(f["W_coupled"])

        # 2. Load Z and S
        cs.frequencies = None
        cs._Z_matrix = None
        cs._S_matrix = None
        
        z_path = path / "z" / "z.h5"
        if not z_path.exists():
            z_path = path / "z.h5"
        if z_path.exists():
            with h5py.File(z_path, "r") as f:
                cs._Z_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None
                
        s_path = path / "s" / "s.h5"
        if not s_path.exists():
            s_path = path / "s.h5"
        if s_path.exists():
            with h5py.File(s_path, "r") as f:
                cs._S_matrix = H5Serializer.load_dataset(f["data"]) if "data" in f else None

        # 3. Load snapshots and frequencies
        cs._snapshots = {}
        snap_path = path / "snapshots" / "snapshots.h5"
        if not snap_path.exists():
            snap_path = path / "snapshots.h5"
        if snap_path.exists():
            with h5py.File(snap_path, "r") as f:
                if cs.frequencies is None:
                    cs.frequencies = H5Serializer.load_dataset(f["frequencies"]) if "frequencies" in f else None
                
                if "field_snapshots" in f:
                    group = f["field_snapshots"]
                    for domain in metadata.get("structure_domains", []):
                        if domain in group:
                            cs._snapshots[domain] = H5Serializer.load_dataset(group[domain])

        # Restore log path if it exists on disk
        if solver_ref and hasattr(solver_ref, '_project_path') and solver_ref._project_path:
            log_file = path / "solve.log"
            if log_file.exists():
                cs._log_path = str(log_file)

        return cs

    def load_results(self, path: Union[str, Path]) -> bool:
        """Attach the sweep (Z, S, frequencies, snapshots) saved in ``path``.

        Only results that fit this coupled system are taken (as many external
        port modes, snapshots of its size).  Returns True if attached.
        """
        path = Path(path)
        z_file = path / "z" / "z.h5"
        snap_file = path / "snapshots" / "snapshots.h5"
        if not (z_file.exists() and snap_file.exists()):
            return False
        with h5py.File(z_file, "r") as f:
            Z = H5Serializer.load_dataset(f["data"]) if "data" in f else None
        S = None
        if (path / "s" / "s.h5").exists():
            with h5py.File(path / "s" / "s.h5", "r") as f:
                S = H5Serializer.load_dataset(f["data"]) if "data" in f else None
        with h5py.File(snap_file, "r") as f:
            freqs = (H5Serializer.load_dataset(f["frequencies"])
                     if "frequencies" in f else None)
            snaps = (H5Serializer.load_dataset(f["coupled_snapshots"])
                     if "coupled_snapshots" in f else None)
        n_ext = self._n_external
        if (Z is None or freqs is None or np.ndim(Z) != 3 or Z.shape[1] != n_ext
                or Z.shape[0] != len(freqs)):
            return False
        if S is not None and np.shape(S) != np.shape(Z):
            S = None
        if snaps is not None and (np.ndim(snaps) != 3 or snaps.shape[0] != len(freqs)
                                  or snaps.shape[1] != self.A_coupled.shape[0]):
            snaps = None
        self._Z_matrix, self._S_matrix = Z, S
        self.frequencies = freqs
        self._snapshots = snaps
        self._invalidate_cache()
        return True

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
        """
        # 1. Merge config and kwargs (a full-order config may be reused: its
        # full-order options are accepted and have no effect here)
        cfg = (config or {}).copy()
        cfg.update(kwargs)
        check_solve_options(cfg, REDUCED_SOLVE_OPTIONS, also_accepted=FOM_SOLVE_OPTIONS,
                            where="concat.solve()")

        # 2. Extract core parameters with defaults
        fmin = fmin if fmin is not None else cfg.get('fmin')
        fmax = fmax if fmax is not None else cfg.get('fmax')
        nsamples = nsamples if nsamples is not None else cfg.get('nsamples', 100)

        if fmin is None or fmax is None:
            raise ValueError("fmin and fmax must be provided (either directly or via config).")
        nsamples = validate_sweep(fmin, fmax, nsamples)

        # 3. Extract other options from merged cfg
        compute_s_params = cfg.get('compute_s_params', True)
        solver_type = cfg.get('solver_type', 'auto')
        verbose = cfg.get('verbose')     # None: keep the console verbosity
        _prev_verbosity = pr.push_verbosity(verbose)

        # Start file logging
        _file_handler = None
        if self._solver_ref and hasattr(self._solver_ref, '_project_path') and self._solver_ref._project_path:
            project_path = Path(self._solver_ref._project_path)
            sub_path = "foms/concat"
            from cavsim3d.rom.reduction import ModelOrderReduction as _MOR
            if isinstance(self._solver_ref, _MOR):
                if self._solver_ref.n_domains == 1:
                    sub_path = "fom/rom/concat"
                else:
                    sub_path = "foms/roms/concat"
            log_dir = project_path / "fds" / sub_path
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = str(log_dir / "solve.log")
            _file_handler = pr.start_file_log(self._log_path)

        try:
            # 4. Requested frequency grid.  NOT assigned to self.frequencies
            # until we actually solve: an early return of cached results must
            # keep the grid those results were computed on.
            new_freqs = np.linspace(fmin, fmax, nsamples) * 1e9

            if self.A_coupled is None:
                raise ValueError("Must call couple() first")

            # Warn if the sweep extrapolates beyond the ROMs' shared training band.
            band = getattr(self, '_training_band', None)
            if band is not None:
                lo, hi = band
                if fmin < lo - 1e-9 or fmax > hi + 1e-9:
                    import warnings
                    warnings.warn(
                        f"Sweeping {fmin:.4g}-{fmax:.4g} GHz extends beyond the "
                        f"sections' shared training band [{lo:.4g}, {hi:.4g}] GHz; "
                        "reduced-order results outside it are extrapolated and "
                        "may be inaccurate or wrong.",
                        UserWarning, stacklevel=2)

            # --- Rerun protection ---
            has_results = (self._Z_matrix is not None)
            rerun = cfg.get('rerun', None)   # None: auto, True: force, False: keep stored

            # Check disk if in-memory is missing
            if not has_results and not rerun and self._solver_ref and getattr(self._solver_ref, '_project_path', None):
                project_path = Path(self._solver_ref._project_path)

                sub_path = "foms/concat"
                from cavsim3d.rom.reduction import ModelOrderReduction
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

                        snap_path = concat_dir / "snapshots" / "snapshots.h5"
                        if not snap_path.exists():
                            snap_path = concat_dir / "snapshots.h5"
                        if snap_path.exists():
                            with h5py.File(snap_path, "r") as f:
                                self.frequencies = H5Serializer.load_dataset(f["frequencies"])

                        has_results = True
                        pr.milestone(f"  Loaded existing Concatenated results from {concat_dir}")
                    except Exception as e:
                        pr.warning(f"  Could not load existing Concatenated results: {e}")

            if has_results and not rerun:
                stored = self.frequencies
                if rerun is False or (stored is not None and len(stored) == len(new_freqs)
                        and np.allclose(stored, new_freqs, rtol=1e-9, atol=0.0)):
                    pr.milestone("  Returning existing concatenated results for "
                                 "this sweep. (Use rerun=True to force a re-solve)")
                    return {
                        "frequencies": self.frequencies,
                        "Z": self._Z_matrix,
                        "S": self._S_matrix if compute_s_params else None,
                        "Z_dict": self.Z_dict,
                        "S_dict": self.S_dict if compute_s_params else None,
                    }
                # A coupled reduced solve is cheap: re-solve for the new band.
                pr.info("  Requested sweep differs from the stored concatenated "
                        "results; re-solving.")

            self.frequencies = new_freqs
            n_ext = self._n_external
            r = self.A_coupled.shape[0]

            pr.running(f"\nConcat Solve: {fmin} - {fmax} GHz, {nsamples} samples, system size {r}")

            if solver_type == 'auto':
                solver_type = 'iterative' if r >= self.ITERATIVE_SIZE_THRESHOLD else 'direct'
                pr.debug(f"  Solver: {solver_type} (system size {r})")

            self._Z_matrix = np.zeros((nsamples, n_ext, n_ext), dtype=complex)
            omegas = 2 * np.pi * self.frequencies

            t0 = time.time()

            if self.is_lossy:
                x_all = self._solve_lossy(omegas)
            elif solver_type == 'direct':
                x_all = self._solve_direct(omegas, n_ext, r)
            else:
                x_all = self._solve_iterative(omegas, n_ext, r)

            _t_concat_solve = time.time() - t0
            pr.done(f"  Concat solve complete: {_t_concat_solve:.3f}s ({nsamples} frequencies)")

            from cavsim3d.utils.timing import get_timing_registry
            get_timing_registry().record(
                "concat solve", _t_concat_solve, category="CONCAT",
                n_samples=nsamples, reduced_dofs=int(r), n_external=n_ext,
            )

            self._snapshots = np.array(x_all)

            if compute_s_params:
                self._compute_s_from_z()
            self._invalidate_cache()

            # Automatic save after simulation
            if hasattr(self, '_solver_ref') and self._solver_ref and hasattr(self._solver_ref, '_project_ref'):
                if self._solver_ref._project_ref:
                    self._solver_ref._project_ref.save()
            elif getattr(self, '_save_dir', None) is not None:
                # a netlist's joined model has no solver: it saves itself
                self.save(self._save_dir)

            return {
                "frequencies": self.frequencies,
                "Z": self._Z_matrix,
                "S": self._S_matrix if compute_s_params else None,
                "Z_dict": self.Z_dict,
                "S_dict": self.S_dict if compute_s_params else None,
            }
        finally:
            pr.pop_verbosity(_prev_verbosity)
            if _file_handler:
                pr.stop_file_log(_file_handler)

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

    @property
    def is_lossy(self) -> bool:
        """True if any coupled section carries loss operators."""
        return (getattr(self, 'C_coupled', None) is not None
                or getattr(self, 'D_coupled', None) is not None)

    def _solve_lossy(self, omegas: np.ndarray) -> List[np.ndarray]:
        """Per-frequency dense solve of (A + jwC - w^2 (I - jD)) x = w B."""
        r = self.A_coupled.shape[0]
        I = np.eye(r)
        C = self.C_coupled if self.C_coupled is not None else np.zeros((r, r))
        D = self.D_coupled if self.D_coupled is not None else np.zeros((r, r))
        x_all = []
        for k, w in enumerate(omegas):
            lhs = self.A_coupled + 1j * w * C - w ** 2 * (I - 1j * D)
            x = np.linalg.solve(lhs, w * self.B_coupled)
            x_all.append(x)
            # bilinear (B real): the lossy system is complex symmetric
            self._Z_matrix[k] = 1j * self.B_coupled.T @ x
        return x_all

    def _solve_iterative(self, omegas: np.ndarray, n_ext: int, r: int) -> List[np.ndarray]:
        """GMRES-based iterative solve."""
        I_ext = np.eye(n_ext, dtype=complex)
        x_all = []
        failures = 0

        num_freq = len(omegas)
        pr.running(f"  Solving {num_freq} frequencies...")
        
        report_interval = max(1, num_freq // 10)
        
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
    # Field Reconstruction - Unified Structure (FIXED: Direct DOF Assignment)
    # =========================================================================

    @property
    def has_snapshots(self) -> bool:
        return self._snapshots is not None

    def can_reconstruct(self) -> bool:
        """Check if field reconstruction is possible.

        Snapshots are NOT required: reconstruction maps a reduced state back
        through each section's basis (``W @ Q_L_inv``), which needs only the
        bases themselves. Requiring ``has_snapshots`` here reported False for
        systems where :meth:`reconstruct_eigenmode` and
        :meth:`reconstruct_section_field` both work.
        """
        if self.W_coupled is None:
            return False
        return all(s.can_reconstruct() for s in self.structures)

    def _reconstruct_field_from_vector(
        self,
        x_uncoupled: np.ndarray,
        scales: np.ndarray,
        interface_mode: str = 'average'
    ) -> GridFunction:
        """
        Reconstruct unified field by creating fresh per-domain FES and
        transferring DOFs element-by-element.

        A ``definedon`` FES that was stored as a reference may become stale
        after the mesh is rebuilt.  Instead of relying on cached FES objects,
        we create a fresh per-domain FES here, fill a local GridFunction, and
        copy DOF values to the global GridFunction via simultaneous
        ``GetDofNrs`` on both FES for each domain element.
        """
        from ngsolve import HCurl

        if self.mesh is None:
            raise ValueError("No mesh available")
        self._ensure_unified_fes()

        # Build domain-dof info (for interface detection only)
        self._build_domain_dof_maps()

        # Determine order and BC
        order = self.fes.globalorder
        bc = 'default'
        if self._solver_ref is not None:
            bc = getattr(self._solver_ref, 'bc',
                         getattr(getattr(self._solver_ref, 'solver', None), 'bc', bc))

        # Resolve domain mesh materials
        def _get_domain_mats(domain_name):
            solver = self._solver_ref
            if solver is not None:
                fds = getattr(solver, 'solver', solver)
                if hasattr(fds, '_get_domain_mesh_materials'):
                    mats = set(fds._get_domain_mesh_materials(domain_name))
                    if mats != {domain_name}:
                        return mats
            all_mats = set(self.mesh.GetMaterials())
            matched = {m for m in all_mats if m == domain_name or m.startswith(domain_name + '/')}
            return matched if matched else {domain_name}

        # Global GridFunction
        E_gf = GridFunction(self.fes)                    # self.fes is complex
        global_vec = E_gf.vec.FV().NumPy()
        global_vec[:] = 0

        dof_values = np.zeros(self.fes.ndof, dtype=complex)
        dof_counts = np.zeros(self.fes.ndof, dtype=int)

        for struct_idx, struct in enumerate(self.structures):
            if not struct.can_reconstruct():
                raise ValueError(
                    f"Structure {struct_idx} ({struct.domain}) cannot reconstruct."
                )

            # Extract and reconstruct
            start_r = self._structure_dof_offsets[struct_idx]
            x_reduced = x_uncoupled[start_r:start_r + struct.r]
            x_full_local = struct.reconstruct(x_reduced)
            x_full_scaled = scales[struct_idx] * x_full_local

            # Create a fresh per-domain FES on the current mesh
            domain_mats = _get_domain_mats(struct.domain)
            region = self.mesh.Materials(region_pattern(domain_mats))
            # complex: GridFunction(real_fes, complex=True) is a REAL vector,
            # which silently dropped the imaginary part of the coefficients
            fes_local = HCurl(self.mesh, order=order, dirichlet=bc, complex=True,
                              definedon=region, **hcurl_flags(kind_of(self.fes)))

            # Fill a local GridFunction with the reconstructed vector
            gf_local = GridFunction(fes_local)
            local_np = gf_local.vec.FV().NumPy()
            n = min(len(x_full_scaled), len(local_np))
            local_np[:n] = x_full_scaled[:n]

            # Transfer DOFs element-by-element
            for el in self.mesh.Elements(VOL):
                if str(el.mat) not in domain_mats:
                    continue
                local_dofs = fes_local.GetDofNrs(el)
                global_dofs = self.fes.GetDofNrs(el)
                for l_dof, g_dof in zip(local_dofs, global_dofs):
                    if l_dof >= 0 and g_dof >= 0:
                        dof_values[g_dof] += gf_local.vec[l_dof]
                        dof_counts[g_dof] += 1

        # Assign values (average at interfaces)
        n_interior = 0
        n_interface = 0
        for dof in range(self.fes.ndof):
            if dof_counts[dof] == 0:
                continue
            elif dof_counts[dof] == 1:
                global_vec[dof] = dof_values[dof]
                n_interior += 1
            else:
                global_vec[dof] = dof_values[dof] / dof_counts[dof]
                n_interface += 1

        pr.debug(f"  Reconstruction: {n_interior} interior + {n_interface} interface DOFs")
        return E_gf

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
            
        Returns
        -------
        E_gf : GridFunction
            Reconstructed electric field over the entire unified mesh
        """
        if not self.has_snapshots:
            raise ValueError("No snapshots available. Call solve() first.")

        if self.mesh is None:
            raise ValueError("No mesh available. Provide mesh to constructor.")

        self._ensure_unified_fes()

        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found. Available: {self.ports}")
        col_idx = self.ports.index(excitation_port)

        # Map from coupled to uncoupled stacked coordinates
        x_coupled = self._snapshots[freq_idx, :, col_idx]
        x_uncoupled = self.W_coupled @ x_coupled

        # Compute interface scaling factors if needed
        if enforce_continuity and self.n_structures > 1 and self.connections:
            scales = self._compute_interface_scaling_factors(
                freq_idx, excitation_port, excitation_mode
            )
            pr.debug(f"  Interface continuity scales: {[f'{abs(s):.3f}' for s in scales]}")
        else:
            scales = np.ones(self.n_structures, dtype=complex)

        return self._reconstruct_field_from_vector(x_uncoupled, scales)

    # =========================================================================
    # Interface Scaling (DOF-based approach)
    # =========================================================================

    def _compute_interface_scaling_factors(
        self,
        freq_idx: int,
        excitation_port: str,
        excitation_mode: int = 0
    ) -> np.ndarray:
        """
        Compute scaling factors to enforce field continuity at interfaces.
        Uses DOF-based matching for robust scaling.
        """
        if self.n_structures == 1:
            return np.array([1.0 + 0j])
        
        # Check cache
        cache_key = (freq_idx, excitation_port, excitation_mode)
        if cache_key in self._interface_scale_cache:
            return self._interface_scale_cache[cache_key]
        
        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found. Available: {self.ports}")
        col_idx = self.ports.index(excitation_port)
        
        # Get uncoupled solution
        x_coupled = self._snapshots[freq_idx, :, col_idx]
        x_uncoupled = self.W_coupled @ x_coupled
        
        # Build DOF maps
        self._build_domain_dof_maps()

        # Reconstruct into global DOF vectors and compute scales
        x_global_list = self._reconstruct_to_global_vectors(x_uncoupled)
        scales = self._propagate_interface_scales_global(x_global_list)
        
        # Cache
        self._interface_scale_cache[cache_key] = scales
        
        return scales

    def _propagate_interface_scales_dof(self, x_full_list: List[np.ndarray]) -> np.ndarray:
        """
        Propagate scaling factors through connection graph using DOF matching.
        """
        scales = np.ones(self.n_structures, dtype=complex)
        processed = {0}  # First structure is reference
        
        if not self.connections:
            return scales
        
        remaining = list(enumerate(self.connections))
        max_iter = len(self.connections) + 1
        
        for _ in range(max_iter):
            if not remaining:
                break
            
            made_progress = False
            still_remaining = []
            
            for conn_idx, ((sA, pA), (sB, pB)) in remaining:
                if sA in processed and sB not in processed:
                    ref_idx, new_idx = sA, sB
                elif sB in processed and sA not in processed:
                    ref_idx, new_idx = sB, sA
                elif sA in processed and sB in processed:
                    continue
                else:
                    still_remaining.append((conn_idx, ((sA, pA), (sB, pB))))
                    continue
                
                # Compute scale using DOF-based matching
                scale = self._compute_dof_scale(
                    x_full_list[ref_idx], x_full_list[new_idx],
                    ref_idx, new_idx,
                    scales[ref_idx]
                )
                
                scales[new_idx] = scale
                processed.add(new_idx)
                made_progress = True
                
                pr.debug(f"  Scale {ref_idx}->{new_idx}: {abs(scale):.4f}")
            
            remaining = still_remaining
            
            if not made_progress and remaining:
                pr.warning("Disconnected structures detected")
                for _, ((sA, _), (sB, _)) in remaining:
                    if sA not in processed:
                        scales[sA] = 1.0
                        processed.add(sA)
                    if sB not in processed:
                        scales[sB] = 1.0
                        processed.add(sB)
                break
        
        return scales

    def _compute_dof_scale(
        self,
        x_full_ref: np.ndarray,
        x_full_new: np.ndarray,
        ref_idx: int,
        new_idx: int,
        cumulative_scale_ref: complex
    ) -> complex:
        """
        Compute scaling factor using interface DOF values.
        """
        # Get interface DOFs between these two domains
        pair_key = (min(ref_idx, new_idx), max(ref_idx, new_idx))
        if pair_key not in self._interface_pairs:
            pr.warning(f"No interface between domains {ref_idx} and {new_idx}")
            return cumulative_scale_ref
        
        interface_global_dofs = self._interface_pairs[pair_key]
        
        # Get local-to-global maps
        l2g_ref = self._local_to_global_maps[ref_idx]
        l2g_new = self._local_to_global_maps[new_idx]
        
        # Invert to get global-to-local
        g2l_ref = {g: l for l, g in l2g_ref.items()}
        g2l_new = {g: l for l, g in l2g_new.items()}
        
        # Collect values at interface DOFs
        vals_ref = []
        vals_new = []
        
        for g_dof in interface_global_dofs:
            if g_dof in g2l_ref and g_dof in g2l_new:
                l_ref = g2l_ref[g_dof]
                l_new = g2l_new[g_dof]
                if l_ref < len(x_full_ref) and l_new < len(x_full_new):
                    vals_ref.append(x_full_ref[l_ref])
                    vals_new.append(x_full_new[l_new])
        
        if not vals_ref:
            pr.warning(f"No valid interface DOF values between {ref_idx} and {new_idx}")
            return cumulative_scale_ref
        
        vals_ref = np.array(vals_ref)
        vals_new = np.array(vals_new)
        
        # Filter near-zero
        mask = (np.abs(vals_ref) > 1e-12) | (np.abs(vals_new) > 1e-12)
        if not np.any(mask):
            return cumulative_scale_ref
        
        vals_ref = vals_ref[mask]
        vals_new = vals_new[mask]
        
        # Apply cumulative scale to reference
        vals_ref_scaled = cumulative_scale_ref * vals_ref
        
        # Least squares: scale * vals_new ≈ vals_ref_scaled
        denom = np.vdot(vals_new, vals_new)
        if abs(denom) < 1e-14:
            return cumulative_scale_ref
        
        scale = np.vdot(vals_new, vals_ref_scaled) / denom
        
        return scale

    def clear_interface_scale_cache(self) -> None:
        """Clear the cached interface scaling factors."""
        self._interface_scale_cache.clear()

    def get_interface_scales(
        self, 
        freq_idx: int, 
        excitation_port: str,
        excitation_mode: int = 0
    ) -> Dict[str, complex]:
        """Get the interface scaling factors for a given excitation."""
        scales = self._compute_interface_scaling_factors(
            freq_idx, excitation_port, excitation_mode
        )
        return {self.domains[i]: scales[i] for i in range(self.n_structures)}

    # =========================================================================
    # Field Visualization
    # =========================================================================

    def plot_field(
        self,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        excitation_mode: int = 0,
        component: Literal['real', 'imag', 'abs'] = 'abs',
        field_type: Literal['E', 'H'] = 'E',
        clipping: Optional[Dict] = None,
        euler_angles: Optional[List] = [45, -45, 0],
        enforce_continuity: bool = True,
        **kwargs
    ) -> None:
        """
        Visualize field over the entire unified structure.
        """
        if self.frequencies is None:
            raise ValueError("No solution available. Call solve() first.")

        if freq_idx >= len(self.frequencies):
            raise ValueError(f"freq_idx {freq_idx} out of range [0, {len(self.frequencies) - 1}]")

        if self._snapshots is None:
            raise ValueError("No snapshots available.")

        freq = self.frequencies[freq_idx]
        omega = 2 * np.pi * freq

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

        # Select field type
        if field_type == 'E':
            field_cf = E_gf
            field_label = "E"
        elif field_type == 'H':
            field_cf = (1j / (omega * mu0)) * curl(E_gf)
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

        # plot_field draws the UNIFIED structure; a netlist concat has no such
        # mesh -- use plot_section_field()/plot_eigenmode(section_idx=...).
        if self.mesh is None:
            raise ValueError(
                "This concatenated system has no unified mesh (its sections "
                "are meshed independently). Use reconstruct_section_field() "
                "or plot_eigenmode(..., section_idx=N) instead.")
        _display_webgui_fallback(Draw(BoundaryFromVolumeCF(cf_plot), self.mesh, plot_name, **draw_kwargs))
    # =========================================================================
    # Per-section field reconstruction (netlist concat: independent meshes)
    # =========================================================================

    def _section_mesh_fes(self, section_idx: int, mesh_dir=None):
        """Load (and cache) a section's saved mesh + FE space for reconstruction.

        Netlist sections live on independent meshes saved as
        ``<project>/mesh/{mesh,fes}_<base>.pkl``.  Returns ``(mesh, fes)``.
        """
        import pickle
        struct = self.structures[section_idx]
        if struct.mesh is not None and struct.fes is not None:
            return struct.mesh, struct.fes
        base = getattr(struct, 'base_domain', struct.domain)
        source_mesh = getattr(struct, 'mesh_source', None)   # referenced section
        mesh_dir = Path(mesh_dir) if mesh_dir is not None else getattr(
            self, '_project_mesh_dir', None)
        if mesh_dir is None and source_mesh is None:
            raise FileNotFoundError(
                "No mesh directory known for section field reconstruction. "
                "Pass mesh_dir=<project>/mesh.")
        if not hasattr(self, '_section_mesh_cache'):
            self._section_mesh_cache = {}
        if base in self._section_mesh_cache:
            mesh, fes = self._section_mesh_cache[base]
        else:
            if source_mesh is not None:
                mesh_dir = Path(source_mesh)
                mf, ff = mesh_dir / "mesh.pkl", mesh_dir / "fes.pkl"
            else:
                mf, ff = mesh_dir / f"mesh_{base}.pkl", mesh_dir / f"fes_{base}.pkl"
            if not mf.exists() or not ff.exists():
                raise FileNotFoundError(
                    f"Section '{base}' mesh/FES not found in {mesh_dir} "
                    f"(need {mf.name} and {ff.name}).")
            with open(mf, "rb") as fh:
                mesh = pickle.load(fh)
            with open(ff, "rb") as fh:
                fes = pickle.load(fh)
            self._section_mesh_cache[base] = (mesh, fes)
        struct.mesh, struct.fes = mesh, fes
        return mesh, fes

    def reconstruct_section_field(
        self,
        section_idx: int = 0,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        excitation_mode: int = 0,
        field_type: Literal['E', 'H'] = 'E',
        component: Literal['real', 'imag', 'abs'] = 'abs',
    ):
        """Reconstruct one section's 3D field FROM THE COUPLED ROM solution.

        The coupled sweep gives the reduced state; this maps that section's
        slice back through its reduced basis (``W @ Q_L_inv``) onto the
        section's own mesh.  Returns ``(coefficient_function, mesh, label)``
        ready for ``WebguiComponent.draw`` / ``netgen.webgui.Draw``.
        """
        from ngsolve import GridFunction, Norm, curl, BoundaryFromVolumeCF
        if not self.has_snapshots:
            raise ValueError("No coupled solution — call solve() first.")
        if not (0 <= section_idx < self.n_structures):
            raise IndexError(f"section_idx {section_idx} out of range "
                             f"[0, {self.n_structures - 1}]")
        if excitation_port is None:
            excitation_port = self.ports[0]
        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found: {self.ports}")

        col = self.ports.index(excitation_port)
        x_uncoupled = self.W_coupled @ self._snapshots[freq_idx, :, col]
        struct = self.structures[section_idx]
        start = self._structure_dof_offsets[section_idx]
        x_full = struct.reconstruct(x_uncoupled[start:start + struct.r])

        mesh, fes = self._section_mesh_fes(section_idx)

        # The saved FE space is real, and `GridFunction(fes, complex=True)` is
        # not a real option -- NGSolve warns about the unknown flag and hands
        # back a REAL vector, so assigning the complex coefficients silently
        # dropped the phase. Rebuild the space as complex so the reconstructed
        # field keeps its imaginary part (only 'abs' was unaffected).
        if not fes.is_complex:
            fes = HCurl(mesh, order=fes.globalorder, complex=True,
                        dirichlet=getattr(fes, '_dirichlet', '') or '',
                        **hcurl_flags(kind_of(fes)))
        E_gf = GridFunction(fes)
        vec = E_gf.vec.FV().NumPy()
        n = min(len(vec), len(x_full))
        vec[:] = 0
        vec[:n] = x_full[:n]

        omega = 2 * np.pi * self.frequencies[freq_idx]
        field_cf = E_gf if field_type == 'E' else (1j / (omega * mu0)) * curl(E_gf)
        cf = {'abs': Norm(field_cf), 'real': field_cf.real,
              'imag': field_cf.imag}[component]
        label = f"{component}({field_type}) — {struct.domain} @ " \
                f"{self.frequencies[freq_idx] / 1e9:.3f} GHz"
        return BoundaryFromVolumeCF(cf), mesh, label

    def _section_coefficient_vectors(self, freq_idx, excitation_port=None):
        """Per-section full-order coefficient vectors from the coupled state."""
        if not self.has_snapshots:
            raise ValueError("No coupled solution - call solve() first.")
        if excitation_port is None:
            excitation_port = self.ports[0]
        if excitation_port not in self.ports:
            raise KeyError(f"Port '{excitation_port}' not found: {self.ports}")
        col = self.ports.index(excitation_port)
        x_uncoupled = self.W_coupled @ self._snapshots[freq_idx, :, col]
        vecs = []
        for i, struct in enumerate(self.structures):
            start = self._structure_dof_offsets[i]
            vecs.append(struct.reconstruct(x_uncoupled[start:start + struct.r]))
        return vecs

    def reconstruct_chain_field(
        self,
        freq_idx: int = 0,
        excitation_port: Optional[str] = None,
        field_type: Literal['E', 'H'] = 'E',
        component: Literal['real', 'imag', 'abs'] = 'abs',
        axis: Optional[Literal['x', 'y', 'z']] = None,
        gap: float = 0.0,
        boundary_only: bool = False,
    ):
        """Reconstruct the field over the WHOLE concatenated geometry.

        :meth:`reconstruct_section_field` returns one section at a time on its
        own mesh. This assembles every section onto a single compound mesh, so
        the coupled solution is viewable across the entire chain.

        Only valid when the sections are rigid copies of ONE reference mesh --
        the ``asm.add(name, geo, n=N)`` case. A rigid transform preserves mesh
        topology and DOF numbering, and an H(curl) DOF is invariant under it,
        so each section's coefficient vector is copied verbatim onto its placed
        copy: no interpolation and no added error.

        Returns ``(coefficient_function, compound_mesh, label)`` ready for
        ``netgen.webgui.Draw``.
        """

        vecs = self._section_coefficient_vectors(freq_idx, excitation_port)
        omega = 2 * np.pi * self.frequencies[freq_idx]
        label = (f"{component}({field_type}) - {self.n_structures}-section chain @ "
                 f"{self.frequencies[freq_idx] / 1e9:.3f} GHz")
        return self._assemble_chain(vecs, omega, field_type, component, label,
                                    axis=axis, gap=gap, boundary_only=boundary_only,
                                    caller="reconstruct_chain_field")

    def _assemble_chain(self, vecs, omega, field_type, component, label,
                        axis=None, gap=0.0, caller="reconstruct_chain_field",
                        boundary_only=False):
        """Place per-section coefficient vectors on one replicated compound mesh."""
        from ngsolve import Norm, curl, BoundaryFromVolumeCF
        E_gf, comp_mesh = self._assemble_chain_gf(vecs, axis=axis, gap=gap,
                                                  caller=caller)
        field_cf = E_gf if field_type == 'E' else (1j / (omega * mu0)) * curl(E_gf)
        cf = {'abs': Norm(field_cf), 'real': field_cf.real,
              'imag': field_cf.imag}[component]
        # Return the VOLUME CoefficientFunction. Wrapping it in
        # BoundaryFromVolumeCF would keep surface values only, and an
        # accelerating mode is axial: it peaks on the beam axis and falls to
        # ~zero on the PEC wall (measured 9x axis-to-wall here), so the surface
        # view shows almost nothing while a clip plane through the axis shows
        # the real field. Pass boundary_only=True for the old surface CF.
        if boundary_only:
            cf = BoundaryFromVolumeCF(cf)
        return cf, comp_mesh, label

    def _assemble_chain_gf(self, vecs, axis=None, gap=0.0,
                           caller="reconstruct_chain_field"):
        """Compound GridFunction of the chain (vector field, complex)."""
        from ngsolve import HCurl
        from cavsim3d.utils.mesh_replication import (
            Placement, replicate_mesh, block_dof_maps, assemble_compound_field)

        n_sec = self.n_structures
        bases = {getattr(st, 'base_domain', st.domain) for st in self.structures}
        meshes = [self._section_mesh_fes(i) for i in range(n_sec)]
        ref_mesh, ref_fes = meshes[0]
        if any(m is not ref_mesh for m, _ in meshes):
            raise ValueError(
                f"{caller}() needs all sections to share one reference mesh "
                "(the repeated-section case, asm.add(..., n=N)). Found "
                f"{len(bases)} distinct section mesh(es): {sorted(bases)}.")

        # Chain along `axis`: each copy is shifted by the reference extent.
        # axis=None (the default) picks the section's LONGEST extent, which is
        # the chaining direction for a beamline component. Assuming 'z' silently
        # stacks copies across the cavity diameter when the CAD axis is x or y.
        pts = np.array([ref_mesh[v].point for v in ref_mesh.vertices])
        extents = pts.max(axis=0) - pts.min(axis=0)
        ax = int(np.argmax(extents)) if axis is None else 'xyz'.index(axis)
        span = float(extents[ax])
        placements = []
        for k in range(n_sec):
            t = [0.0, 0.0, 0.0]
            t[ax] = k * (span + gap)
            placements.append(Placement(tuple(t)))

        comp_mesh = replicate_mesh(ref_mesh, placements)
        order = ref_fes.globalorder
        kind = hcurl_flags(kind_of(ref_fes))
        comp_fes = HCurl(comp_mesh, order=order, complex=True, **kind)
        if not ref_fes.is_complex:
            ref_fes = HCurl(ref_mesh, order=order, complex=True, **kind)
        maps = block_dof_maps(ref_fes, comp_fes, ref_mesh, comp_mesh, n_sec)
        E_gf = assemble_compound_field(comp_fes, vecs, maps)
        return E_gf, comp_mesh

    def chain_axis_profile(
        self,
        mode_idx: Optional[int] = None,
        freq_idx: Optional[int] = None,
        excitation_port: Optional[str] = None,
        n_points: int = 400,
        axis: Optional[Literal['x', 'y', 'z']] = None,
        transverse: Optional[Tuple[float, float]] = None,
        gap: float = 0.0,
        enforce_continuity: bool = True,
    ):
        """On-axis longitudinal field profile along the chain.

        For an accelerating structure the quantity of interest is
        :math:`E_z(z)` on the beam axis. Give either ``mode_idx`` (a mode of
        :meth:`get_resonant_frequencies`, as :meth:`chain_eigenfrequencies`
        lists them) or ``freq_idx`` (a sample of the coupled sweep).

        Returns ``(coord, E_long, label)`` where ``coord`` is the position along
        the chain [m] and ``E_long`` the complex longitudinal component.
        Points falling outside the mesh come back as NaN.
        """
        if (mode_idx is None) == (freq_idx is None):
            raise ValueError("give exactly one of mode_idx or freq_idx")

        if mode_idx is not None:
            evals, evecs, k = self._coupled_mode(mode_idx)
            x_uncoupled = self.W_coupled @ evecs[:, k]
            scales = np.ones(self.n_structures, dtype=complex)
            if (enforce_continuity and self.n_structures > 1 and self.connections
                    and self.mesh is not None and self.fes is not None):
                scales = self._compute_eigenmode_scales(x_uncoupled)
            vecs = []
            for i, st in enumerate(self.structures):
                start = self._structure_dof_offsets[i]
                vecs.append(np.asarray(
                    st.reconstruct(x_uncoupled[start:start + st.r])) * scales[i])
            f_ghz = float(np.sqrt(evals[k]) / (2 * np.pi) / 1e9)
            label = f"eigenmode {mode_idx} @ {f_ghz:.4f} GHz"
        else:
            vecs = self._section_coefficient_vectors(freq_idx, excitation_port)
            label = f"sweep @ {self.frequencies[freq_idx] / 1e9:.4f} GHz"

        E_gf, comp_mesh = self._assemble_chain_gf(
            vecs, axis=axis, gap=gap, caller="chain_axis_profile")

        pts = np.array([comp_mesh[v].point for v in comp_mesh.vertices])
        if axis is None:
            ref_pts = np.array([m.point for m in
                                [comp_mesh[v] for v in comp_mesh.vertices]])
            ax = int(np.argmax(ref_pts.max(axis=0) - ref_pts.min(axis=0)))
        else:
            ax = 'xyz'.index(axis)
        lo, hi = float(pts[:, ax].min()), float(pts[:, ax].max())
        if transverse is None:                     # centre of the cross-section
            other = [i for i in range(3) if i != ax]
            transverse = tuple(
                float((pts[:, i].min() + pts[:, i].max()) / 2) for i in other)

        coord = np.linspace(lo, hi, n_points)
        E_long = np.full(n_points, np.nan, dtype=complex)
        others = [i for i in range(3) if i != ax]
        for k, c in enumerate(coord):
            xyz = [0.0, 0.0, 0.0]
            xyz[ax] = float(c)
            xyz[others[0]], xyz[others[1]] = transverse
            try:
                val = E_gf(comp_mesh(*xyz))
                E_long[k] = complex(val[ax])
            except Exception:
                pass                                # outside the mesh -> NaN
        return coord, E_long, label

    def chain_eigenfrequencies(self, fmin_ghz=None, fmax_ghz=None):
        """Eigenfrequencies (GHz) of the coupled chain, with their mode indices.

        The modes of :meth:`get_resonant_frequencies` (those within 10 % of
        the training band's edges) between *fmin_ghz* and *fmax_ghz*.  Their
        indices count that list, as the ``mode_idx`` of
        :meth:`reconstruct_chain_eigenmode`, :meth:`chain_axis_profile` and
        :meth:`plot_eigenmode` and the ``mode_index`` of :meth:`get_eigenmode`,
        :meth:`get_rq` and :meth:`get_figures_of_merit` do.
        """
        if self.A_coupled is None:
            raise ValueError("System not coupled.")
        f = self.get_resonant_frequencies() / 1e9
        idx = np.arange(len(f))
        if fmin_ghz is not None:
            keep = f >= fmin_ghz
            f, idx = f[keep], idx[keep]
        if fmax_ghz is not None:
            keep = f <= fmax_ghz
            f, idx = f[keep], idx[keep]
        return idx, f

    def _coupled_mode(self, mode_idx: int):
        """``(evals, evecs, k)``: the eigenpairs of ``A_coupled`` above 1e-6, which
        the chain reconstruction uses, and the position among them of mode
        *mode_idx* of :meth:`get_resonant_frequencies` (a run of them, from the
        first mode near the training band on)."""
        if self.A_coupled is None:
            raise ValueError("System not coupled.")
        listed = self.get_resonant_frequencies()
        if not 0 <= int(mode_idx) < len(listed):
            raise ValueError(f"mode_idx {mode_idx} out of range: "
                             f"get_resonant_frequencies() lists {len(listed)} modes")
        evals, evecs = np.linalg.eigh(self.A_coupled)
        keep = evals > 1e-6
        evals, evecs = evals[keep], evecs[:, keep]
        first = (2 * np.pi * listed[0]) ** 2
        return evals, evecs, int(np.searchsorted(evals, first * (1 - 1e-9))) + int(mode_idx)

    def _balance_degenerate_mode(self, evals, evecs, mode_idx, tol=1e-6):
        """Pick the most evenly distributed member of a degenerate group.

        Identical cells in a chain put the chain's modes into near-exact
        degenerate groups, one member per cell. Every orthonormal basis of such
        a group is an equally valid set of eigenvectors, and ``eigh`` returns an
        arbitrary one -- in practice localised on a single cell. The exported
        field then shows one cavity lit and the rest dark, which looks like a
        reconstruction bug but is only the basis choice: for the 2-cavity
        module the accelerating doublet came back as 0.090/0.996 per cavity,
        while the sum of its two members is 0.640/0.768.

        Rotate inside the group to the combination whose energy is spread as
        evenly as possible over the sections, and return its coefficient vector
        in the eigenvector basis.
        """
        lam = float(np.real(evals[mode_idx]))
        cluster = np.flatnonzero(
            np.abs(np.real(evals) - lam) <= tol * max(abs(lam), 1e-300))
        if len(cluster) < 2:
            return evecs[:, mode_idx]

        U = self.W_coupled @ evecs[:, cluster]
        blocks = [U[self._structure_dof_offsets[i]:
                    self._structure_dof_offsets[i] + self.structures[i].r, :]
                  for i in range(self.n_structures)]
        d = len(cluster)

        def spread(a):
            """Smallest section share of the total energy; bigger is better."""
            e = np.array([float(np.sum(np.abs(B @ a) ** 2)) for B in blocks])
            tot = e.sum()
            return 0.0 if tot <= 0 else float(e.min() / tot)

        # d is one per cell, so a short random search with a shrinking local
        # step is cheaper and more robust here than pulling in an optimiser.
        rng = np.random.default_rng(0)
        best_a = np.zeros(d, dtype=complex)
        best_a[list(cluster).index(mode_idx)] = 1.0
        best = spread(best_a)
        for a in ([np.ones(d, dtype=complex)] +
                  [rng.normal(size=d) + 1j * rng.normal(size=d)
                   for _ in range(200)]):
            a = a / max(np.linalg.norm(a), 1e-300)
            v = spread(a)
            if v > best:
                best, best_a = v, a
        step = 0.5
        for _ in range(60):
            improved = False
            for _ in range(20):
                a = best_a + step * (rng.normal(size=d)
                                     + 1j * rng.normal(size=d))
                a = a / max(np.linalg.norm(a), 1e-300)
                v = spread(a)
                if v > best:
                    best, best_a, improved = v, a, True
            if not improved:
                step *= 0.6
        pr.debug(f"  chain eigenmode {mode_idx}: degenerate group of {d}, "
                 f"balanced to a {best * 100:.1f}% minimum section share "
                 f"(an even split is {100 / self.n_structures:.1f}%)")
        return evecs[:, cluster] @ best_a

    def reconstruct_chain_eigenmode(
        self,
        mode_idx: int = 0,
        field_type: Literal['E', 'H'] = 'E',
        component: Literal['real', 'imag', 'abs'] = 'abs',
        enforce_continuity: bool = True,
        axis: Optional[Literal['x', 'y', 'z']] = None,
        gap: float = 0.0,
        boundary_only: bool = False,
        balance_degenerate: bool = True,
        degeneracy_tol: float = 1e-6,
    ):
        """Reconstruct an EIGENMODE of the coupled chain over the whole geometry.

        :meth:`reconstruct_eigenmode` needs a single glued mesh, which a netlist
        of repeated sections does not have. This assembles the mode onto a
        compound mesh built by rigidly replicating the reference section, so a
        chain eigenmode can be viewed across the full structure.

        ``mode_idx`` is a mode of :meth:`get_resonant_frequencies`;
        :meth:`chain_eigenfrequencies` lists them with their frequencies.
        Returns ``(coefficient_function, compound_mesh, label)`` for
        ``netgen.webgui.Draw``.
        """
        evals, evecs, k = self._coupled_mode(mode_idx)
        coeffs = evecs[:, k]
        if balance_degenerate and self.n_structures > 1:
            coeffs = self._balance_degenerate_mode(evals, evecs, k, degeneracy_tol)
        x_uncoupled = self.W_coupled @ coeffs
        scales = np.ones(self.n_structures, dtype=complex)
        if enforce_continuity and self.n_structures > 1 and self.connections:
            # The interface rescaling compares DOFs on a unified mesh, which a
            # netlist of replicated sections does not have. The coupling has
            # already enforced continuity in the reduced space, so fall back to
            # unit scales rather than failing.
            if self.mesh is not None and self.fes is not None:
                scales = self._compute_eigenmode_scales(x_uncoupled)
            else:
                pr.debug("  chain eigenmode: no unified mesh, skipping "
                         "interface rescaling (unit scales)")

        vecs = []
        for i, struct in enumerate(self.structures):
            start = self._structure_dof_offsets[i]
            xi = struct.reconstruct(x_uncoupled[start:start + struct.r])
            vecs.append(np.asarray(xi) * scales[i])

        omega = float(np.sqrt(evals[k]))
        label = (f"{component}({field_type}) - chain eigenmode {mode_idx} @ "
                 f"{omega / (2 * np.pi) / 1e9:.4f} GHz")
        return self._assemble_chain(vecs, omega, field_type, component, label,
                                    axis=axis, gap=gap, boundary_only=boundary_only,
                                    caller="reconstruct_chain_eigenmode")

    def plot_field_at_frequency(self, freq: float, **kwargs) -> None:
        """
        Plot field at specific frequency (Hz).
        """
        if self.frequencies is None:
            raise ValueError("No solution available.")
        freq_idx = int(np.argmin(np.abs(self.frequencies - freq)))
        actual_freq = self.frequencies[freq_idx]
        if abs(actual_freq - freq) / max(freq, 1e-10) > 0.01:
            pr.debug(f"  Note: Using nearest frequency {actual_freq / 1e9:.4f} GHz")
        self.plot_field(freq_idx=freq_idx, **kwargs)

    # =========================================================================
    # Eigenmode Reconstruction and Visualization
    # =========================================================================

    def reconstruct_section_eigenmode(
        self,
        mode_idx: int = 0,
        section_idx: int = 0
    ) -> GridFunction:
        """Eigenmode of the coupled system, drawn on ONE section's own mesh.

        Netlist sections have independent meshes, so there is no single space to
        draw a chain mode on. The coupled eigenvector still spans every section;
        this slices out ``section_idx``'s reduced coordinates and lifts them
        through that section's basis.  ``mode_idx`` is a mode of
        :meth:`get_resonant_frequencies`.
        """
        from ngsolve import GridFunction
        if not (0 <= section_idx < self.n_structures):
            raise IndexError(f"section_idx {section_idx} out of range "
                             f"[0, {self.n_structures - 1}]")

        evals, evecs, k = self._coupled_mode(mode_idx)
        pr.info(f"\nEigenmode {mode_idx} at f = "
                f"{np.sqrt(evals[k]) / (2 * np.pi) / 1e9:.4f} GHz "
                f"(section {section_idx})")

        x_uncoupled = self.W_coupled @ evecs[:, k]
        struct = self.structures[section_idx]
        start = self._structure_dof_offsets[section_idx]
        x_full = struct.reconstruct(x_uncoupled[start:start + struct.r])

        mesh, fes = self._section_mesh_fes(section_idx)
        E_gf = GridFunction(fes)
        vec = E_gf.vec.FV().NumPy()
        n = min(len(vec), len(x_full))
        vec[:] = 0
        vec[:n] = np.real(x_full[:n]) if not fes.is_complex else x_full[:n]
        return E_gf

    def reconstruct_eigenmode(
        self,
        mode_idx: int = 0,
        enforce_continuity: bool = True,
        section_idx: int = 0
    ) -> GridFunction:
        """
        Reconstruct eigenmode field over the entire structure.
        
        Parameters
        ----------
        mode_idx : int
            A mode of :meth:`get_resonant_frequencies`
        enforce_continuity : bool
            If True, scale fields to enforce continuity at interfaces

        Returns
        -------
        E_gf : GridFunction
            Reconstructed eigenmode field
        """
        if self.A_coupled is None:
            raise ValueError("System not coupled. Call couple() first.")

        if self.mesh is None:
            # A NETLIST concat has no unified mesh -- its sections live on
            # independent meshes. Reconstruct on the requested section instead.
            return self.reconstruct_section_eigenmode(
                mode_idx, section_idx=section_idx)

        self._ensure_unified_fes()
        _evals, evecs, k = self._coupled_mode(mode_idx)
        x_coupled = evecs[:, k]
        
        # Map to uncoupled coordinates
        x_uncoupled = self.W_coupled @ x_coupled
        
        # Compute scaling factors
        if enforce_continuity and self.n_structures > 1 and self.connections:
            scales = self._compute_eigenmode_scales(x_uncoupled)
            pr.debug(f"  Eigenmode scales: {[f'{abs(s):.3f}' for s in scales]}")
        else:
            scales = np.ones(self.n_structures, dtype=complex)
        
        return self._reconstruct_field_from_vector(x_uncoupled, scales)

    def _compute_eigenmode_scales(self, x_uncoupled: np.ndarray) -> np.ndarray:
        """Compute scaling factors for eigenmode reconstruction.

        Reconstructs each domain into a global-DOF vector using a fresh
        per-domain FES, then compares values at interface DOFs.
        """
        self._build_domain_dof_maps()

        if not self._interface_pairs:
            return np.ones(self.n_structures, dtype=complex)

        # Reconstruct each domain into global DOF space
        x_global_list = self._reconstruct_to_global_vectors(x_uncoupled)

        return self._propagate_interface_scales_global(x_global_list)

    def _reconstruct_to_global_vectors(self, x_uncoupled: np.ndarray) -> List[np.ndarray]:
        """Reconstruct each domain into a global-DOF-sized vector.

        Creates a fresh per-domain FES, fills it with the reconstructed
        local solution, then copies DOFs into a global-sized array via
        element-by-element GetDofNrs.
        """
        from ngsolve import HCurl

        self._ensure_unified_fes()
        order = self.fes.globalorder
        bc = 'default'
        if self._solver_ref is not None:
            bc = getattr(self._solver_ref, 'bc',
                         getattr(getattr(self._solver_ref, 'solver', None), 'bc', bc))

        def _get_domain_mats(domain_name):
            solver = self._solver_ref
            if solver is not None:
                fds = getattr(solver, 'solver', solver)
                if hasattr(fds, '_get_domain_mesh_materials'):
                    mats = set(fds._get_domain_mesh_materials(domain_name))
                    if mats != {domain_name}:
                        return mats
            all_mats = set(self.mesh.GetMaterials())
            matched = {m for m in all_mats if m == domain_name or m.startswith(domain_name + '/')}
            return matched if matched else {domain_name}

        x_global_list = []
        for struct_idx, struct in enumerate(self.structures):
            start_r = self._structure_dof_offsets[struct_idx]
            x_reduced = x_uncoupled[start_r:start_r + struct.r]
            x_full_local = struct.reconstruct(x_reduced)

            domain_mats = _get_domain_mats(struct.domain)
            region = self.mesh.Materials(region_pattern(domain_mats))
            fes_local = HCurl(self.mesh, order=order, dirichlet=bc, complex=True,
                              definedon=region, **hcurl_flags(kind_of(self.fes)))

            gf_local = GridFunction(fes_local)
            local_np = gf_local.vec.FV().NumPy()
            n = min(len(x_full_local), len(local_np))
            local_np[:n] = x_full_local[:n]

            x_global = np.zeros(self.fes.ndof, dtype=complex)
            for el in self.mesh.Elements(VOL):
                if str(el.mat) not in domain_mats:
                    continue
                local_dofs = fes_local.GetDofNrs(el)
                global_dofs = self.fes.GetDofNrs(el)
                for l_dof, g_dof in zip(local_dofs, global_dofs):
                    if l_dof >= 0 and g_dof >= 0:
                        x_global[g_dof] = gf_local.vec[l_dof]

            x_global_list.append(x_global)

        return x_global_list

    def _propagate_interface_scales_global(
        self, x_global_list: List[np.ndarray]
    ) -> np.ndarray:
        """Propagate scaling factors using global-DOF vectors at interfaces."""
        scales = np.ones(self.n_structures, dtype=complex)
        processed = {0}

        if not self.connections:
            return scales

        remaining = list(enumerate(self.connections))
        max_iter = len(self.connections) + 1

        for _ in range(max_iter):
            if not remaining:
                break
            made_progress = False
            still_remaining = []

            for conn_idx, ((sA, pA), (sB, pB)) in remaining:
                if sA in processed and sB not in processed:
                    ref_idx, new_idx = sA, sB
                elif sB in processed and sA not in processed:
                    ref_idx, new_idx = sB, sA
                elif sA in processed and sB in processed:
                    continue
                else:
                    still_remaining.append((conn_idx, ((sA, pA), (sB, pB))))
                    continue

                # Find shared DOFs between domains
                pair_key = (min(ref_idx, new_idx), max(ref_idx, new_idx))
                if pair_key not in self._interface_pairs:
                    pr.warning(f"No interface between domains {ref_idx} and {new_idx}")
                    scales[new_idx] = scales[ref_idx]
                    processed.add(new_idx)
                    made_progress = True
                    continue

                interface_dofs = self._interface_pairs[pair_key]
                vals_ref = x_global_list[ref_idx][list(interface_dofs)]
                vals_new = x_global_list[new_idx][list(interface_dofs)]

                # Filter near-zero
                mask = (np.abs(vals_ref) > 1e-12) | (np.abs(vals_new) > 1e-12)
                if np.any(mask):
                    vr = scales[ref_idx] * vals_ref[mask]
                    vn = vals_new[mask]
                    denom = np.vdot(vn, vn)
                    if abs(denom) > 1e-14:
                        scales[new_idx] = np.vdot(vn, vr) / denom
                    else:
                        scales[new_idx] = scales[ref_idx]
                else:
                    scales[new_idx] = scales[ref_idx]

                processed.add(new_idx)
                made_progress = True
                pr.debug(f"  Scale {ref_idx}->{new_idx}: {abs(scales[new_idx]):.4f}")

            remaining = still_remaining
            if not made_progress and remaining:
                pr.warning("Disconnected structures detected")
                for _, ((sA, _), (sB, _)) in remaining:
                    for s in (sA, sB):
                        if s not in processed:
                            scales[s] = 1.0
                            processed.add(s)
                break

        return scales

    def plot_eigenmode(
        self,
        mode_idx: int = 0,
        section_idx: int = 0,
        component: Literal['real', 'imag', 'abs'] = 'abs',
        field_type: Literal['E', 'H'] = 'E',
        clipping: Optional[Dict] = None,
        euler_angles: Optional[List] = [45, -45, 0],
        enforce_continuity: bool = True,
        **kwargs
    ) -> None:
        """
        Visualize eigenmode over the entire unified structure.
        
        Parameters
        ----------
        mode_idx : int
            A mode of :meth:`get_resonant_frequencies`, as
            :meth:`get_eigenmode` and :meth:`get_external_q` count them
        component : {'real', 'imag', 'abs'}
            Field component to plot
        field_type : {'E', 'H'}
            Electric or magnetic field
        clipping : dict, optional
            Clipping plane specification
        euler_angles : list, optional
            View orientation
        enforce_continuity : bool
            If True, scale fields for interface continuity
        """
        evals, _evecs, k = self._coupled_mode(mode_idx)
        freq = np.sqrt(evals[k]) / (2 * np.pi)
        omega = 2 * np.pi * freq
        
        if self.mesh is not None:   # the per-section path prints its own
            pr.info(f"\nEigenmode {mode_idx} at f = {freq / 1e9:.4f} GHz")
        
        # Reconstruct field
        E_gf = self.reconstruct_eigenmode(
            mode_idx, enforce_continuity=enforce_continuity,
            section_idx=section_idx)
        
        # Select field type
        if field_type == 'E':
            field_cf = E_gf
            field_label = "E"
        elif field_type == 'H':
            field_cf = (1j / (omega * mu0)) * curl(E_gf)
            field_label = "H"
        else:
            raise ValueError(f"Invalid field_type: {field_type}")
        
        # Select component
        if component == 'abs':
            cf_plot = Norm(field_cf)
            plot_name = f"|{field_label}| mode {mode_idx}"
        elif component == 'real':
            cf_plot = field_cf.real
            plot_name = f"Re({field_label}) mode {mode_idx}"
        elif component == 'imag':
            cf_plot = field_cf.imag
            plot_name = f"Im({field_label}) mode {mode_idx}"
        else:
            raise ValueError(f"Invalid component: {component}")
        
        draw_kwargs = kwargs.copy()
        if clipping:
            draw_kwargs['clipping'] = clipping
        if euler_angles:
            draw_kwargs['euler_angles'] = euler_angles
        
        # A netlist concat has no unified mesh; the field was reconstructed on
        # ONE section, so draw it on that section's own mesh.
        draw_mesh = self.mesh
        if draw_mesh is None:
            draw_mesh, _ = self._section_mesh_fes(section_idx)
        _display_webgui_fallback(Draw(BoundaryFromVolumeCF(cf_plot), draw_mesh, plot_name, **draw_kwargs))
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

    def get_eigenvalues_filtered(
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
        filter_static: bool = True,
        fmax: float = None
    ) -> np.ndarray:
        """Resonant frequencies [Hz] of the unified structure, ascending.

        By default only the modes within 10 % of the training band's edges
        (where the joined parts' bands overlap) are listed: far from it the
        reduced models leave spurious modes.  The mode indices of
        :meth:`get_eigenmode`, :meth:`get_rq` and :meth:`get_figures_of_merit`
        count the same list.  *fmin*/*fmax* [GHz] give another band
        (``fmin=0``: every mode above the static ones).
        """
        explicit = fmin is not None or fmax is not None
        if fmin is not None:
            min_eigenvalue = (2 * np.pi * fmin * 1e9) ** 2
            filter_static = True
        else:
            min_eigenvalue = self.DEFAULT_MIN_EIGENVALUE if filter_static else None

        eigs = self.get_eigenvalues_filtered(
            filter_static=filter_static,
            min_eigenvalue=min_eigenvalue
        )
        window = self._training_window() if filter_static and not explicit else None
        if window is not None:
            eigs = eigs[(eigs >= window[0]) & (eigs <= window[1])]
        eigs_pos = eigs[eigs > 0]
        freqs = np.sqrt(eigs_pos) / (2 * np.pi)
        if fmax is not None:
            freqs = freqs[freqs <= fmax * 1e9]

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
        pr.debug("\nUnified system dimensions:")
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
            print("\nSolution:")
            print(f"  Range: {self.frequencies[0] / 1e9:.4f} - {self.frequencies[-1] / 1e9:.4f} GHz")
            print(f"  Samples: {len(self.frequencies)}")

        print("=" * 60)


# =============================================================================
# ReducedConcatenatedSystem - Further POD reduction of concatenated system
# =============================================================================

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
        C_reduced: Optional[np.ndarray] = None,
        D_reduced: Optional[np.ndarray] = None,
    ):
        # Don't call parent __init__, manually copy state
        BaseEMSolver.__init__(self)

        # Copy all parent state
        self.structures = parent.structures
        self.n_structures = parent.n_structures
        self._port_impedance_func = parent._port_impedance_func
        self._port_wave_impedance_func = getattr(
            parent, '_port_wave_impedance_func', None)
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
        self.C_coupled = C_reduced
        self.D_coupled = D_reduced

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

        # Initialize caches
        self._resonant_mode_cache = {}
        self._interface_scale_cache = {}
        
        # DOF mapping caches (will be built on first use)
        self._domain_dofs = None
        self._interface_dofs = None
        self._interface_pairs = None
        self._local_to_global_maps = None

        # Snapshot storage (populated by solve())
        self._snapshots = None
        
        # Initialize result matrices
        self._Z_matrix = None
        self._S_matrix = None
        self.frequencies = None

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

        pr.debug("\nReduction:")
        pr.debug(f"  Parent coupled DOFs: {self._parent_coupled_dofs}")
        pr.debug(f"  This level DOFs: {self.coupled_dofs}")
        if self._parent_coupled_dofs and self._parent_coupled_dofs > 0:
            compression = (1 - self.coupled_dofs / self._parent_coupled_dofs) * 100
            pr.debug(f"  Compression: {compression:.1f}%")

        pr.debug("\nSingular values (top 5):")
        for i, sv in enumerate(self._singular_values[:5]):
            pr.echo(f"  σ_{i} = {sv:.4e}")
        if len(self._singular_values) > 5:
            print(f"  ... ({len(self._singular_values)} total)")

        print(f"\nField reconstruction: {'Ready' if self.can_reconstruct() else 'Not available'}")

        if self.frequencies is not None:
            print("\nSolution:")
            print(f"  Range: {self.frequencies[0] / 1e9:.4f} - {self.frequencies[-1] / 1e9:.4f} GHz")
            print(f"  Samples: {len(self.frequencies)}")

        print("=" * 60)


# =============================================================================
# POD Reduction Function
# =============================================================================

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
    from cavsim3d.rom.reduction import check_reduce_args
    check_reduce_args(tol, max_rank)
    if concat.A_coupled is None:
        raise ValueError("System must be coupled first")
    if concat._snapshots is None:
        raise ValueError("No snapshots. Call solve() first.")

    r_current = concat.A_coupled.shape[0]

    # Collect snapshots: shape (n_freq, r_coupled, n_ext) -> (r_coupled, n_freq * n_ext)
    W_snap = np.hstack([concat._snapshots[k] for k in range(len(concat._snapshots))])

    # SVD for POD basis.  A REAL basis (from [Re, Im] of the snapshots) keeps
    # the complex-symmetric structure of a lossy system under projection.
    if np.iscomplexobj(W_snap):
        W_snap = np.hstack([W_snap.real, W_snap.imag])
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

    C_reduced = D_reduced = None
    if getattr(concat, 'C_coupled', None) is not None:
        C_reduced = W_r.T @ concat.C_coupled @ W_r
    if getattr(concat, 'D_coupled', None) is not None:
        D_reduced = W_r.T @ concat.D_coupled @ W_r

    print(f"\nReduced unified system: {r_current} -> {r_new} DOFs")
    print(f"  Compression: {100 * (1 - r_new / r_current):.1f}%")
    print(f"  Singular value decay: {S[0]:.2e} -> {S[min(r_new, len(S) - 1)]:.2e}")

    return ReducedConcatenatedSystem(
        parent=concat,
        A_reduced=A_reduced,
        B_reduced=B_reduced,
        W_reduction=W_r,
        singular_values=S,
        C_reduced=C_reduced,
        D_reduced=D_reduced,
    )