"""Model order reduction with the beam (docs/theory/beam_reduction.md §10).

The beam column of a full-order sweep is reduced with the port columns into
one basis (§10.3).  Its load and outputs change shape with frequency through
the beam's phase exp(-j k_b s); every part is either separable (a finite sum of
known phases, §10.4) or a *phase integral*, which the full-order sweep
evaluated at Chebyshev points of the band (``BeamSystem.affine_data``).  The
reduced beam column (:class:`ReducedBeam`) therefore needs no mesh: it is
saved with the reduced model and evaluated at any frequency of its band.

The PEC walls carry the beam's data as a lift g(w) (§10.2): the basis is built
from the free part of the beam snapshots only, and the lift enters the reduced
load as -V^T A_fd(w) g(w).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from cavsim3d.solvers import beam as bm

#: version of the saved reduced beam data
REDUCED_BEAM_VERSION = 1


# =============================================================================
# Snapshots (§10.3)
# =============================================================================

def _real_columns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    return np.hstack([X.real, X.imag]) if np.iscomplexobj(X) else X


def _largest_singular_value(X: np.ndarray) -> float:
    if not X.size:
        return 0.0
    gram = X.T @ X
    return float(np.sqrt(max(np.linalg.eigvalsh(gram)[-1], 0.0)))


def pod_snapshots(port_snapshots: np.ndarray, beam_snapshots: np.ndarray,
                  free: np.ndarray) -> np.ndarray:
    """One real snapshot matrix for the ports and the beam (§10.3).

    The beam snapshots are the scattered field e_s; only their free part
    enters (the wall values are the lift, added back after a reduced solve).
    Real and imaginary parts are separate columns, and each family is divided
    by its largest singular value: the port columns are fields per unit modal
    current, the beam columns per unit beam current, and the truncation would
    otherwise drop the smaller family.
    """
    Xb = np.array(beam_snapshots, dtype=complex)
    Xb[~np.asarray(free, dtype=bool), :] = 0.0
    parts = []
    for X in (_real_columns(port_snapshots), _real_columns(Xb)):
        s1 = _largest_singular_value(X)
        if X.shape[1] and s1 > 0:
            parts.append(X / s1)
    return np.hstack(parts)


def missing_beam_reason(snapshots, data) -> str:
    """Why a full-order beam column cannot be reduced (for a warning)."""
    if snapshots is None:
        return ("the full-order sweep kept no field snapshots of the beam, so the reduced "
                "model carries no beam. Solve with store_snapshots=True, and reduce again.")
    return ("the full-order beam data hold no interpolation data (solved by an older "
            "version), so the reduced model carries no beam. Solve again (rerun=True), "
            "and reduce again.")


def staged_beam_inputs(fom_dir, tag: str) -> Optional[Dict]:
    """The beam inputs of a reduction from a section's full-order files in
    ``fom_dir`` (``fds/fom`` or the flat ``fds/foms`` tree, files suffixed
    ``tag``): field snapshots, beam data and the metadata of its S~.  None
    without a beam, or (reported) without the field snapshots of the beam."""
    import h5py
    import cavsim3d.utils.printing as pr
    from cavsim3d.core.persistence import H5Serializer
    fom_dir = Path(fom_dir)
    tilde = bm.load_tilde(fom_dir / "s_tilde" / f"s_tilde_{tag}.h5")
    if tilde is None:
        return None
    snaps = None
    f = fom_dir / "snapshots_beam" / f"snapshots_beam_{tag}.h5"
    if f.exists():
        with h5py.File(f, "r") as fh:
            if "field_snapshots" in fh:
                snaps = H5Serializer.load_dataset(fh["field_snapshots"])
    data = bm.load_beam_data(fom_dir / "matrices" / f"beam_{tag}.h5")
    if snaps is None or data is None or data.get('affine') is None:
        pr.warning(f"  Beam ({tag}): " + missing_beam_reason(snaps, data))
        return None
    if data.get('fingerprint') and data['fingerprint'] != tilde.get('fingerprint'):
        return None
    return {'snapshots': snaps, 'data': data, 'port_modes': tilde['port_modes'],
            'meta': {'names': tilde.get('names'), 'ports': tilde.get('ports'),
                     'fingerprints': tilde.get('fingerprints')}}


# =============================================================================
# The reduced beam column
# =============================================================================

class ReducedBeam:
    """The beam column of one reduced section, evaluated without the mesh.

    Per beam ``j`` (``sources[j]``): ``k_over_w`` (1/v_b), the faces it
    crosses (``crossed``: axis position, reduced unit-phase load f_p and
    B^T e_reg), the reduced lift and load at the interpolation frequencies
    (``H0 + w H1 + w^2 H2``, r x m), the port-voltage part ``Q`` (port modes
    x m) and per path the beam-line value of the lift ``Pg`` (Gauss points
    x m).  Per path ``l`` (``paths[l]``): Gauss points ``s``, weights ``w``,
    ``k_over_w`` and the reduced line values ``C`` (Gauss points x r).
    ``port_modes``: the (port, mode) of each column of the section's B_r.
    """

    def __init__(self, *, setup: Dict, fingerprint: str, port_modes, nodes, band, zc: float,
                 sources: List[Dict], paths: List[Dict], meta: Optional[Dict] = None):
        self.setup = setup
        self.fingerprint = fingerprint
        self.port_modes = [(str(p), int(m)) for p, m in port_modes]
        self.nodes = np.asarray(nodes, dtype=float)
        self.band = [float(v) for v in band]
        self.zc = float(zc)
        self.sources = sources
        self.paths = paths
        self.meta = dict(meta or {})

    @property
    def r(self) -> int:
        return int(self.sources[0]['H0'].shape[0]) if self.sources else 0

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    # -- evaluation -----------------------------------------------------------
    def check_band(self, frequencies) -> None:
        """The interpolation of the beam's phase holds in its band only."""
        w = 2 * np.pi * np.asarray(frequencies, dtype=float)
        lo, hi = self.band
        tol = 1e-9 * hi
        if np.any(w < lo - tol) or np.any(w > hi + tol):
            raise ValueError(
                f"The reduced model's beam data hold from {lo / 2e9 / np.pi:.6g} to "
                f"{hi / 2e9 / np.pi:.6g} GHz (the band of its snapshots plus "
                f"{bm.PHASE_BAND_MARGIN:.0%} on each side); the sweep reaches "
                f"{w.min() / 2e9 / np.pi:.6g} to {w.max() / 2e9 / np.pi:.6g} GHz. "
                "Solve the full-order model over a band that covers it, and reduce again.")

    def loads(self, omega: float) -> np.ndarray:
        """Reduced load of every beam at ``omega``, (r, S) (§10.5)."""
        lag = bm.lagrange_values(omega, self.nodes)
        out = np.zeros((self.r, self.n_sources), dtype=complex)
        for j, src in enumerate(self.sources):
            k = omega * src['k_over_w']
            b = np.exp(-1j * k * self.zc) * (
                (src['H0'] + omega * src['H1'] + omega ** 2 * src['H2']) @ lag)
            for s_face, fhat, _breg in src['crossed']:
                b = b + 1j * omega * np.exp(-1j * k * s_face) * fhat
            out[:, j] = b
        return out

    def evaluate(self, frequencies, A: np.ndarray, B: np.ndarray,
                 C: Optional[np.ndarray] = None, D: Optional[np.ndarray] = None) -> Dict:
        """The section's port and beam blocks at ``frequencies`` [Hz] from its
        mass-normalised reduced operators ``A``, ``B`` (and ``C``, ``D``).

        Returns the raw (wave-normalised) ``Z`` (n_f, N, N), ``kZ`` (n_f, N, S),
        ``hZ`` (n_f, L, N), ``zoc`` (n_f, L, S) and the reduced beam columns
        ``y_b`` (n_f, r, S), as the full-order sweep defines them.
        """
        freqs = np.asarray(frequencies, dtype=float)
        self.check_band(freqs)
        r, N = B.shape
        S, L = self.n_sources, self.n_paths
        n_f = len(freqs)
        Z = np.zeros((n_f, N, N), dtype=complex)
        kZ = np.zeros((n_f, N, S), dtype=complex)
        hZ = np.zeros((n_f, L, N), dtype=complex)
        zoc = np.zeros((n_f, L, S), dtype=complex)
        y_b = np.zeros((n_f, r, S), dtype=complex)
        lossy = C is not None or D is not None
        if not lossy:
            lam, Phi = np.linalg.eigh(A)
            PB = Phi.T @ B
        I = np.eye(r)
        Cm = np.zeros((r, r)) if C is None else C
        Dm = np.zeros((r, r)) if D is None else D
        for k, f in enumerate(freqs):
            w = 2 * np.pi * f
            b = self.loads(w)
            if lossy:
                sol = np.linalg.solve(A + 1j * w * Cm - w ** 2 * (I - 1j * Dm),
                                      np.hstack([w * B, b]))
                Y, yb = sol[:, :N], sol[:, N:]
            else:
                d = 1.0 / (lam - w ** 2)
                Y = w * (Phi @ (d[:, None] * PB))
                yb = Phi @ (d[:, None] * (Phi.T @ b))
            y_b[k] = yb
            Z[k] = 1j * (B.T @ Y)
            lag = bm.lagrange_values(w, self.nodes)
            kz = B.T @ yb
            for j, src in enumerate(self.sources):
                kb = w * src['k_over_w']
                kz[:, j] += np.exp(-1j * kb * self.zc) * (src['Q'] @ lag)
                for s_face, _fhat, breg in src['crossed']:
                    kz[:, j] -= np.exp(-1j * kb * s_face) * breg
            kZ[k] = kz
            for i, path in enumerate(self.paths):
                if not len(path['s']):
                    continue
                cw = path['w'] * np.exp(1j * w * path['k_over_w'] * path['s'])
                cvec = cw @ path['C']
                hZ[k, i] = 1j * (cvec @ Y)
                zoc[k, i] = cvec @ yb
                for j, src in enumerate(self.sources):
                    pg = src['Pg'][i]
                    if pg.size:
                        kb = w * src['k_over_w']
                        zoc[k, i, j] += np.exp(-1j * kb * self.zc) * (cw @ (pg @ lag))
        return {'Z': Z, 'kZ': kZ, 'hZ': hZ, 'zoc': zoc, 'y_b': y_b}

    # -- persistence ----------------------------------------------------------
    def save(self, path) -> None:
        import h5py
        from cavsim3d.core.persistence import H5Serializer
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["version"] = REDUCED_BEAM_VERSION
            f.attrs["setup"] = json.dumps(self.setup)
            f.attrs["fingerprint"] = str(self.fingerprint)
            f.attrs["port_modes"] = json.dumps([[p, m] for p, m in self.port_modes])
            f.attrs["meta"] = json.dumps(self.meta, default=float)
            f.attrs["zc"] = self.zc
            f.create_dataset("band", data=np.asarray(self.band))
            f.create_dataset("nodes", data=self.nodes)
            for j, src in enumerate(self.sources):
                g = f.create_group(f"source_{j + 1}")
                g.attrs["k_over_w"] = float(src['k_over_w'])
                for name in ("H0", "H1", "H2", "Q"):
                    H5Serializer.save_dataset(g, name, np.asarray(src[name]))
                for i, pg in enumerate(src['Pg']):
                    H5Serializer.save_dataset(g, f"Pg_{i + 1}", np.asarray(pg))
                g.attrs["n_crossed"] = len(src['crossed'])
                for c, (s_face, fhat, breg) in enumerate(src['crossed']):
                    gc = g.create_group(f"crossed_{c + 1}")
                    gc.attrs["s_face"] = float(s_face)
                    H5Serializer.save_dataset(gc, "fhat", np.asarray(fhat))
                    H5Serializer.save_dataset(gc, "breg", np.asarray(breg))
            for i, p in enumerate(self.paths):
                g = f.create_group(f"path_{i + 1}")
                g.attrs["k_over_w"] = float(p['k_over_w'])
                g.create_dataset("s", data=np.asarray(p['s']))
                g.create_dataset("w", data=np.asarray(p['w']))
                g.create_dataset("C", data=np.asarray(p['C']))

    @classmethod
    def load(cls, path) -> Optional["ReducedBeam"]:
        import h5py
        from cavsim3d.core.persistence import H5Serializer
        path = Path(path)
        if not path.exists():
            return None
        with h5py.File(path, "r") as f:
            if "nodes" not in f:            # the full-order beam data, not a reduced one
                return None
            sources, paths = [], []
            j = 1
            while f"source_{j}" in f:
                g = f[f"source_{j}"]
                src = {'k_over_w': float(g.attrs["k_over_w"])}
                for name in ("H0", "H1", "H2", "Q"):
                    src[name] = H5Serializer.load_dataset(g[name])
                src['Pg'] = []
                i = 1
                while f"Pg_{i}" in g:
                    src['Pg'].append(H5Serializer.load_dataset(g[f"Pg_{i}"]))
                    i += 1
                src['crossed'] = []
                for c in range(int(g.attrs.get("n_crossed", 0))):
                    gc = g[f"crossed_{c + 1}"]
                    src['crossed'].append((float(gc.attrs["s_face"]),
                                           H5Serializer.load_dataset(gc["fhat"]),
                                           H5Serializer.load_dataset(gc["breg"])))
                sources.append(src)
                j += 1
            i = 1
            while f"path_{i}" in f:
                g = f[f"path_{i}"]
                paths.append({'k_over_w': float(g.attrs["k_over_w"]), 's': g["s"][()],
                              'w': g["w"][()], 'C': g["C"][()]})
                i += 1
            return cls(setup=json.loads(f.attrs["setup"]),
                       fingerprint=str(f.attrs["fingerprint"]),
                       port_modes=json.loads(f.attrs["port_modes"]),
                       nodes=f["nodes"][()], band=list(f["band"][()]),
                       zc=float(f.attrs["zc"]), sources=sources, paths=paths,
                       meta=json.loads(f.attrs.get("meta", "{}")))


def _project(Vt_dense: np.ndarray, X) -> np.ndarray:
    """V^T X for a sparse or dense X (n x m); ``Vt_dense`` is V (n x r)."""
    if X is None:
        return None
    if sp.issparse(X):
        return np.asarray((X.T @ Vt_dense).T)
    return Vt_dense.T @ X


def reduce_beam(V: np.ndarray, K, M, data: Dict, port_modes, C=None, D=None,
                meta: Optional[Dict] = None) -> ReducedBeam:
    """Project the full-order beam data of one section (``data``: as
    :func:`cavsim3d.solvers.beam.load_beam_data` returns it, with its
    ``affine`` part) onto the reduced basis ``V`` = W Q_L^-1 (§10.4).

    ``K``, ``M`` (``C``, ``D``) are the section's full matrices, wall rows
    and columns included: the lift's load is -V^T A(w) g(w), and V vanishes
    on the walls.  ``port_modes``: the (port, mode) of each column of B.
    """
    aff = data.get('affine')
    if aff is None:
        raise ValueError("The beam data hold no interpolation data: the sweep kept no "
                         "field snapshots.")
    setup = bm.BeamSetup.from_dict(data['setup'])
    r = V.shape[1]

    def proj_mat(Mat, G):
        return None if Mat is None else _project(V, sp.csr_matrix(Mat) @ G)

    sources = []
    for j, line in enumerate(setup.sources):
        src = aff['sources'][j]
        G = src['G']
        m = G.shape[1]
        H0 = -proj_mat(K, G)
        H1 = np.zeros((r, m), dtype=complex)
        H2 = proj_mat(M, G).astype(complex)
        if src.get('F1') is not None:
            H1 = H1 + _project(V, src['F1'])
            H2 = H2 + _project(V, src['F2'])
        if C is not None:
            H1 = H1 - 1j * proj_mat(C, G)
        if D is not None:
            H2 = H2 - 1j * proj_mat(D, G)
        crossed = [(face['s_face'], V.T @ np.asarray(face['f_unit'][j], dtype=float),
                    np.asarray(face['b_reg'][j], dtype=float))
                   for face in data['faces'] if face['f_unit'][j] is not None]
        sources.append({'k_over_w': 1.0 / (line.beta * bm.c0), 'H0': H0, 'H1': H1,
                        'H2': H2, 'Q': np.asarray(src['Q']), 'Pg': list(src['Pg']),
                        'crossed': crossed})
    paths = []
    for line, p in zip(setup.paths, data['paths']):
        P = sp.csc_matrix(p['P'])
        paths.append({'k_over_w': 1.0 / (line.beta * bm.c0), 's': np.asarray(p['s']),
                      'w': np.asarray(p['w']),
                      'C': np.asarray((P.T @ V)) if P.shape[1] else np.zeros((0, r))})
    return ReducedBeam(setup=data['setup'], fingerprint=data['fingerprint'],
                       port_modes=port_modes, nodes=aff['nodes'], band=aff['band'],
                       zc=aff['zc'], sources=sources, paths=paths, meta=meta)


# =============================================================================
# Generalised matrices of a reduced section, and joins (§10.7)
# =============================================================================

def section_tilde(rb: ReducedBeam, frequencies, A, B, C=None, D=None,
                  zref: Optional[Callable] = None, zwave: Optional[Callable] = None) -> Dict:
    """S~ and Z~ of one reduced section at ``frequencies`` [Hz].

    ``zref(port, mode, f)`` is the reference impedance of a port mode,
    ``zwave`` the wave impedance its mode was normalised to: a port mode
    referred to another impedance (a TEM line) is rescaled as the reported Z
    is.  The labels and metadata of the section's full-order S~ are kept.
    """
    freqs = np.asarray(frequencies, dtype=float)
    ev = rb.evaluate(freqs, A, B, C, D)
    pm = rb.port_modes
    scale = np.ones(len(pm))
    if zref is not None and zwave is not None and len(freqs):
        for i, (p, m) in enumerate(pm):
            try:
                zw, zt = zwave(p, m, freqs[0]), zref(p, m, freqs[0])
            except Exception:
                continue
            if zw is not None and abs(zw) > 1e-12:
                scale[i] = abs(zt) / abs(zw)
    dsc = np.sqrt(scale)
    Z = ev['Z'] * np.outer(dsc, dsc)[None]
    kZ = dsc[None, :, None] * ev['kZ']
    hZ = ev['hZ'] * dsc[None, None, :]
    Zt = bm.z_tilde(Z, kZ, hZ, ev['zoc'])
    St, zr = None, None
    if zref is not None:
        Zref = np.array([np.diag([zref(p, m, f) for p, m in pm]) for f in freqs])
        St = bm.s_tilde(Z, kZ, hZ, ev['zoc'], Zref)
        zr = np.array([np.diag(z) for z in Zref])
    setup = bm.BeamSetup.from_dict(rb.setup)
    numbers: Dict[str, int] = {}
    labels = [f"{numbers.setdefault(p, len(numbers) + 1)}({m + 1})" for p, m in pm]
    rows, cols = bm.matrix_labels(labels, setup)
    meta = rb.meta
    return {'Z_tilde': Zt, 'S_tilde': St, 'rows': rows, 'cols': cols,
            'frequencies': freqs.copy(),
            'names': meta.get('names') or {lab: line.name for lab, line in
                                          zip(setup.path_labels, setup.paths)},
            'setup': rb.setup, 'fingerprint': rb.fingerprint,
            'summary': {'reduced': True, 'r': rb.r},
            'port_modes': list(pm), 'zref': zr, 'ports': meta.get('ports'),
            'fingerprints': meta.get('fingerprints'), 'y_b': ev['y_b']}


def join_sections(concat, tildes: Sequence[Dict], setup: "bm.BeamSetup",
                  shifts: Optional[Sequence[float]] = None) -> Dict:
    """S~ (and Z~) of sections joined at their cuts (docs/theory/beam.md §9.9).

    ``concat``: a coupled system whose structure ``i`` is section ``i``, with
    its connections defined; ``tildes[i]``: that section's S~ (with its
    ``port_modes`` and ``zref``) at the common frequencies; ``shifts[i]``:
    the section's position along the beam axis in the frame of the first
    (None: every section in one frame).  A section at z_i sees the beam's
    phase exp(-j k_b z_i): its beam columns get that factor and its path rows
    exp(+j k_b z_i); the beam voltages of the sections add.
    """
    freqs = np.asarray(tildes[0]['frequencies'])
    w = 2 * np.pi * freqs
    k_over_w = [1.0 / (l.beta * bm.c0) for l in setup.sources]
    kp_over_w = [1.0 / (l.beta * bm.c0) for l in setup.paths]
    blocks, block_modes, col_phase, row_phase = [], [], [], []
    for i, t in enumerate(tildes):
        blocks.append(np.asarray(t['S_tilde']))
        block_modes.append([concat.port_mode_map[(i, p, int(m))] for p, m in t['port_modes']])
        z = 0.0 if shifts is None else float(shifts[i])
        col_phase.append(np.exp(-1j * np.outer(w, k_over_w) * z))
        row_phase.append(np.exp(1j * np.outer(w, kp_over_w) * z))
    pairs = []
    for (sa, pa), (sb, pb) in concat.connections:
        n = concat.port_to_mode_range[(sa, pa)][1]
        pairs += [(concat.port_mode_map[(sa, pa, m)], concat.port_mode_map[(sb, pb, m)])
                  for m in range(n)]
    external = list(concat._external_port_modes)
    St = bm.join_s_tilde(blocks, block_modes, pairs, external, col_phase, row_phase)

    ext = [concat._global_to_local[g] for g in external]
    labels, numbers = [], {}
    for s_idx, port, m in ext:
        n = numbers.setdefault((s_idx, port), len(numbers) + 1)
        labels.append(f"{n}({m + 1})")
    rowmaps = [{(p, int(m)): r for r, (p, m) in enumerate(t['port_modes'])} for t in tildes]
    Zref = np.array([np.diag([tildes[s]['zref'][k, rowmaps[s][(p, int(m))]]
                              for (s, p, m) in ext]) for k in range(len(freqs))])
    Zt = bm.z_tilde_from_s_tilde(St, Zref)
    return {'S_tilde': St, 'Z_tilde': Zt, 'rows': labels + setup.path_labels,
            'cols': labels + setup.source_labels, 'frequencies': freqs,
            'names': {lab: line.name for lab, line in zip(setup.path_labels, setup.paths)},
            'setup': setup.to_dict(), 'fingerprint': setup.fingerprint(),
            'n_external': len(ext)}


class ReducedBeamJoin:
    """The beam of a coupled system of reduced sections: their S~ joined at
    any frequency (§10.7).  ``sections[i]``: the reduced model of structure
    ``i`` -- a dict with ``beam`` (:class:`ReducedBeam`), ``A``, ``B``, ``C``,
    ``D``, ``zref``, ``zwave`` and ``key`` (sections with the same key share
    one evaluation); ``shifts[i]``: its position along the beam axis (None:
    one frame)."""

    def __init__(self, concat, sections: List[Dict], setup: "bm.BeamSetup",
                 shifts: Optional[Sequence[float]] = None, summary: Optional[Dict] = None):
        self.concat = concat
        self.sections = sections
        self.setup = setup
        self.shifts = None if shifts is None else [float(s) for s in shifts]
        self.summary = dict(summary or {})

    def __call__(self, frequencies) -> Dict:
        cache: Dict[str, Dict] = {}
        tildes = []
        for sec in self.sections:
            key = sec['key']
            if key not in cache:
                cache[key] = section_tilde(sec['beam'], frequencies, sec['A'], sec['B'],
                                           sec.get('C'), sec.get('D'),
                                           sec.get('zref'), sec.get('zwave'))
            tildes.append(cache[key])
        out = join_sections(self.concat, tildes, self.setup, self.shifts)
        out['summary'] = dict(self.summary, reduced=True)
        return out
