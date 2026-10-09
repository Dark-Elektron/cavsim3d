"""
Model order reduction with the beam (cavsim3d.rom.beam_reduction,
docs/theory/beam_reduction.md §10).

Validates:
  - the interpolation of the beam's phase: the wall lift interpolated in
    frequency equals the lift on the mesh, within the a-priori bound
  - a reduced model with the beam reproduces the full-order S~ between its
    snapshots: a cell with the beam off the axis (wall lift), a lossy
    dielectric slab (contrast load, complex system) and a side port the beam
    does not cross
  - the reduced beam is saved with the reduced model and read back on
    reopening, without the mesh; a sweep outside its band is refused
  - without a beam a reduced model carries none
  - joins of reduced parts with the beam: copies of a part (live and imported,
    reference and copy), and glued parts, against the full-order joins at the
    same frequencies and the model solved in one piece
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.base import BaseGeometry
from cavsim3d.rom.beam_reduction import ReducedBeam
from cavsim3d.solvers import beam as bm
from netgen.occ import Box, Pnt

from tests.test_beam import CELL_BEAM, CellChain, SlabGuide, StepGuide

TRAIN = dict(fmin=2.6, fmax=3.4, nsamples=9, nportmodes=1, order=3, solver_type="direct")


class SideArm(BaseGeometry):
    """A 60 x 40 mm guide along z with a 90 x 40 mm side arm along x (one solid):
    ports at both ends of the guide and at the end of the arm, which the beam
    does not cross."""

    def __init__(self, maxh=0.01):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        a, b, L, z1, z2, d = 0.06, 0.04, 0.12, 0.035, 0.085, 0.09
        self.geo = Box(Pnt(0, 0, 0), Pnt(a, b, L)) + Box(Pnt(a, 0, z1), Pnt(a + d, b, z2))
        self.geo.mat('vacuum')
        for f in self.geo.faces:
            lo, hi = f.bounding_box
            name = 'default'
            if hi.z - lo.z < 1e-6:
                name = {0.0: 'port1', L: 'port2'}.get(round(lo.z, 6), 'default')
            elif hi.x - lo.x < 1e-6 and round(lo.x, 6) == round(a + d, 6):
                name = 'port3'
            f.name = name
        self.bc = 'default'


def _tree_hash(root) -> str:
    h = hashlib.sha1()
    for f in sorted(Path(root).rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _project(tmp_path, name, geometry, beam):
    p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
    p.geometry = geometry
    p.add_beam('beam', **beam)
    return p


def _rom_against_fom(tmp_path, make, cfg, beam, n_fine=17, tol=1e-8):
    """A reduced model trained on ``cfg`` and the full-order model solved at
    ``n_fine`` frequencies of the same band: (rom, fom result)."""
    p = _project(tmp_path, "train", make(), beam)
    p.fds.solve(**cfg)
    rom = p.fds.fom.reduce(tol=tol)
    rom.solve(cfg['fmin'], cfg['fmax'], n_fine)
    q = _project(tmp_path, "fine", make(), beam)
    q.fds.solve(**dict(cfg, nsamples=n_fine), store_snapshots=False)
    return rom, q.fds.fom


def _assert_close(rom, fom, s_tol, zb_tol, kh_tol):
    A, R = rom.s_tilde, fom.s_tilde
    P = A.shape[1] - 1
    assert rom.tilde_labels == fom.tilde_labels
    assert np.abs(A - R)[:, :P, :P].max() < s_tol
    zb = np.abs(A[:, P, P] - R[:, P, P]) / np.abs(R[:, P, P])
    assert zb.max() < zb_tol
    for block in (np.s_[:, :P, P], np.s_[:, P, :P]):
        assert np.abs(A[block] - R[block]).max() < kh_tol * np.abs(R[block]).max()
    zo, zo_ref = rom.beam_impedance(ports='open'), fom.beam_impedance(ports='open')
    assert np.max(np.abs(zo - zo_ref) / np.abs(zo_ref)) < zb_tol


# --------------------------------------------------------------------------- #
# The interpolation of the beam's phase
# --------------------------------------------------------------------------- #
def test_lift_interpolation_within_bound(tmp_path):
    """The wall lift at any frequency of the band, from its values at the
    Chebyshev points, equals the lift on the mesh (§10.4)."""
    p = _project(tmp_path, "lift", CellChain(n=1), CELL_BEAM)
    p.fds.solve(**TRAIN)
    system = p.fds._beam_systems['global']
    aff = system.affine
    lo, hi = aff['band']
    assert lo < 2 * np.pi * 2.6e9 and hi > 2 * np.pi * 3.4e9
    G = aff['sources'][0]['G']
    assert G.nnz < 0.5 * np.prod(G.shape)                       # the walls only
    rng = np.random.default_rng(1)
    k_over_w = 1.0 / bm.c0
    for w in rng.uniform(lo, hi, 5):
        exact, _ = bm._combine(system.lift_and_load(0, 1.0, k=w * k_over_w, crossed=False))
        interp = np.exp(-1j * w * k_over_w * aff['zc']) * (G @ bm.lagrange_values(w, aff['nodes']))
        assert np.abs(interp - exact).max() < 1e-10 * np.abs(exact).max()


def test_chebyshev_count_meets_bound():
    from scipy.special import jv
    for c in (0.5, 3.0, 15.7, 40.0):
        m = bm.chebyshev_count(c, tol=1e-12)
        assert m > c
        assert 4 * np.sum(np.abs(jv(np.arange(m, m + 200), c))) <= 1e-12
        x = np.linspace(-1, 1, 2001)
        nodes = bm.chebyshev_nodes(-1.0, 1.0, m)
        vals = np.exp(-1j * c * nodes)
        err = max(abs(bm.lagrange_values(t, nodes) @ vals - np.exp(-1j * c * t)) for t in x)
        assert err < 1e-11


# --------------------------------------------------------------------------- #
# Reduced models against the full-order model
# --------------------------------------------------------------------------- #
def test_reduced_beam_between_snapshots(tmp_path):
    """Beam off the axis of a cell: the lift on the walls carries the beam."""
    rom, fom = _rom_against_fom(tmp_path, lambda: CellChain(n=1), TRAIN, CELL_BEAM)
    _assert_close(rom, fom, s_tol=1e-8, zb_tol=1e-5, kh_tol=1e-4)    # measured 3e-7 (z_b)
    assert rom.has_beam and rom.s_tilde.shape == (17, 3, 3)
    np.testing.assert_allclose(rom.s_tilde[:, :2, :2], rom._S_matrix, atol=1e-12)


@pytest.mark.parametrize("tand", [0.0, 0.05])
def test_reduced_beam_with_a_dielectric(tmp_path, tand):
    """A dielectric slab (lossless: real system; lossy: complex): the contrast
    load and its factors w, w^2."""
    def slab():
        g = SlabGuide()
        g.set_materials({'slab': dict(eps_r=4.0, **({'tan_delta': tand} if tand else {}))})
        return g
    cfg = dict(fmin=0.5, fmax=1.5, nsamples=9, nportmodes=1, order=2, solver_type='direct')
    rom, fom = _rom_against_fom(tmp_path, slab, cfg, dict(x=0.025, y=0.015))
    _assert_close(rom, fom, s_tol=1e-8, zb_tol=1e-5, kh_tol=1e-4)    # measured 4e-8


def test_reduced_beam_side_port(tmp_path):
    """A port face along the beam, which it does not cross: its load and its
    share of k_Z carry the phase across the face."""
    cfg = dict(fmin=2.8, fmax=3.4, nsamples=9, nportmodes=2, order=2, solver_type='direct')
    rom, fom = _rom_against_fom(tmp_path, SideArm, cfg, dict(x=0.03, y=0.02))
    _assert_close(rom, fom, s_tol=1e-8, zb_tol=1e-5, kh_tol=1e-4)    # measured 1.2e-7


# --------------------------------------------------------------------------- #
# Files, reopening, limits
# --------------------------------------------------------------------------- #
def test_reduced_beam_saved_and_reopened(tmp_path, monkeypatch):
    p = _project(tmp_path, "saved", CellChain(n=1), CELL_BEAM)
    p.fds.solve(**TRAIN)
    rom = p.fds.fom.reduce(tol=1e-8)
    rom.solve(2.6, 3.4, 11)
    St = rom.s_tilde
    rom_dir = tmp_path / "saved" / "fds" / "fom" / "rom"
    for f in ("matrices/beam_global.h5", "s_tilde/s_tilde_global.h5",
              "z_tilde/z_tilde_global.h5", "snapshots_beam/snapshots_beam_global.h5"):
        assert (rom_dir / f).exists(), f

    # the reduced beam needs no mesh: evaluated from the files alone
    rb = ReducedBeam.load(rom_dir / "matrices" / "beam_global.h5")
    import h5py
    with h5py.File(rom_dir / "matrices" / "A_r_global.h5") as f:
        A = f["data"][()]
    with h5py.File(rom_dir / "matrices" / "B_r_global.h5") as f:
        B = f["data"][()]
    ev = rb.evaluate(rom.frequencies, A, B)
    np.testing.assert_allclose(ev['zoc'][:, 0, 0], rom.z_tilde[:, 2, 2], rtol=1e-10)
    # a long sweep is evaluated in pieces: the same values
    from cavsim3d.rom import beam_reduction as brom
    monkeypatch.setattr(brom, "EVAL_CHUNK_BYTES", 1.0)
    for key, val in rb.evaluate(rom.frequencies, A, B).items():
        np.testing.assert_allclose(val, ev[key], rtol=0,
                                   atol=1e-12 * np.abs(ev[key]).max(), err_msg=key)
    monkeypatch.undo()

    q = EMProject(name="saved", base_dir=str(tmp_path))
    r2 = q.fds.fom.rom
    assert r2.has_beam
    np.testing.assert_allclose(r2.s_tilde, St, rtol=1e-12)
    r2.solve(2.7, 3.3, 7)                        # a new sweep after reopening
    assert r2.s_tilde.shape == (7, 3, 3)

    # beyond the band of the beam data (snapshots' band + 10 % each side)
    with pytest.raises(ValueError, match="beam data hold"):
        r2.solve(2.0, 3.4, 5)


def test_reduced_model_without_a_beam(tmp_path):
    p = EMProject(name="plain", base_dir=str(tmp_path), overwrite=True)
    p.geometry = CellChain(n=1)
    p.fds.solve(**TRAIN)
    rom = p.fds.fom.reduce(tol=1e-8)
    rom.solve(2.6, 3.4, 5)
    assert not rom.has_beam and not rom._reduced_beam
    rom_dir = tmp_path / "plain" / "fds" / "fom" / "rom"
    assert not (rom_dir / "matrices" / "beam_global.h5").exists()
    assert not (rom_dir / "s_tilde").exists()


# --------------------------------------------------------------------------- #
# Joins of reduced parts
# --------------------------------------------------------------------------- #
def test_reduced_copies_join(tmp_path):
    """A part repeated twice, reduced with the beam: the reduced join equals the
    full-order join at its samples and, between them, the two cells solved in
    one piece as closely as the full-order join does."""
    one = _project(tmp_path, "one", CellChain(n=2), CELL_BEAM)
    one.fds.solve(**dict(TRAIN, nsamples=17), store_snapshots=False)
    ref = one.fds.fom.s_tilde

    chain = EMProject(name="chain", base_dir=str(tmp_path), overwrite=True)
    chain.add("cell", CellChain(n=1), n=2)
    chain.add_beam('beam', **CELL_BEAM)
    chain.fds.solve(**TRAIN)
    fom_join = chain.fds.foms.concatenate().s_tilde
    concat = chain.fds.foms.reduce(tol=1e-8).concatenate()
    concat.solve(2.6, 3.4, 9)
    assert concat.tilde_labels == (['1(1)', '2(1)', 'b(1)'], ['1(1)', '2(1)', 'b(1)'])
    assert np.abs(concat.s_tilde - fom_join).max() < 1e-4 * np.abs(fom_join).max()
    np.testing.assert_allclose(concat.s_tilde[:, :2, :2], concat._S_matrix, atol=1e-10)
    concat.solve(2.6, 3.4, 17)
    St = concat.s_tilde
    assert St.shape == (17, 3, 3)
    assert np.abs(St - ref)[:, :2, :2].max() < 1e-2                   # measured 3.1e-3
    assert np.allclose(St[:, 2, 2], ref[:, 2, 2], rtol=2e-2)           # measured 0.6 %
    assert (tmp_path / "chain" / "fds" / "foms" / "roms" / "matrices" / "beam_cell.h5").exists()
    assert (tmp_path / "chain" / "fds" / "foms" / "roms" / "concat" / "s_tilde" / "s_tilde.h5").exists()


def test_imported_reduced_part_joins(tmp_path):
    """A part imported from a project solved and reduced with the beam: its
    reduced beam is referenced in place or copied, the source is never
    written, and the join is that of the part solved here, also reopened."""
    src = _project(tmp_path, "cell", CellChain(n=1), CELL_BEAM)
    src.fds.solve(**TRAIN)
    src.fds.fom.reduce(tol=1e-8)
    before = _tree_hash(tmp_path / "cell")

    live = EMProject(name="live", base_dir=str(tmp_path), overwrite=True)
    live.add("cell", CellChain(n=1), n=2)
    live.add_beam('beam', **CELL_BEAM)
    live.fds.solve(**TRAIN)
    expected = live.fds.foms.reduce(tol=1e-8).concatenate()
    expected.solve(2.6, 3.4, 13)

    for mode in ('reference', 'copy'):
        p = EMProject(name=f"imp_{mode}", base_dir=str(tmp_path), overwrite=True)
        p.import_project(tmp_path / "cell", name="cell", n=2, mode=mode)
        p.add_beam('beam', **CELL_BEAM)
        p.fds.solve(**TRAIN)
        concat = p.fds.foms.reduce(tol=1e-8).concatenate()
        concat.solve(2.6, 3.4, 13)
        # two reductions of the same results (the part's own project, and the
        # staged files): h, read on the beam line, differs at 7e-7 relative
        assert (np.abs(concat.s_tilde - expected.s_tilde).max()
                < 1e-5 * np.abs(expected.s_tilde).max())
        assert _tree_hash(tmp_path / "cell") == before
        reopened = EMProject(name=f"imp_{mode}", base_dir=str(tmp_path))
        c2 = reopened.fds.foms.roms.concat
        assert c2.has_beam
        np.testing.assert_allclose(c2.s_tilde, concat.s_tilde, atol=1e-12)
        c2.solve(2.7, 3.3, 5)
        assert c2.s_tilde.shape == (5, 3, 3)

    # a project solved with the beam but not reduced: reduced here, read-only
    src2 = _project(tmp_path, "cell_full", CellChain(n=1), CELL_BEAM)
    src2.fds.solve(**TRAIN)
    before2 = _tree_hash(tmp_path / "cell_full")
    p = EMProject(name="imp_full", base_dir=str(tmp_path), overwrite=True)
    p.import_project(tmp_path / "cell_full", name="cell", n=2)
    p.add_beam('beam', **CELL_BEAM)
    p.fds.solve(**TRAIN)
    concat = p.fds.foms.reduce(tol=1e-8).concatenate()
    concat.solve(2.6, 3.4, 13)
    assert (np.abs(concat.s_tilde - expected.s_tilde).max()
            < 1e-5 * np.abs(expected.s_tilde).max())
    assert _tree_hash(tmp_path / "cell_full") == before2


def test_reduced_part_without_beam_joins_ports_only(tmp_path, capsys):
    """A part whose own project was reduced without a beam: the reduced join has
    the port results only, and says why."""
    src = EMProject(name="cell_plain", base_dir=str(tmp_path), overwrite=True)
    src.geometry = CellChain(n=1)
    src.fds.solve(**TRAIN)
    src.fds.fom.reduce(tol=1e-8)
    p = EMProject(name="imp_plain", base_dir=str(tmp_path), overwrite=True)
    p.import_project(tmp_path / "cell_plain", name="cell", n=2)
    p.add_beam('beam', **CELL_BEAM)
    p.fds.solve(**TRAIN)
    concat = p.fds.foms.reduce(tol=1e-8).concatenate()
    assert "no reduced beam column" in capsys.readouterr().out
    concat.solve(2.6, 3.4, 5)
    assert not concat.has_beam and concat.S_dict is not None


def test_reduced_glued_parts_join(tmp_path):
    """Glued parts, reduced per domain with the beam: the reduced join equals the
    full-order join of the parts at its samples."""
    cfg = dict(fmin=3.0, fmax=4.0, nsamples=9, nportmodes=3, order=2, solver_type='direct')
    pa = _project(tmp_path, "parts", StepGuide(), dict(x=0.03, y=0.0125))
    pa.fds.solve(**cfg, per_domain=True, global_method=None)
    fom_join = pa.fds.foms.concatenate()
    concat = pa.fds.foms.reduce(tol=1e-8).concatenate()
    concat.solve(3.0, 4.0, 9)
    z, z_ref = concat.beam_impedance(), fom_join.beam_impedance()
    assert np.max(np.abs(z - z_ref) / np.abs(z_ref)) < 1e-5            # measured 6e-8
    concat.solve(3.0, 4.0, 17)
    assert concat.s_tilde.shape == (17, 7, 7)
