"""
Beam excitation (cavsim3d.solvers.beam, docs/theory/beam.md §9).

Validates:
  - a uniform guide: the beam's field is the pipe's own, so Z_par, k and h vanish
  - without a beam the port results are bit-identical; a beam leaves them untouched
  - a beam added to a solved (or reopened) project solves only the beam columns
    and equals a beam solved from the start; removing it drops its results
  - direct and iterative (COCG) beam columns agree
  - a lossy slab: Re Z_par (open ports, below cutoff) is the absorbed power
  - two glued parts: their S~ joined at the cut (foms.concatenate) matches the
    model solved in one piece
  - coupled parts (a part repeated, or imported): each solved once with the beam
    in its own frame and joined through S~ with the phase of its position, match
    the model solved in one piece; an imported part gets its beam columns
    computed in the importing project, its own project is never written
  - the beam definition: positions, labels, names; a beam leaving through a wall
  - an interrupted sweep with a beam resumes; a changed beam starts it afresh
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.core.constants import eps0
from cavsim3d.geometry.base import BaseGeometry
from cavsim3d.solvers import sweep_checkpoint as sc
from cavsim3d.solvers.beam import BeamLine, BeamSetup
from netgen.occ import Box, Glue, Pnt
from ngsolve import Conj, InnerProduct, Integrate

GUIDE = dict(a=0.1, b=0.05, L=0.06667)
CFG = dict(fmin=2.0, fmax=2.4, nsamples=2, nportmodes=1, order=2, solver_type="direct")


def _name_faces(shape, port_planes, walls=None):
    """Faces across the axis at the given z are ports, the rest walls (or
    'interface' where ``walls(face)`` is False)."""
    for f in shape.faces:
        lo, hi = f.bounding_box
        name = 'default'
        if hi.z - lo.z < 1e-5:
            for z0, port in port_planes:
                if abs(lo.z - z0) < 1e-5:
                    name = port
        if name == 'default' and walls is not None and not walls(lo, hi):
            name = 'interface'
        f.name = name


class StepGuide(BaseGeometry):
    """A 60 x 40 mm guide stepping down to 60 x 25 mm (domain 'wide'), then the
    narrow guide (domain 'narrow'); the two meet at port3, 60 mm after the step."""

    def __init__(self, maxh=0.012):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        a, b, b2, L1, Lc, L2 = 0.06, 0.04, 0.025, 0.05, 0.06, 0.05
        p1 = Box(Pnt(0, 0, 0), Pnt(a, b, L1)) + Box(Pnt(0, 0, L1), Pnt(a, b2, L1 + Lc))
        p2 = Box(Pnt(0, 0, L1 + Lc), Pnt(a, b2, L1 + Lc + L2))
        p1.mat('wide')
        p2.mat('narrow')
        self.geo = Glue([p1, p2])
        _name_faces(self.geo, [(0.0, 'port1'), (L1 + Lc + L2, 'port2'), (L1 + Lc, 'port3')])
        self.bc = 'default'


class SlabGuide(BaseGeometry):
    """A 50 x 30 mm guide, 240 mm long, with a 8 mm high slab ('slab') on its
    floor between z = 100 and 140 mm."""

    A, B, L = 0.05, 0.03, 0.24

    def __init__(self, maxh=0.008):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        a, b, L = self.A, self.B, self.L
        slab = Box(Pnt(0, 0, 0.10), Pnt(a, 0.008, 0.14))
        core = Box(Pnt(0, 0, 0), Pnt(a, b, L)) - slab
        core.mat('vacuum')
        slab.mat('slab')
        self.geo = Glue([core, slab])

        def on_wall(lo, hi):
            return any(abs(lo[i] - v) < 1e-5 and abs(hi[i] - v) < 1e-5
                       for i, v in ((0, 0.0), (0, a), (1, 0.0), (1, b)))
        _name_faces(self.geo, [(0.0, 'port1'), (L, 'port2')], walls=on_wall)
        self.bc = 'default'


class SideArmGuide(BaseGeometry):
    """A 60 x 40 mm guide along z with a 50 x 40 mm side arm along x; solids 'main'
    (the guide and 40 mm of the arm) and 'arm' meet inside the arm at port3."""

    def __init__(self, maxh=0.01):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        a, b, L, z1, z2, d, ls = 0.06, 0.04, 0.12, 0.035, 0.085, 0.04, 0.05
        main = Box(Pnt(0, 0, 0), Pnt(a, b, L)) + Box(Pnt(a, 0, z1), Pnt(a + d, b, z2))
        arm = Box(Pnt(a + d, 0, z1), Pnt(a + d + ls, b, z2))
        main.mat('main')
        arm.mat('arm')
        self.geo = Glue([main, arm])
        for f in self.geo.faces:
            lo, hi = f.bounding_box
            name = 'default'
            if hi.z - lo.z < 1e-6:
                name = {0.0: 'port1', L: 'port2'}.get(round(lo.z, 6), 'default')
            elif hi.x - lo.x < 1e-6:
                name = {round(a + d, 6): 'port3', round(a + d + ls, 6): 'port4'}.get(
                    round(lo.x, 6), 'default')
            f.name = name
        self.bc = 'default'


class NarrowingGuide(BaseGeometry):
    """A guide whose second half is half as wide (one solid)."""

    def __init__(self, maxh=0.03):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        self.geo = (Box(Pnt(0, 0, 0), Pnt(0.1, 0.05, 0.05))
                    + Box(Pnt(0, 0, 0.05), Pnt(0.05, 0.05, 0.1)))
        self.geo.mat('vacuum')
        _name_faces(self.geo, [(0.0, 'port1'), (0.1, 'port2')])
        self.bc = 'default'


class CellChain(BaseGeometry):
    """``n`` cells along z, 200 mm each: a 70 x 50 x 40 mm box between two
    60 x 40 mm pipes of 80 mm (one solid)."""

    def __init__(self, n=1, maxh=0.014):
        super().__init__()
        self.n = n
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        lp, L = 0.08, 0.2
        shape = None
        for i in range(self.n):
            z0 = i * L
            cell = (Box(Pnt(0.005, 0.005, z0), Pnt(0.065, 0.045, z0 + lp))
                    + Box(Pnt(0.0, 0.0, z0 + lp), Pnt(0.07, 0.05, z0 + lp + 0.04))
                    + Box(Pnt(0.005, 0.005, z0 + lp + 0.04), Pnt(0.065, 0.045, z0 + L)))
            shape = cell if shape is None else shape + cell
        shape.mat('vacuum')
        self.geo = shape
        _name_faces(self.geo, [(0.0, 'port1'), (self.n * L, 'port2')])
        self.bc = 'default'


CELL_CFG = dict(fmin=2.6, fmax=3.4, nsamples=3, nportmodes=1, order=3, solver_type="direct")
CELL_BEAM = dict(x=0.035, y=0.03)        # 5 mm off the pipe's centre


def _tree_hash(root) -> str:
    h = hashlib.sha1()
    for f in sorted(Path(root).rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _guide(tmp_path, name, beam=True, **guide):
    p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
    p.create_primitive("rwg", name="guide", **dict(GUIDE, maxh=0.025, **guide))
    if beam:
        p.add_beam('beam', x=0.05, y=0.025)
    return p


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
def test_uniform_guide_beam_field_is_the_pipes(tmp_path):
    """A beam in a uniform guide excites nothing: Z_par, k and h vanish."""
    p = EMProject(name="uniform", base_dir=str(tmp_path), overwrite=True)
    p.create_primitive("rwg", name="guide", **dict(GUIDE, maxh=0.01))
    p.add_beam('on', x=0.05, y=0.025)
    p.add_beam('off', x=0.03, y=0.02)
    res = p.fds.solve(**dict(CFG, nsamples=1, order=3))
    fom = p.fds.fom
    St = res['S_tilde']
    assert St.shape == (1, 4, 4)
    assert np.all(np.abs(fom.beam_impedance('on')) < 2e-3)      # measured 4e-4 Ohm
    assert np.all(np.abs(fom.beam_impedance('off')) < 5e-3)     # measured 2e-3 Ohm
    assert np.abs(St[:, :2, 2:]).max() < 1e-3                  # k (measured 8e-5)
    assert np.abs(St[:, 2:, :2]).max() < 1e-3                  # h (measured 2e-4)


def test_lossy_slab_power_balance(tmp_path):
    """Re Z_par with open ports below cutoff is the power the slab absorbs."""
    p = EMProject(name="slab", base_dir=str(tmp_path), overwrite=True)
    geo = SlabGuide()
    geo.set_materials({'slab': {'eps_r': 4.0, 'tan_delta': 0.5}})
    p.geometry = geo
    p.add_beam('beam', x=0.025, y=0.015)
    p.fds.solve(fmin=0.5, fmax=1.0, nsamples=2, nportmodes=1, order=2, solver_type='direct')
    fds = p.fds
    assert fds.fes.is_complex
    zpar = fds.fom.beam_impedance(ports='open')
    for k, f in enumerate(fds.frequencies):
        w = 2 * np.pi * f
        E = fds.fom.beam_field(k)                          # E_s + E_free
        absorbed = Integrate(w * eps0 * 4.0 * 0.5 * InnerProduct(E, Conj(E)).real, fds.mesh,
                             definedon=fds.mesh.Materials('slab'))
        assert zpar[k].real == pytest.approx(absorbed, rel=3e-2)   # measured 0.7 / 2 %

    # lossless: no absorption, no real part (energy conservation)
    q = EMProject(name="slab0", base_dir=str(tmp_path), overwrite=True)
    geo = SlabGuide()
    geo.set_materials({'slab': {'eps_r': 4.0}})
    q.geometry = geo
    q.add_beam('beam', x=0.025, y=0.015)
    q.fds.solve(fmin=0.5, fmax=1.0, nsamples=2, nportmodes=1, order=2, solver_type='direct')
    assert not q.fds.fes.is_complex
    # discretisation only: 0.054 Ohm here, 3e-4 Ohm at order 3
    assert np.all(np.abs(q.fds.fom.beam_impedance(ports='open').real) < 0.05 * zpar.real)


def test_beam_impedance_pole_gives_the_eigenmode_rq(tmp_path):
    """Open ports below cut-off: near a mode, Z_par ~ j (R/Q) w0 / (4 (w0 - w)), with the
    R/Q of the eigenmode (V^2 / (w U)) -- the beam against an independent route."""
    p = EMProject(name="pillbox", base_dir=str(tmp_path), overwrite=True)
    p.create_primitive('pillbox', name='cav', n_cells=1, dims=[100, 100, 30, 0, 100],
                       beampipe='both')
    p.generate_mesh(maxh=0.025, curve_order=4)
    p.add_beam('beam')
    cfg = dict(nportmodes=1, order=2, solver_type='direct')
    p.fds.solve(fmin=1.1, fmax=1.2, nsamples=2, **cfg)
    f0 = p.fds.fom.get_resonant_frequencies(n_modes=1)[0]
    rq = p.fds.fom.get_rq(0)['RQ']
    for off in (-1e-3, 1e-3):
        p.fds.solve(fmin=f0 * (1 + off) / 1e9, fmax=f0 * (1 + off) / 1e9, nsamples=1, **cfg)
        z = p.fds.fom.beam_impedance(ports='open')[0]
        rq_beam = z * 4 * (-off) / 1j
        assert abs(rq_beam.imag) < 1e-3 * rq
        assert rq_beam.real == pytest.approx(rq, rel=1.5e-2)    # measured 0.4 %


def test_glued_parts_joined_match_one_piece(tmp_path):
    """Per-part S~ joined at the cut equals the model solved in one piece."""
    cfg = dict(fmin=3.0, fmax=4.0, nsamples=2, nportmodes=3, order=2, solver_type='direct')
    pa = EMProject(name="parts", base_dir=str(tmp_path), overwrite=True)
    pa.geometry = StepGuide()
    pa.add_beam('beam', x=0.03, y=0.0125)
    pa.fds.solve(**cfg, per_domain=True, global_method=None)
    assert pa.fds.internal_ports == ['port3']
    assert all(f.has_beam for f in pa.fds.foms)
    joined = pa.fds.foms.concatenate()
    assert joined.has_beam and joined.s_tilde.shape == (2, 7, 7)

    pb = EMProject(name="one", base_dir=str(tmp_path), overwrite=True)
    pb.geometry = StepGuide()
    pb.add_beam('beam', x=0.03, y=0.0125)
    res = pb.fds.solve(**cfg, per_domain=False)
    z_join, z_one = joined.beam_impedance(), pb.fds.fom.beam_impedance()
    assert np.all(np.abs(z_join - z_one) < 0.05 * np.abs(z_one))   # measured 1 %
    np.testing.assert_allclose(joined.S_dict['1(1)2(1)'], res['S_dict']['1(1)2(1)'],
                               atol=0.02)
    # z_oc of the join follows from S~ (Z~ derived), close to the one piece
    assert np.all(np.abs(joined.beam_impedance(ports='open')
                         - pb.fds.fom.beam_impedance(ports='open'))
                  < 0.05 * np.abs(pb.fds.fom.beam_impedance(ports='open')))
    # no matrices: other frequencies need reduced models
    with pytest.raises(RuntimeError, match="full-order"):
        joined.solve(fmin=3.0, fmax=4.0, nsamples=5)

    # the join is restored when the project is reopened
    pc = EMProject(name="parts", base_dir=str(tmp_path))
    np.testing.assert_allclose(pc.fds.foms.concat.s_tilde, joined.s_tilde, rtol=1e-10)


def test_coupled_copies_join_like_one_piece(tmp_path):
    """A part repeated (n=2) is solved once, with the beam in its own frame; the
    copies are joined through S~ with the phase of their position along the
    axis.  The joined S~ matches the two cells solved as one piece."""
    one = EMProject(name="one", base_dir=str(tmp_path), overwrite=True)
    one.geometry = CellChain(n=2)
    one.add_beam('beam', **CELL_BEAM)
    one.fds.solve(**CELL_CFG)
    ref = one.fds.fom.s_tilde

    chain = EMProject(name="chain", base_dir=str(tmp_path), overwrite=True)
    chain.add("cell", CellChain(n=1), n=2)
    chain.add_beam('beam', **CELL_BEAM)
    chain.fds.solve(**CELL_CFG)
    joined = chain.fds.foms.concatenate()
    St = joined.s_tilde
    assert St.shape == ref.shape == (3, 3, 3)
    assert joined.tilde_labels == (['1(1)', '2(1)', 'b(1)'], ['1(1)', '2(1)', 'b(1)'])
    assert np.abs(St - ref)[:, :2, :2].max() < 1e-2                     # measured 2.6e-3
    assert np.allclose(St[:, 2, 2], ref[:, 2, 2], rtol=2e-2)            # z_b: 0.6 %
    for block in (np.s_[:, :2, 2], np.s_[:, 2, :2]):                    # k and h: 1.5, 2.4 %
        assert np.abs(St[block] - ref[block]).max() < 6e-2 * np.abs(ref[block]).max()
    # the joined model is saved, and the parts' beam files are in the flat tree
    root = tmp_path / "chain"
    assert (root / "fds" / "foms" / "s_tilde" / "s_tilde_cell.h5").exists()
    assert (root / "fds" / "foms" / "concat" / "s_tilde" / "s_tilde.h5").exists()
    with pytest.raises(RuntimeError, match="full-order"):
        joined.solve(fmin=2.6, fmax=3.4, nsamples=5)


def test_imported_part_gets_its_beam_columns_here(tmp_path):
    """A part imported from a project solved without a beam: its beam columns are
    computed in the importing project from its stored port solutions; the
    source is never written.  Reference and copy imports give the same join as
    the part solved here, also after reopening."""
    src = EMProject(name="cell", base_dir=str(tmp_path), overwrite=True)
    src.geometry = CellChain(n=1)
    src.fds.solve(**CELL_CFG)
    before = _tree_hash(tmp_path / "cell")

    live = EMProject(name="live", base_dir=str(tmp_path), overwrite=True)
    live.add("cell", CellChain(n=1), n=2)
    live.add_beam('beam', **CELL_BEAM)
    live.fds.solve(**CELL_CFG)
    expected = live.fds.foms.concatenate().s_tilde

    for mode in ('reference', 'copy'):
        p = EMProject(name=f"imp_{mode}", base_dir=str(tmp_path), overwrite=True)
        p.import_project(tmp_path / "cell", name="cell", n=2, mode=mode)
        p.add_beam('beam', **CELL_BEAM)
        p.fds.solve(**CELL_CFG)
        assert np.abs(p.fds.foms.concatenate().s_tilde - expected).max() < 1e-9
        assert (tmp_path / f"imp_{mode}" / "fds" / "foms" / "s_tilde" / "s_tilde_cell.h5").exists()
        assert _tree_hash(tmp_path / "cell") == before
        reopened = EMProject(name=f"imp_{mode}", base_dir=str(tmp_path))
        assert np.abs(reopened.fds.foms.concat.s_tilde - expected).max() < 1e-9


def test_coupled_beam_removed_and_added_again(tmp_path):
    """Without the beam the parts keep their port results and lose their beam
    files; the join then needs the reduced models again."""
    chain = EMProject(name="chain", base_dir=str(tmp_path), overwrite=True)
    chain.add("cell", CellChain(n=1), n=2)
    chain.add_beam('beam', **CELL_BEAM)
    chain.fds.solve(**CELL_CFG)
    first = chain.fds.foms.concatenate().s_tilde
    root = tmp_path / "chain"
    s_before = (root / "fds" / "foms" / "s" / "s_cell.h5").read_bytes()

    chain.remove_beam('beam')
    chain.fds.solve(**CELL_CFG)
    assert not list((root / "fds" / "foms").rglob("*tilde*.h5"))
    assert not (root / "fds" / "foms" / "concat").exists()
    assert (root / "fds" / "foms" / "s" / "s_cell.h5").read_bytes() == s_before
    with pytest.raises(NotImplementedError, match="Reduce first"):
        chain.fds.foms.concatenate()

    chain.add_beam('beam', **CELL_BEAM)
    chain.fds.solve(**CELL_CFG)
    assert np.abs(chain.fds.foms.concatenate().s_tilde - first).max() < 1e-9


def test_side_port_the_beam_does_not_cross(tmp_path):
    """A side arm: its port faces (one between the two parts, across x) are not crossed
    by the beam and get n x H_s = -n x H_free with each part's outward normal."""
    cfg = dict(fmin=2.8, fmax=3.4, nsamples=2, nportmodes=3, order=2, solver_type='direct')
    pa = EMProject(name="arm_parts", base_dir=str(tmp_path), overwrite=True)
    pa.geometry = SideArmGuide()
    pa.add_beam('beam', x=0.03, y=0.02)
    pa.fds.solve(**cfg, per_domain=True, global_method=None)
    arm = pa.fds._beam_systems['arm'].faces
    assert [(f.port, f.crossed[0], f.sign) for f in arm] == [('port3', False, -1.0),
                                                             ('port4', False, 1.0)]
    joined = pa.fds.foms.concatenate()

    pb = EMProject(name="arm_one", base_dir=str(tmp_path), overwrite=True)
    pb.geometry = SideArmGuide()
    pb.add_beam('beam', x=0.03, y=0.02)
    pb.fds.solve(**cfg, per_domain=False)
    z_join, z_one = joined.beam_impedance(), pb.fds.fom.beam_impedance()
    assert np.all(np.abs(z_join - z_one) < 5e-3 * np.abs(z_one))       # measured 5e-4
    # (with the internal face's normal taken as NGSolve's, it is 16-27 % off)


# --------------------------------------------------------------------------- #
# Pipeline: invariance, adding later, reopening, removing, solvers
# --------------------------------------------------------------------------- #
def test_no_beam_bit_identical_and_added_later(tmp_path):
    with_beam = _guide(tmp_path, "with_beam")
    ra = with_beam.fds.solve(**CFG)
    later = _guide(tmp_path, "later", beam=False)
    r0 = later.fds.solve(**CFG)
    assert 'S_tilde' not in r0 and not later.fds.fom.has_beam
    # the port results do not depend on the beam
    np.testing.assert_array_equal(ra['S'], r0['S'])
    np.testing.assert_array_equal(ra['Z'], r0['Z'])

    S0, Z0 = r0['S'].copy(), r0['Z'].copy()
    later.add_beam('beam', x=0.05, y=0.025)
    rb = later.fds.solve(**CFG)          # only the beam columns are solved
    np.testing.assert_array_equal(rb['S'], S0)
    np.testing.assert_array_equal(rb['Z'], Z0)
    np.testing.assert_allclose(rb['S_tilde'], ra['S_tilde'], rtol=1e-9, atol=1e-12)

    # reopened, then a beam: the same
    plain = _guide(tmp_path, "plain", beam=False)
    plain.fds.solve(**CFG)
    reopened = EMProject(name="plain", base_dir=str(tmp_path))
    reopened.add_beam('beam', x=0.05, y=0.025)
    rc = reopened.fds.solve(**CFG)
    np.testing.assert_allclose(rc['S_tilde'], ra['S_tilde'], rtol=1e-9, atol=1e-12)


def test_files_reopen_and_remove(tmp_path):
    p = _guide(tmp_path, "files")
    p.fds.solve(**CFG)
    root = tmp_path / "files"
    fom = root / "fds" / "fom"
    for f in ("z_tilde/z_tilde_global.h5", "s_tilde/s_tilde_global.h5",
              "snapshots_beam/snapshots_beam_global.h5", "matrices/beam_global.h5"):
        assert (fom / f).exists(), f
    assert (root / "fds" / "port_modes" / "beam_port_fields.pkl").exists()
    assert json.loads((root / "project.json").read_text())["beam"]["lines"][0]["name"] == 'beam'
    St = p.fds.fom.s_tilde

    q = EMProject(name="files", base_dir=str(tmp_path))
    assert list(q.beams) == ['beam']
    np.testing.assert_array_equal(q.fds.fom.s_tilde, St)
    assert q.fds.fom.tilde_labels == (['1(1)', '2(1)', 'b(1)'], ['1(1)', '2(1)', 'b(1)'])
    q.fds.solve(**CFG)                                  # same request: nothing to do
    assert q.fds.fom.has_beam

    S_ports = q.fds.fom._S_matrix.copy()
    q.remove_beam('beam')
    r = q.fds.solve(**CFG)                              # beam results dropped
    assert 'S_tilde' not in r and not q.fds.fom.has_beam
    np.testing.assert_array_equal(r['S'], S_ports)
    assert not any((fom / d).exists() for d in ("z_tilde", "s_tilde", "snapshots_beam"))
    assert not (fom / "matrices" / "beam_global.h5").exists()
    assert not (root / "fds" / "port_modes" / "beam_port_fields.pkl").exists()


def test_direct_and_iterative_beam_columns_agree(tmp_path):
    rd = _guide(tmp_path, "direct").fds.solve(**CFG)
    ri = _guide(tmp_path, "iterative").fds.solve(**dict(CFG, solver_type='iterative'))
    scale = np.abs(rd['S_tilde']).max()
    np.testing.assert_allclose(ri['S_tilde'], rd['S_tilde'], atol=1e-6 * scale)


# --------------------------------------------------------------------------- #
# The beam definition
# --------------------------------------------------------------------------- #
def test_beam_definition_and_labels(tmp_path):
    p = _guide(tmp_path, "api", beam=False)
    with pytest.raises(TypeError, match="x= and y="):
        p.add_beam('b', z=0.0)
    with pytest.raises(NotImplementedError, match="beta = 1"):
        p.add_beam('slow', beta=0.9)
    p.add_beam('beam', x=0.05, y=0.025)
    p.add_beam('beam', x=0.05, y=0.026)                 # same name: replaced
    p.add_beam_path('probe', x=0.06, y=0.025)
    with pytest.raises(ValueError, match="beam path"):
        p.add_beam('probe')
    assert list(p.beams) == ['beam'] and list(p.beam_paths) == ['beam', 'probe']
    assert p.beams['beam'].point == (0.05, 0.026, 0.0)

    p.fds.solve(**CFG)
    fom = p.fds.fom
    rows, cols = fom.tilde_labels
    assert rows == ['1(1)', '2(1)', 'b(1)', 'b(2)'] and cols == ['1(1)', '2(1)', 'b(1)']
    assert fom.beam_names == {'b(1)': 'beam', 'b(2)': 'probe'}
    d = fom.s_tilde_dict
    np.testing.assert_array_equal(d['b(1)b(1)'], fom.s_tilde[:, 2, 2])     # z_b
    np.testing.assert_array_equal(d['b(1)2(1)'], fom.s_tilde[:, 1, 2])     # k: beam -> port 2
    np.testing.assert_array_equal(d['1(1)b(2)'], fom.s_tilde[:, 3, 0])     # h: port 1 -> probe
    np.testing.assert_array_equal(fom.beam_impedance(), -fom.s_tilde[:, 2, 2])
    np.testing.assert_array_equal(fom.beam_impedance('beam', path='probe'),
                                  fom.beam_impedance('b(1)', path='b(2)'))
    with pytest.raises(KeyError, match="no beam"):
        fom.beam_impedance('nonexistent')

    # a different position makes the stored beam results stale: recomputed
    p.add_beam('beam', x=0.05, y=0.025)
    p.fds.solve(**CFG)
    assert p.fds._beam_fingerprint == p.beam_setup.fingerprint()


def test_beam_leaving_through_a_wall(tmp_path):
    p = EMProject(name="wall", base_dir=str(tmp_path), overwrite=True)
    p.geometry = NarrowingGuide()
    p.add_beam('beam', x=0.075, y=0.025)                # misses the narrow half
    with pytest.raises(ValueError, match="through a wall"):
        p.fds.solve(**dict(CFG, nsamples=1))


def test_mesh_curve_order_with_a_beam(tmp_path):
    """With a beam, generate_mesh() curves to order 4 unless told otherwise."""
    p = EMProject(name="curving", base_dir=str(tmp_path), overwrite=True)
    p.create_primitive('pillbox', name='cav', n_cells=1, dims=[100, 100, 30, 0, 100],
                       beampipe='both')
    p.generate_mesh(maxh=0.06)
    assert p.geometry.curve_order == 3
    p.add_beam('beam')
    p.generate_mesh(maxh=0.06, force=True)
    assert p.geometry.curve_order == 4
    p.generate_mesh(maxh=0.06, curve_order=2, force=True)
    assert p.geometry.curve_order == 2


def test_setup_round_trip():
    s = BeamSetup('Z', [BeamLine('beam', (0.0, 0.0, 0.0)),
                        BeamLine('probe', (0.001, 0.0, 0.0), current=False)])
    t = BeamSetup.from_dict(json.loads(json.dumps(s.to_dict())))
    assert t.fingerprint() == s.fingerprint()
    assert [l.name for l in t.paths] == ['beam', 'probe'] and len(t.sources) == 1
    assert BeamSetup('X', s.paths).fingerprint() != s.fingerprint()


# --------------------------------------------------------------------------- #
# Interrupted sweeps
# --------------------------------------------------------------------------- #
class _Interrupt(Exception):
    pass


def test_interrupted_beam_sweep_resumes(tmp_path, monkeypatch):
    cfg = dict(CFG, nsamples=4)
    ref = _guide(tmp_path, "ref").fds.solve(**cfg)

    p = _guide(tmp_path, "resume")
    real = sc.SweepCheckpoint.write
    count = [0]

    def write(self, *a, **k):
        real(self, *a, **k)
        count[0] += 1
        if count[0] == 2:
            raise _Interrupt
    monkeypatch.setattr(sc.SweepCheckpoint, "write", write)
    with pytest.raises(_Interrupt):
        p.fds.solve(**cfg)
    monkeypatch.setattr(sc.SweepCheckpoint, "write", real)

    computed = [0]

    def counting(self, *a, **k):
        computed[0] += 1
        real(self, *a, **k)
    monkeypatch.setattr(sc.SweepCheckpoint, "write", counting)
    res = p.fds.solve(**cfg)
    assert computed[0] == 2                              # only the missing samples
    np.testing.assert_allclose(res['S_tilde'], ref['S_tilde'], rtol=1e-12, atol=1e-14)

    # a different beam: the samples of the old one are not reused
    q = _guide(tmp_path, "moved")
    monkeypatch.setattr(sc.SweepCheckpoint, "write", write)
    count[0] = 0
    with pytest.raises(_Interrupt):
        q.fds.solve(**cfg)
    monkeypatch.setattr(sc.SweepCheckpoint, "write", counting)
    q.add_beam('beam', x=0.05, y=0.02)
    computed[0] = 0
    q.fds.solve(**cfg)
    assert computed[0] == 4
