"""Loaded Q, external Q, R/Q and the cavity figures of merit of eigenmodes;
mesh-curving fallback."""
import numpy as np
import pytest
from scipy.special import j0, j1, jv

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.base import BaseGeometry, _curve_with_fallback
from cavsim3d.solvers.figures_of_merit import beam_line, transverse_kick
from netgen.occ import Axes, Box, Cylinder, Pnt, Z as OCC_Z

C0, EPS0, MU0 = 299792458.0, 8.8541878128e-12, 4e-7 * np.pi
ETA = MU0 * C0
CHI01, CHI11 = 2.404825557695773, 3.831705970207512


def _name_ends(geo):
    for f in geo.faces:
        f.name = "default"
    geo.faces.Min(OCC_Z).name = "port1"
    geo.faces.Max(OCC_Z).name = "port2"
    geo.mat("vacuum")


class _IrisCavity(BaseGeometry):
    """Guide 100 x 50 mm; a cavity between two irises with 20 mm windows.

    *z* = (z0, z1) keeps only that slice of it, a part of a joined model."""
    a, b, lin, lc, t, win = 0.10, 0.05, 0.06, 0.10, 0.003, 0.020

    def __init__(self, z=None, mesh=True):
        super().__init__()
        self.z = z
        self.build()
        if mesh:
            self.generate_mesh(maxh=0.015)

    def build(self):
        a, b, lin, lc, t, win = self.a, self.b, self.lin, self.lc, self.t, self.win
        z0, z1 = self.z or (0.0, 2 * lin + lc + 2 * t)
        guide = Box(Pnt(0, 0, z0), Pnt(a, b, z1))
        for zi in (lin, lin + t + lc):
            if z0 < zi < z1:
                guide = guide - (Box(Pnt(0, 0, zi), Pnt(a, b, zi + t))
                                 - Box(Pnt((a - win) / 2, 0, zi), Pnt((a + win) / 2, b, zi + t)))
        self.geo = guide
        _name_ends(self.geo)
        self.bc = "default"


class _Tubes(BaseGeometry):
    """Coaxial cylinders ``[(z0, z1, radius), ...]``; the two ends are ports."""

    def __init__(self, segments, materials=None, maxh=0.015, mesh=True):
        super().__init__()
        self.segments = segments
        self.build()
        if materials:
            self.set_materials(materials)
        if mesh:
            self.generate_mesh(maxh=maxh)

    def build(self):
        geo = None
        for z0, z1, rad in self.segments:
            c = Cylinder(Axes((0, 0, z0), OCC_Z), r=rad, h=z1 - z0)
            geo = c if geo is None else geo + c
        self.geo = geo
        _name_ends(self.geo)
        self.bc = "default"


# pillbox R x L with beam pipes r x Lp on both sides
R, L, r, Lp = 0.10, 0.08, 0.015, 0.08
PILLBOX = [(-Lp, 0.0, r), (0.0, L, R), (L, L + Lp, r)]
SPAN = (-Lp * 0.999, L + Lp * 0.999)


def _solved(tmp_path, geometry, fmin, fmax, **config):
    proj = EMProject(name="p", base_dir=str(tmp_path))
    proj.geometry = geometry
    proj.fds.solve(config=dict(dict(fmin=fmin, fmax=fmax, nsamples=11, order=2,
                                    nportmodes=1, solver_type="direct"), **config))
    return proj, proj.fds.fom.reduce(tol=1e-10)


def _nearest(model, f):
    return int(np.argmin(np.abs(model.get_resonant_frequencies() - f)))


@pytest.fixture(scope="module")
def iris(tmp_path_factory):
    return _solved(tmp_path_factory.mktemp("iris"), _IrisCavity(), 1.8, 2.6)


@pytest.fixture(scope="module")
def pillbox(tmp_path_factory):
    return _solved(tmp_path_factory.mktemp("pillbox"), _Tubes(PILLBOX), 0.8, 2.0)


def test_loaded_q_matches_s21_bandwidth(iris):
    """Q_L of the loaded eigenproblem is the |S21| 3-dB width of the ROM."""
    _, rom = iris
    q = rom.get_external_q(fmin=1.8, fmax=2.6)
    k = int(np.argmax(q['Q_L']))                      # the cavity mode
    f0, q_l = q['frequencies'][k], q['Q_L'][k]
    assert q_l > 500                                  # a weakly coupled mode
    inv = 1 / q['Qext']['port1'][k] + 1 / q['Qext']['port2'][k]
    assert inv == pytest.approx(1 / q_l, rel=1e-9)

    rom.solve(fmin=f0 * (1 - 4 / q_l) / 1e9, fmax=f0 * (1 + 4 / q_l) / 1e9,
              nsamples=8001, rerun=True)
    f = rom.frequencies
    s21 = np.abs(np.asarray(rom.S_dict['1(1)2(1)']))
    above = np.flatnonzero(s21 >= s21.max() / np.sqrt(2))
    assert f[np.argmax(s21)] == pytest.approx(f0, rel=2e-5)
    assert f0 / (f[above[-1]] - f[above[0]]) == pytest.approx(q_l, rel=0.01)


def test_external_q_of_a_joined_model(iris, tmp_path):
    """The concat's loaded eigenproblem gives the Q of the one-piece model."""
    _, rom = iris
    ref = rom.get_external_q(fmin=1.8, fmax=2.6)
    k = int(np.argmax(ref['Q_L']))

    total = 2 * _IrisCavity.lin + _IrisCavity.lc + 2 * _IrisCavity.t
    cut = _IrisCavity.lin / 2                         # a join in the input guide
    proj = EMProject(name="joined", base_dir=str(tmp_path))
    proj.add("stub", _IrisCavity(z=(0.0, cut), mesh=False))
    proj.add("cavity", _IrisCavity(z=(cut, total), mesh=False))
    proj.generate_mesh(maxh=0.015)
    proj.fds.solve(config=dict(fmin=1.8, fmax=2.6, nsamples=11, order=2, nportmodes=1,
                               solver_type="direct", per_domain=True,
                               store_snapshots=True, global_method=None))
    concat = proj.fds.foms.reduce(tol=1e-10).concatenate()
    q = concat.get_external_q(fmin=1.8, fmax=2.6)
    j = int(np.argmax(q['Q_L']))
    assert q['frequencies'][j] == pytest.approx(ref['frequencies'][k], rel=1e-3)
    assert q['Q_L'][j] == pytest.approx(ref['Q_L'][k], rel=0.02)
    for port in ('port1', 'port2'):
        assert q['Qext'][port][j] == pytest.approx(ref['Qext'][port][k], rel=0.02)


def test_loaded_resonances_do_not_depend_on_the_band(tmp_path):
    """Beam pipes wide enough that TE11 is cut off at the middle of the band but
    carries a dipole pair above it away: every closed mode keeps its own loaded
    resonance, found with Z0 at its own frequency, whatever the band."""
    wide_pipes = [(-Lp, 0.0, 0.055), (0.0, L, R), (L, L + Lp, 0.055)]
    _, rom = _solved(tmp_path, _Tubes(wide_pipes), 0.5, 2.0, nportmodes=2, nsamples=15)
    whole = rom.get_external_q(fmin=0.5, fmax=2.0)
    assert len(set(whole['mode_index'])) == len(whole['mode_index'])
    assert np.count_nonzero(whole['Q_L'] < 10) == 2           # both polarisations
    part = rom.get_external_q(fmin=1.6, fmax=2.0)
    for f, q_l, i in zip(part['frequencies'], part['Q_L'], part['mode_index']):
        k = list(whole['mode_index']).index(i)
        assert whole['frequencies'][k] == pytest.approx(f, rel=1e-6)
        assert whole['Q_L'][k] == pytest.approx(q_l, rel=1e-4)

    # the eigenpairs near each closed mode (shift-invert) give what every
    # eigenpair of the loaded problem gives, the strongly damped pair included
    import cavsim3d.solvers.eigen_mixin as em
    calls, real = [], em.nearest_eigs

    def no_shift_invert(*_a, **_k):
        raise em.ArpackNoConvergence("forced", np.array([]), np.array([]))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(em, "nearest_eigs", lambda *a, **k: calls.append(1) or real(*a, **k))
        fast = rom.get_external_q(fmin=0.5, fmax=2.0)
        assert calls                                  # the shift-invert path ran
        mp.setattr(em, "nearest_eigs", no_shift_invert)
        every = rom.get_external_q(fmin=0.5, fmax=2.0)
    assert list(every['mode_index']) == list(fast['mode_index'])
    np.testing.assert_allclose(fast['frequencies'], every['frequencies'], rtol=1e-9)
    np.testing.assert_allclose(fast['Q_L'], every['Q_L'], rtol=1e-6)


def test_rq_of_tm010_against_closed_form(pillbox):
    """Off the axis, outside the beam-pipe holes, the pillbox is exact."""
    proj, rom = pillbox
    f010 = CHI01 * C0 / (2 * np.pi * R)
    i = _nearest(rom, f010)
    w = 2 * np.pi * f010
    T = np.sin(w * L / (2 * C0)) / (w * L / (2 * C0))
    x = 0.03
    exact = 2 * L * T ** 2 / (w * EPS0 * np.pi * R ** 2 * j1(CHI01) ** 2) * j0(CHI01 * x / R) ** 2

    rq_rom = rom.get_rq(i, offset=(x, 0.0), span=SPAN)
    assert rq_rom['RQ'] == pytest.approx(exact, rel=0.01)
    rq_fom = proj.fds.get_rq(_nearest(proj.fds, f010), offset=(x, 0.0), span=SPAN)
    assert rq_fom['RQ'] == pytest.approx(rq_rom['RQ'], rel=1e-4)


def test_figures_of_merit_of_tm010_against_closed_form(pillbox):
    """Wall Q, geometry factor and peak magnetic field of the pillbox TM010."""
    proj, rom = pillbox
    f010 = CHI01 * C0 / (2 * np.pi * R)
    x = 0.03                                          # clear of the pipe holes
    q = proj.fds.get_figures_of_merit(_nearest(proj.fds, f010), offset=(x, 0.0),
                                      span=SPAN, active_length=L)
    w = 2 * np.pi * f010
    T = np.sin(w * L / (2 * C0)) / (w * L / (2 * C0))
    # G = w mu0 L R / (2 (L + R)); H_phi peaks at J1's maximum on the end walls
    assert q["G [Ohm]"] == pytest.approx(CHI01 * ETA * L / (2 * (L + R)), rel=0.01)
    assert q["Bpk/Eacc [mT/MV/m]"] == pytest.approx(
        MU0 * 0.581865 / (ETA * T * j0(CHI01 * x / R)) * 1e9, rel=0.04)

    rs = np.sqrt(MU0 * 2 * np.pi * q["freq [MHz]"] * 1e6 / (2 * 5.96e7))
    assert q["Rs [Ohm]"] == pytest.approx(rs, rel=1e-12)
    assert q["G [Ohm]"] == pytest.approx(q["Q []"] * rs, rel=1e-12)
    assert q["Rsh [MOhm]"] == pytest.approx(q["R/Q [Ohm]"] * q["Q []"] * 1e-6, rel=1e-12)
    assert q["Bpk [mT]"] == pytest.approx(MU0 * q["Hpk [A/m]"] * 1e3, rel=1e-12)
    assert q["U [J]"] == 1.0
    assert "Q_diel []" not in q                       # no lossy material

    # a superconducting wall: Q scales with 1/Rs, G does not change
    sc = proj.fds.get_figures_of_merit(_nearest(proj.fds, f010), offset=(x, 0.0),
                                       span=SPAN, surface_resistance=10e-9)
    assert sc["G [Ohm]"] == pytest.approx(q["G [Ohm]"], rel=1e-9)
    assert sc["Q []"] == pytest.approx(q["G [Ohm]"] / 10e-9, rel=1e-9)

    qr = rom.get_figures_of_merit(_nearest(rom, f010), offset=(x, 0.0), span=SPAN,
                                  active_length=L)
    for key in ("freq [MHz]", "R/Q [Ohm]", "G [Ohm]"):
        assert qr[key] == pytest.approx(q[key], rel=1e-3)
    # peak surface fields are maxima over the mesh, more sensitive than the
    # integrals above: 1e-6 apart with PARDISO, 1.3e-3 with sparsecholesky (macOS)
    for key in ("Epk/Eacc []", "Bpk/Eacc [mT/MV/m]"):
        assert qr[key] == pytest.approx(q[key], rel=5e-3)


def test_transverse_kick_of_tm110(pillbox):
    """Panofsky-Wenzel kick of the dipole: the direct Lorentz force, and the
    closed form up to the beam-pipe holes; ~0 for the monopole."""
    from ngsolve import curl
    proj, _ = pillbox
    fds = proj.fds
    f110 = CHI11 * C0 / (2 * np.pi * R)
    i = _nearest(fds, f110)
    q = fds.get_figures_of_merit(i, span=SPAN)
    w = 2 * np.pi * f110
    T = np.sin(w * L / (2 * C0)) / (w * L / (2 * C0))
    vt = C0 / w * L * T * CHI11 / R / 2               # per unit E0
    u = EPS0 * L * np.pi * R ** 2 * jv(2, CHI11) ** 2 / 4
    assert q["R/Q_t [Ohm]"] == pytest.approx(vt ** 2 / (w * u), rel=0.05)
    assert q["R/Q [Ohm]"] < 1e-3 * q["R/Q_t [Ohm]"]   # no voltage on the axis

    # int (E + c z x B)_t exp(j w z / c) dz on the axis, B = j curl E / w
    freq, x = fds._eigenpair(i, 'global')
    w = 2 * np.pi * freq
    piece = fds._eigen_mode_pieces(x, 'global', 'Z')[0]
    s = beam_line([piece], 2, SPAN, 4001)
    mips = piece.mesh(np.zeros_like(s), np.zeros_like(s), s)
    E = np.asarray(piece.E(mips)).reshape(-1, 3)
    B = 1j * np.asarray(curl(piece.E)(mips)).reshape(-1, 3) / w
    phase = np.exp(1j * w * s / C0)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    v_x = trapezoid((E[:, 0] - C0 * B[:, 1]) * phase, s)
    v_y = trapezoid((E[:, 1] + C0 * B[:, 0]) * phase, s)
    U, _ = fds._eigen_energy(x, 'global', w)
    direct = np.hypot(abs(v_x), abs(v_y)) / np.sqrt(U)
    assert q["Vt [MV]"] * 1e6 == pytest.approx(direct, rel=0.01)
    # per plane, with the phase: V_t,u = j (c / w) dV/du is the Lorentz-force integral
    for key, v in (("Vt_x [MV]", v_x), ("Vt_y [MV]", v_y)):
        assert q[key] * 1e6 == pytest.approx(abs(v) / np.sqrt(U), abs=0.01 * direct)
    assert q["R/Q_t_x [Ohm]"] + q["R/Q_t_y [Ohm]"] == pytest.approx(q["R/Q_t [Ohm]"], rel=1e-12)
    kick = transverse_kick([piece], 2, (0.0, 0.0), s, w)
    assert kick['planes'] == ('x', 'y')
    np.testing.assert_allclose(kick['Vt_planes'], [v_x, v_y], rtol=0,
                               atol=0.01 * np.hypot(abs(v_x), abs(v_y)))

    # the phase of V: a dipole's voltage is odd in the offset, a monopole's even
    d = 0.02
    vp, vm = (fds.get_rq(i, offset=(sx, 0.0), span=SPAN)["V_complex"] for sx in (d, -d))
    assert abs(vp + vm) < 1e-2 * abs(vp)
    i010 = _nearest(fds, CHI01 * C0 / (2 * np.pi * R))
    vp, vm = (fds.get_rq(i010, offset=(sx, 0.0), span=SPAN)["V_complex"] for sx in (d, -d))
    assert abs(vp - vm) < 1e-2 * abs(vp)
    assert fds.get_rq(i010, span=SPAN)["V"] == pytest.approx(
        abs(fds.get_rq(i010, span=SPAN)["V_complex"]), rel=1e-15)

    monopole = fds.get_figures_of_merit(i010, span=SPAN)
    assert monopole["R/Q_t [Ohm]"] < 1e-3 * q["R/Q_t [Ohm]"]


def test_dielectric_q_is_one_over_tan_delta(tmp_path):
    """A uniform filling: Q_diel = 1 / tan_delta, the wall part unchanged."""
    fill = _Tubes(PILLBOX, materials={'*': {'eps_r': 2.0, 'tan_delta': 1e-3}}, maxh=0.02)
    proj, rom = _solved(tmp_path, fill, 0.6, 1.2)
    f = CHI01 * C0 / (2 * np.pi * R * np.sqrt(2.0))
    for model in (proj.fds, rom):
        q = model.get_figures_of_merit(_nearest(model, f), span=SPAN)
        assert q["Q_diel []"] == pytest.approx(1e3, rel=1e-9)
        assert 1 / q["Q []"] == pytest.approx(1 / q["Q_wall []"] + 1 / q["Q_diel []"], rel=1e-9)
        assert q["G [Ohm]"] == pytest.approx(q["Q_wall []"] * q["Rs [Ohm]"], rel=1e-9)


class _LoadedPillbox(BaseGeometry):
    """The pillbox with a lossy ceramic disk inside it."""

    def __init__(self):
        super().__init__()
        self.build()
        self.set_materials({'ceramic': {'eps_r': 4.0, 'tan_delta': 1e-4}})
        self.generate_mesh(maxh=0.02)

    def build(self):
        from netgen.occ import Glue
        outer = _Tubes(PILLBOX, mesh=False).geo
        disk = Cylinder(Axes((0, 0, 0.03), OCC_Z), r=0.05, h=0.02)
        disk.faces.name = "interface"
        disk.mat("ceramic")
        outer = outer - disk
        outer.mat("vacuum")
        self.geo = Glue([outer, disk])
        self.bc = "default"


def test_material_shares_and_dielectric_q(tmp_path):
    """A lossy region: its energy share and Q_diel = 1 / (share tan_delta)."""
    from ngsolve import InnerProduct, Integrate
    proj, _ = _solved(tmp_path, _LoadedPillbox(), 0.7, 1.3, nsamples=7)
    fds = proj.fds
    q = fds.get_figures_of_merit(0)
    freq, x = fds._eigenpair(0, 'global')
    E = fds._reconstruct_eigenmode_field(x, 'global')
    mesh = fds.mesh
    part = {m: e * Integrate(InnerProduct(E, E), mesh, definedon=mesh.Materials(m))
            for m, e in (('vacuum', 1.0), ('ceramic', 4.0))}
    share = part['ceramic'] / (part['ceramic'] + part['vacuum'])
    assert q["U_frac_ceramic []"] == pytest.approx(share, rel=1e-9)
    assert q["U_frac_ceramic []"] + q["U_frac_vacuum []"] == pytest.approx(1.0, rel=1e-12)
    assert q["Q_diel []"] == pytest.approx(1 / (share * 1e-4), rel=1e-9)
    assert q["Epk_ceramic [MV/m]"] > 0


def test_figures_of_merit_of_a_netlist_chain(tmp_path):
    """Two coupled copies of a cell give the 2-cell structure solved in one piece."""
    rp, lp = 0.03, 0.02
    cell = [(0.0, lp, rp), (lp, lp + L, R), (lp + L, 2 * lp + L, rp)]
    two = cell + [(z0 + 2 * lp + L, z1 + 2 * lp + L, rad) for z0, z1, rad in cell]
    cfg = dict(fmin=0.9, fmax=1.4, nsamples=11, order=2, nportmodes=3, solver_type="direct")

    _, ref = _solved(tmp_path / "one", _Tubes(two, maxh=0.02), **cfg)
    proj = EMProject(name="chain", base_dir=str(tmp_path / "chain"))
    proj.add("cell", _Tubes(cell, mesh=False), n=2)
    proj.generate_mesh(maxh=0.02)
    proj.fds.solve(config=cfg)
    concat = proj.fds.foms.reduce(tol=1e-10).concatenate()

    f010 = CHI01 * C0 / (2 * np.pi * R)
    pair = [np.flatnonzero(np.abs(m.get_resonant_frequencies() - f010) < 0.05 * f010)
            for m in (ref, concat)]
    assert [len(p) for p in pair] == [2, 2]           # the 0 and pi modes
    assert concat.get_cell_coupling(*pair[1]) == pytest.approx(
        ref.get_cell_coupling(*pair[0]), rel=0.02)
    kw = dict(offset=(0.05, 0.0), active_length=2 * L)
    q_ref = ref.get_figures_of_merit(pair[0][1], **kw)
    q = concat.get_figures_of_merit(pair[1][1], **kw)
    for key in ("freq [MHz]", "R/Q [Ohm]", "G [Ohm]", "Q []"):
        assert q[key] == pytest.approx(q_ref[key], rel=5e-3)


class _FakeMesh:
    """Curving fails above *works_up_to*."""

    def __init__(self, works_up_to):
        self.works_up_to, self.curved, self.tried = works_up_to, None, []

    def Curve(self, order):
        self.tried.append(order)
        if order > self.works_up_to:
            raise RuntimeError("StdFail_NotDone: GeomAPI_ProjectPointOnCurve::NearestPoint")
        self.curved = order


def test_rom_lists_the_modes_near_its_training_band(iris):
    # far from its band a ROM has spurious modes: by default it lists those
    # within 10 % of the band's edges, and get_eigenmode(i) counts that list
    proj, rom = iris
    f = rom.get_resonant_frequencies() / 1e9
    assert len(f) and f.min() >= 0.9 * 1.8 and f.max() <= 1.1 * 2.6
    wide = rom.get_resonant_frequencies(fmin=0) / 1e9
    assert len(wide) > len(f) and np.all(np.isin(np.round(f, 9), np.round(wide, 9)))
    for i in range(len(f)):
        assert rom.get_eigenmode(i)[0] / 1e9 == pytest.approx(f[i], rel=1e-9)
    # the band is saved with the model, so a reopened project lists the same
    again = EMProject(name="p", base_dir=str(proj.base_dir)).fds.fom.rom
    assert np.allclose(again.get_resonant_frequencies() / 1e9, f)


def test_curving_falls_back_to_a_lower_order():
    mesh = _FakeMesh(works_up_to=2)
    with pytest.warns(UserWarning, match="order 3 failed.*order 2"):
        assert _curve_with_fallback(mesh, 3) == 2
    assert mesh.curved == 2
    assert _curve_with_fallback(_FakeMesh(3), 3) == 3
    with pytest.raises(RuntimeError, match="NearestPoint"):
        _curve_with_fallback(_FakeMesh(0), 3)


def test_curving_fallback_names_the_sliver_edges():
    from netgen.occ import Box, Pnt
    from cavsim3d.geometry.base import tiny_edges
    # a step of 2 micrometres where two boxes meet: four edges 2 um long
    shape = (Box(Pnt(0, 0, 0), Pnt(0.1, 0.05, 0.2))
             + Box(Pnt(0, 0, 0.2), Pnt(0.1, 0.05 + 2e-6, 0.3)))
    found = tiny_edges(shape)
    assert len(found) == 2 and all(length == pytest.approx(2e-6) for length, _ in found)
    assert {round(p[0], 6) for _, p in found} == {0.0, 0.1}
    assert tiny_edges(Box(Pnt(0, 0, 0), Pnt(0.1, 0.05, 0.2))) == []
    with pytest.warns(UserWarning, match=r"2 sliver edge\(s\).*2\.0 um at \(0\.00, 50\.00, 200\.00\) mm"):
        assert _curve_with_fallback(_FakeMesh(works_up_to=2), 4, shape=shape) == 2


def test_curving_starts_at_the_order_reached_before():
    import warnings
    mesh = _FakeMesh(works_up_to=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")              # nothing failed: no warning
        assert _curve_with_fallback(mesh, 4, known=2) == 2
    assert mesh.tried == [2] and mesh.curved == 2
    # a known order that fails after all: the usual fallback from there
    mesh = _FakeMesh(works_up_to=1)
    with pytest.warns(UserWarning, match="order 2 failed.*order 1"):
        assert _curve_with_fallback(mesh, 4, known=2) == 1
