"""Bodies of revolution: the cavsim2d models revolved about the beam axis.

    proj.create_primitive('elliptical_cavity', name=..., n_cells=..., mid_cell=[...])

Each model keeps its cavsim2d name and constructor (dimensions in mm), builds
its meridian as a Profile and sweeps it 360 degrees about Z. The apertures
become port1 (low z) / port2 and the rest of the surface is the PEC wall
'default'. The checks here are geometric: the meshed volume must equal the
exact volume of revolution of the meridian (Pappus), which catches a wrong
contour, a missing face or an inside-out revolve.
"""
import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.axisymmetric import (
    BLA, Beampipe, Bellows, EllipticalCavity, EllipticalCavityFlatTop, Pillbox, Profile,
    RFGun, SplineCavity, Taper, DegenerateGeometry, revolve)
from cavsim3d.geometry.base import BaseGeometry
from ngsolve import Integrate, CoefficientFunction

TESLA = [42, 42, 12, 19, 35, 57.7, 103.353]
TESLA_END = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
GUN = {'geometry': {
    'y1': 1.5e-2, 'R2': 3e-2, 'T2': 0.7853981633974483, 'L3': 24e-2,
    'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
    'T9': 0.13962634015954636, 'R10': 3e-2, 'T10': 0.6981317007977318,
    'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
SPLINE = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
          'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}


def revolved_volume(profile):
    """Exact volume of revolution of the meridian (Pappus), in m^3."""
    p = profile.contour_points(n=400)
    z, r = p[:, 0], p[:, 1]
    zn, rn = np.roll(z, -1), np.roll(r, -1)
    cross = z * rn - zn * r
    area = 0.5 * cross.sum()
    r_bar = ((r + rn) * cross).sum() / (6 * area)
    return abs(2 * np.pi * area * r_bar)


MODELS = {
    'elliptical_1cell': lambda: EllipticalCavity(1, TESLA, beampipe='both', maxh=0.03),
    'elliptical_3cell': lambda: EllipticalCavity(3, TESLA, TESLA_END, TESLA_END,
                                                 beampipe='both', maxh=0.04),
    'elliptical_chain': lambda: EllipticalCavity(1, TESLA, beampipe='both', chain=2,
                                                 spacing=50, maxh=0.04),
    'flattop': lambda: EllipticalCavityFlatTop(2, TESLA + [20], beampipe='left', maxh=0.04),
    'pillbox': lambda: Pillbox(2, [100, 100, 20, 30, 50], beampipe='both', maxh=0.04),
    'beampipe': lambda: Beampipe(35, 100, maxh=0.03),
    'bellows': lambda: Bellows(35, 8, 12, 1, 2.5, 2.5, L_bp=10, maxh=0.02),
    'taper': lambda: Taper(35, 80, 120, straight_left=20, straight_right=20,
                           R_fillet=10, maxh=0.03),
    'spline_bezier': lambda: SplineCavity({'geometry': SPLINE, 'n_cells': 2,
                                           'beampipe': 'both'}, maxh=0.04),
    'spline_bspline': lambda: SplineCavity({'geometry': SPLINE}, kind='bspline', maxh=0.04),
    'rfgun': lambda: RFGun(GUN, maxh=0.06),
    'rfgun_cathode_pipe': lambda: RFGun(GUN, beampipe='left', maxh=0.06),
}


@pytest.mark.parametrize('name', sorted(MODELS))
def test_revolved_model_has_two_ports_and_the_exact_volume(name):
    geo = MODELS[name]()
    assert geo.mesh is None                          # built, not meshed
    assert geo.ports == ['port1', 'port2']           # read from the solid
    geo.generate_mesh()
    assert sorted(geo.ports) == ['port1', 'port2']
    assert set(geo.boundaries) == {'default', 'port1', 'port2'}
    assert geo.bc == 'default'
    vol = Integrate(CoefficientFunction(1), geo.mesh)
    assert vol == pytest.approx(revolved_volume(geo.profile()), rel=5e-4)


def test_unit_option_gives_the_same_solid():
    mm = EllipticalCavity(2, TESLA, beampipe='both')
    m = EllipticalCavity(2, np.array(TESLA) * 1e-3, beampipe='both', unit='m')
    assert (mm.unit, m.unit) == ('mm', 'm')
    assert np.allclose(mm.profile().contour_points(), m.profile().contour_points())
    gun_mm = {'geometry': {k: v if k.startswith('T') else v * 1e3
                           for k, v in GUN['geometry'].items()}}
    assert np.allclose(RFGun(GUN).profile().contour_points(),            # metres default
                       RFGun(gun_mm, unit='mm').profile().contour_points())
    with pytest.raises(ValueError, match='unit must be'):
        Beampipe(35, 100, unit='inch')


def test_arguments_can_come_from_a_config_dict():
    cfg = {'n_cells': 3, 'mid_cell': TESLA, 'beampipe': 'both', 'beampipe_length': 100}
    cav = EllipticalCavity(config=cfg)
    assert (cav.n_cells, cav.beampipe, cav.beampipe_length) == (3, 'both', 100)
    assert EllipticalCavity(2, config=cfg).n_cells == 2              # explicit wins
    assert EllipticalCavity(config=cfg, n_cells=1).n_cells == 1
    assert Bellows(config=dict(Ri=35, A=8, L_p=12, N_conv=1, R_root=2, R_crest=2)).N_conv == 1
    with pytest.raises(TypeError):
        Beampipe(config={'R': 35, 'L': 100, 'radius': 3})           # unknown key


def test_generate_mesh_keeps_earlier_settings():
    geo = Beampipe(35, 100, maxh=0.03)
    geo.generate_mesh(curvaturesafety=0.5)
    assert geo.maxh == 0.03
    geo.generate_mesh(maxh=0.02)
    assert geo._mesh_params == {'maxh': 0.02, 'curve_order': 3, 'curvaturesafety': 0.5}


def test_rounded_gun_angles_still_close_on_the_axis():
    rounded = {'geometry': {k: round(v, 6) for k, v in GUN['geometry'].items()}}
    geo = RFGun(rounded, maxh=0.06)
    assert geo.ports == ['port1', 'port2']


def test_port1_is_at_the_low_z_end():
    geo = Beampipe(35, 100, maxh=0.03)
    z = {f.name: f.center[2] for f in geo.geo.faces if f.name.startswith('port')}
    assert z['port1'] == pytest.approx(-0.05, abs=1e-9)
    assert z['port2'] == pytest.approx(0.05, abs=1e-9)


def test_closed_end_is_wall_not_port():
    geo = Beampipe(35, 100, ends=('pmc', 'pec'), maxh=0.03)
    assert geo.ports == ['port1']


def test_bla_absorber_is_a_glued_second_material_of_one_domain():
    geo = BLA(35, 150, 80, 5, eps_r=12, tan_delta=0.2, absorber_maxh=0.004, maxh=0.03)
    geo.generate_mesh()
    assert set(geo.mesh.GetMaterials()) == {'vacuum', 'absorber'}
    assert 'interface' in geo.boundaries
    assert geo._domain_materials == {'bla': ['vacuum', 'absorber']}
    assert geo.get_material('absorber')['eps_r'] == 12
    geo.set_materials({'absorber': {'eps_r': 7}})       # the user can override
    assert geo.get_material('absorber')['eps_r'] == 7
    ring = Integrate(CoefficientFunction(1), geo.mesh,
                     definedon=geo.mesh.Materials('absorber'))
    assert ring == pytest.approx(np.pi * (0.040 ** 2 - 0.035 ** 2) * 0.080, rel=1e-3)


def test_elliptical_parameters_follow_cavsim2d():
    cav = EllipticalCavity(3, {'IC': TESLA, 'OC': TESLA_END, 'OC_R': TESLA_END}, maxh=0.05)
    assert cav.parameters['Req_m'] == pytest.approx(103.353)
    assert cav.parameters['Ri_el'] == pytest.approx(39)
    assert cav.half_cells().shape == (6, 7)
    # Default beam pipe (none): the ports sit at the end irises.
    zs = [p[0] for p in cav.profile().points]
    length = 2 * (TESLA_END[5] + 2 * TESLA[5]) * 1e-3
    assert max(zs) - min(zs) == pytest.approx(length)


def test_mismatched_req_is_unified_with_a_warning():
    other = list(TESLA)
    other[6] = 100.0
    with pytest.warns(UserWarning, match='Req differs'):
        cav = EllipticalCavity(2, TESLA, other, other, maxh=0.05)
    assert cav.end_cell_left[6] == pytest.approx(TESLA[6])


def test_degenerate_half_cell_raises():
    bad = [60, 60, 40, 40, 35, 57.7, 103.353]     # ellipses overlap
    with pytest.raises(DegenerateGeometry):
        EllipticalCavity(1, bad, maxh=0.05)


def test_infeasible_bellows_and_taper_raise_before_meshing():
    with pytest.raises(ValueError, match='root corners do not fit'):
        Bellows(35, 8, 6, 1, 5, 1)
    with pytest.raises(ValueError, match='fillet does not fit'):
        Taper(35, 80, 120, straight_left=1, straight_right=20, R_fillet=10)


def test_repeated_bspline_polygon_is_refused():
    with pytest.raises(ValueError, match='stationary corner'):
        SplineCavity({'geometry': SPLINE, 'n_cells': 2}, kind='bspline', maxh=0.05)


def test_aperture_must_be_a_flat_face_from_the_axis():
    p = (Profile('bad').start(0, 0).line_to(0, 0.01, 'PEC')
         .line_to(0.05, 0.02, 'PMC').line_to(0.05, 0, 'PEC').close('AXI'))
    with pytest.raises(ValueError, match='flat aperture'):
        revolve(p)


@pytest.mark.parametrize('kind', ['elliptical_cavity', 'EllipticalCavity', 'Elliptical_Cavity'])
def test_create_primitive_and_reload(tmp_path, kind):
    proj = EMProject('axi', base_dir=str(tmp_path), overwrite=True)
    part = proj.create_primitive(kind, name='cell', n_cells=1, mid_cell=TESLA,
                                 beampipe='both', maxh=0.05)
    assert isinstance(part, EllipticalCavity) and list(proj.parts) == ['cell']

    part.generate_mesh(maxh=0.04, curvaturesafety=1)
    proj.save()

    again = BaseGeometry.load_geometry(proj.project_path)
    assert type(again) is EllipticalCavity
    assert again.parameters == part.parameters
    assert again.beampipe == 'both' and again.ports == ['port1', 'port2']
    # The mesh settings come back without meshing again (the project reloads its mesh).
    assert again.mesh is None
    assert again._mesh_params == {'maxh': 0.04, 'curve_order': 3, 'curvaturesafety': 1}


def test_first_solve_meshes_an_unmeshed_part(tmp_path):
    proj = EMProject('pipe', base_dir=str(tmp_path), overwrite=True)
    proj.create_primitive('beampipe', name='pipe', config={'R': 35, 'L': 100, 'maxh': 0.03})
    assert proj.geometry.mesh is None
    proj.fds.solve(config=dict(fmin=2.6, fmax=3.0, nsamples=2, nportmodes=1, order=1))
    assert proj.mesh is not None and proj.mesh is proj.geometry.mesh


def test_revolved_parts_chain_along_the_main_axis(tmp_path):
    proj = EMProject('line', base_dir=str(tmp_path), overwrite=True)
    proj.create_primitive('beampipe', name='inlet', R=35, L=60, maxh=0.03)
    proj.create_primitive('elliptical_cavity', name='cavity', n_cells=1, mid_cell=TESLA,
                          maxh=0.04)
    assert list(proj.parts) == ['inlet', 'cavity']
    proj.generate_mesh(maxh=0.04)
    ports = [b for b in proj.mesh.GetBoundaries() if b.startswith('port')]
    assert {'port1', 'port2'} <= set(ports)
