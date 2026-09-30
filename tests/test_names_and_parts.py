"""Names of ports, materials and parts, as they come from CAD files and users."""
from types import SimpleNamespace

import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.assembly import Assembly
from cavsim3d.geometry.primitives import CircularWaveguide, RectangularWaveguide
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.solvers.ports import group_port_faces, sorted_logical_ports
from cavsim3d.utils.names import is_port_name, region_pattern


def _guide(L=0.06667):
    return RectangularWaveguide(a=0.1, L=L, maxh=0.06)


def _two_box_mesh(mat_a, mat_b, maxh=0.6, ports=False):
    from netgen.occ import Box, Glue, OCCGeometry, Pnt, Z
    from ngsolve import Mesh
    a = Box(Pnt(0, 0, 0), Pnt(1, 0.5, 1)).mat(mat_a)
    b = Box(Pnt(0, 0, 1), Pnt(1, 0.5, 2)).mat(mat_b)
    if ports:
        for f in list(a.faces) + list(b.faces):
            f.name = "default"
        a.faces.Min(Z).name = "port1"
        b.faces.Max(Z).name = "port2"
    return Mesh(OCCGeometry(Glue([a, b])).GenerateMesh(maxh=maxh))


# --- ports ----------------------------------------------------------------

def test_a_port_is_a_boundary_whose_name_starts_with_port():
    assert all(map(is_port_name, ["port1", "Port2", "port1_air", "portA"]))
    assert not any(map(is_port_name, ["support", "transport_wall", "import", "", None]))
    faces = group_port_faces(["default", "support_ring", "port2_air", "port1",
                              "port2_substrate", "transport", "port10"])
    assert sorted_logical_ports(faces) == ["port1", "port2", "port10"]
    assert faces["port2"] == "port2_air|port2_substrate"


def test_port_names_that_differ_only_in_case_are_refused():
    with pytest.raises(ValueError, match="differ only in case"):
        group_port_faces(["port3", "Port3"])


# --- region patterns ----------------------------------------------------------

def test_region_patterns_match_cad_names_literally():
    names = ["ceramic+window", "cell(1)"]
    mesh = _two_box_mesh(*names)
    for n in names:                        # Mask(): one bit per mesh material
        assert sum(mesh.Materials(region_pattern([n])).Mask()) == 1
        assert sum(mesh.Materials(n).Mask()) == 0         # the raw name selects nothing
    assert sum(mesh.Materials(region_pattern(names)).Mask()) == 2


def test_per_domain_spaces_on_materials_with_regex_characters():
    from ngsolve import CoefficientFunction, Integrate, dx
    mesh = _two_box_mesh("cell(1)", "win+dow", maxh=0.5, ports=True)
    fds = FrequencyDomainSolver(SimpleNamespace(mesh=mesh, bc="default"), order=1)
    assert fds.domains == ["cell(1)", "win+dow"]
    assert all(fds._fes[d].ndof > 0 for d in fds.domains)
    for d in fds.domains:        # the per-domain forms integrate over each box
        volume = Integrate(CoefficientFunction(1.0) * dx(region_pattern([d])), mesh)
        assert volume == pytest.approx(0.5)


@pytest.mark.parametrize("materials, domains", [
    (("cell1", "window"), ["cell1", "window"]),
    (("vacuum", "excellent_dielectric"), ["vacuum", "excellent_dielectric"]),
    (("cell_10", "cell_2"), ["cell_2", "cell_10"]),
])
def test_no_material_is_dropped_from_the_domains(materials, domains):
    solver = SimpleNamespace(geometry=None, mesh=_two_box_mesh(*materials))
    assert FrequencyDomainSolver._detect_domains(solver) == domains


# --- assemblies -----------------------------------------------------------------

def test_connect_places_the_part_after_the_one_it_joins(tmp_path):
    asm = Assembly()
    asm.add("a", _guide())
    asm.add("b", _guide())
    asm.connect("c", _guide(0.03), to_component="a", gap=0.001)
    assert asm._component_order == ["a", "c", "b"]
    conn = next(c for c in asm._connections if c.to_key == "c")
    assert (conn.from_key, conn.from_port, conn.to_port, conn.gap, conn.explicit) == \
        ("a", "port2", "port1", 0.001, True)

    p = EMProject("conn", base_dir=tmp_path, overwrite=True)
    p.geometry = asm
    q = EMProject("conn", base_dir=tmp_path)
    assert q.geometry._component_order == ["a", "c", "b"]


def test_a_gap_survives_reopening(tmp_path):
    asm = Assembly()
    asm.add("a", _guide())
    asm.add("b", _guide(), after=("a", 0.01))
    p = EMProject("gap", base_dir=tmp_path, overwrite=True)
    p.geometry = asm
    q = EMProject("gap", base_dir=tmp_path)
    assert q.geometry._component_order == ["a", "b"]
    assert q.geometry._connections[0].gap == pytest.approx(0.01)


def test_an_old_connect_record_is_not_replayed_twice(tmp_path):
    part = _guide(0.03)
    history = [
        {"op": "__init__"},
        {"op": "add", "name": "a", "geometry_type": "RectangularWaveguide",
         "geometry_history": _guide().get_history(), "n": 1},
        # what connect() used to record: an 'add' and a 'connect' of one part
        {"op": "add", "name": "c", "geometry_type": "RectangularWaveguide",
         "geometry_history": part.get_history(), "n": 1},
        {"op": "connect", "name": "c", "geometry_type": "RectangularWaveguide",
         "geometry_history": part.get_history(), "to_component": "a",
         "from_port": "port1", "to_port": "port2", "gap": 0.0},
    ]
    asm = Assembly._rebuild_from_history(history, tmp_path)
    assert asm._component_order == ["a", "c"]


def test_waveguide_lengths_take_either_name():
    assert RectangularWaveguide(a=0.1, length=0.05, maxh=0.06).L == 0.05
    assert CircularWaveguide(radius=0.05, L=0.1, maxh=0.06).length == 0.1
    with pytest.raises(TypeError, match="not both"):
        RectangularWaveguide(a=0.1, L=0.05, length=0.05)
    with pytest.raises(TypeError, match="missing the length"):
        CircularWaveguide(radius=0.05)
