"""Material keys resolve for solid names with underscores; assign_ports
replaces the automatic port names.

The STEP file is written here: a vacuum chamber with three PEC parts cut
out of its walls, labelled CST-style (``component|solid``), no port
entities (so the importer names ports by position).
"""
import re
import warnings
from pathlib import Path

import pytest

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopoDS import TopoDS_Compound

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.importers import OCCImporter


def _box(x0, y0, z0, x1, y1, z1):
    return BRepPrimAPI_MakeBox(gp_Pnt(x0, y0, z0), gp_Pnt(x1, y1, z1)).Shape()


def _write_step(path, labels):
    parts = [_box(0, 40, 60, 20, 60, 80), _box(0, 40, 80, 20, 60, 100),
             _box(80, 40, 120, 100, 60, 140)]
    chamber = _box(0, 0, 0, 100, 100, 200)
    for p in parts:
        chamber = BRepAlgoAPI_Cut(chamber, p).Shape()
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for s in (chamber, *parts):
        builder.Add(comp, s)
    writer = STEPControl_Writer()
    writer.Transfer(comp, STEPControl_AsIs)
    writer.Write(str(path))
    it = iter(labels)
    text = re.sub(r"MANIFOLD_SOLID_BREP\('[^']*'",
                  lambda m: f"MANIFOLD_SOLID_BREP('{next(it)}'", path.read_text())
    path.write_text(text)
    return str(path)


@pytest.fixture(scope="module")
def step_file(tmp_path_factory):
    return _write_step(tmp_path_factory.mktemp("step") / "coupler.step",
                       ['vac|chamber', 'hooks|hook_base', 'hooks|hook_end', 'probe|probe_2'])


def _set_materials(geo, cfg):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        geo.set_materials(cfg)
    return [str(w.message) for w in caught]


@pytest.mark.parametrize("keys", [
    ['hook_base', 'hook_end', 'probe_2'],                          # solid names
    ['hooks', 'probe'],                                            # components
    ['hooks|hook_base', 'hooks|hook_end', 'probe|probe_2'],        # full labels
    ['hook_*', 'probe_*'],                                         # wildcards
])
def test_pec_keys_match_solids_with_underscores(step_file, keys):
    geo = OCCImporter(step_file, unit='mm')
    msgs = _set_materials(geo, {k: 'PEC' for k in keys})
    assert [s.name for s in geo.geo.solids] == ['chamber']
    assert not msgs


def test_dielectric_key_by_component(step_file):
    geo = OCCImporter(step_file, unit='mm')
    geo.set_materials({'hooks': {'eps_r': 4.0}})
    assert geo.get_material('hook_base')['eps_r'] == 4.0
    assert geo.get_material('chamber')['eps_r'] == 1.0


def test_unmatched_key_warns(step_file):
    geo = OCCImporter(step_file, unit='mm')
    msgs = _set_materials(geo, {'hookz': 'PEC', 'hook_base': 'PEC'})
    assert len(msgs) == 1 and "Material key 'hookz' does not match" in msgs[0]


def test_shared_short_name_warns(tmp_path):
    path = _write_step(tmp_path / "shared.step",
                       ['vac|chamber', 'hookA|part', 'hookB|part', 'probe|probe_2'])
    geo = OCCImporter(path, unit='mm')
    msgs = _set_materials(geo, {'hookA': 'PEC'})
    assert any("all become the material 'part'" in m for m in msgs)


def _port_names(geo):
    return {f.name for f in geo.geo.faces if 'port' in f.name}


def test_automatic_ports_only_on_end_faces(step_file):
    """No STEP ports: two ports at the z ends, none on the embedded parts."""
    geo = OCCImporter(step_file, unit='mm')
    ends = {f.name: f.center[2] for f in geo.geo.faces if 'port' in f.name}
    assert ends == pytest.approx({'port1': 0.0, 'port2': 0.2}, abs=1e-9)


def test_assign_ports_replaces_other_ports(step_file):
    geo = OCCImporter(step_file, unit='mm')
    geo.set_materials({'hooks': 'PEC', 'probe': 'PEC'})

    # Ports on the chamber's two x faces instead of the z ends
    faces = geo.list_planar_faces()
    lo = min(faces, key=lambda f: f['center'][0])['index']
    hi = max(faces, key=lambda f: f['center'][0])['index']
    geo.assign_ports({'portA': lo, 'portB': hi})

    assert _port_names(geo) == {'portA', 'portB'}
    assert geo.mesh is None                  # the old mesh had the old names
    geo.generate_mesh(maxh=0.05)
    assert sorted(geo.ports) == ['portA', 'portB']


def test_reopened_project_keeps_assigned_ports(step_file, tmp_path):
    """The geometry replayed on reopening has the assigned ports only."""
    proj = EMProject(name='coupler', base_dir=str(tmp_path))
    geo = proj.import_geometry(step_file, name='coupler', unit='mm', auto_build=False)
    geo.set_materials({'hooks': 'PEC', 'probe': 'PEC'})
    faces = geo.list_planar_faces()
    lo = min(faces, key=lambda f: f['center'][0])['index']
    hi = max(faces, key=lambda f: f['center'][0])['index']
    geo.assign_ports({'port1': lo, 'port2': hi})
    proj.generate_mesh(maxh=0.05)
    del proj

    reopened = EMProject(name='coupler', base_dir=str(tmp_path))
    assert _port_names(reopened.geometry) == {'port1', 'port2'}
    assert not reopened.fds.is_compound
    reopened.generate_mesh(maxh=0.05)
    assert sorted(b for b in set(reopened.mesh.GetBoundaries()) if 'port' in b) == ['port1', 'port2']


def test_split_ports_in_axial_order_and_kept_by_assign_ports():
    step = Path(__file__).resolve().parents[1] / "docs" / "example_models" / "rectangular_waveguide.step"
    geo = OCCImporter(str(step), unit='m', auto_build=False)
    lo, hi = geo.get_bounding_box()
    geo.add_splitting_plane_at_z((lo[2] + hi[2]) / 2)
    geo.split()
    assert _port_names(geo) == {'port1', 'port2', 'port3'}
    assert geo.internal_ports == ['port2']   # the cut, between the two ends

    faces = geo.list_planar_faces()
    zlo = min(faces, key=lambda f: f['center'][2])['index']
    zhi = max(faces, key=lambda f: f['center'][2])['index']
    geo.assign_ports({'port1': zlo, 'port3': zhi})
    assert _port_names(geo) == {'port1', 'port2', 'port3'}
