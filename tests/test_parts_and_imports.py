"""The project's part list and project imports (reference / copy).

    proj.import_geometry(path, name=...)   # a CAD part
    proj.create_primitive(kind, name=...)  # a primitive part
    proj.import_project(path, mode=...)    # another project as a part
    proj.add(name, part)                   # anything else

One part is the project's geometry itself; a second part turns the geometry
into a chain (an Assembly) along ``main_axis``.  The same name replaces a part.
"""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.assembly import Assembly
from cavsim3d.geometry.primitives import RectangularWaveguide

STEP = (Path(__file__).resolve().parents[1] / "docs" / "example_models"
        / "rectangular_waveguide.step")
A, SEC_L, MAXH = 0.1, 0.06667, 0.06
CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)


def _section():
    return RectangularWaveguide(a=A, L=SEC_L, maxh=MAXH)


def _solved_source(tmp_path, name="source"):
    src = EMProject(name, base_dir=str(tmp_path), overwrite=True)
    src.geometry = _section()
    src.fds.solve(config=CFG)
    src.fds.fom.reduce(tol=1e-9)
    src.save()
    return tmp_path / name


def _module(tmp_path, src, name, mode, n=3):
    m = EMProject(name, base_dir=str(tmp_path), overwrite=True)
    m.import_project(src, name="sec", mode=mode, n=n)
    m.fds.solve(config=CFG)
    concat = m.fds.foms.reduce(tol=1e-9).concatenate()
    res = concat.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=5))
    return m, concat, res["S"]


class TestPartList:

    def test_one_part_is_the_geometry_and_same_name_replaces(self, tmp_path):
        p = EMProject("one", base_dir=str(tmp_path), overwrite=True)
        g = p.import_geometry(STEP, unit="m")
        assert p.geometry is g and list(p.parts) == ["rectangular_waveguide"]
        g2 = p.import_geometry(STEP, unit="m")          # re-run of the same cell
        assert p.geometry is g2 and len(p.parts) == 1

    def test_second_part_makes_a_chain_in_list_order(self, tmp_path):
        p = EMProject("two", base_dir=str(tmp_path), overwrite=True)
        p.import_geometry(STEP, name="head", unit="m")
        p.create_primitive("rwg", name="tail", a=0.02286, b=0.01016, L=0.05, maxh=0.01)
        assert isinstance(p.geometry, Assembly)
        assert list(p.parts) == ["head", "tail"]
        p.geometry.compute_layout()
        z = [p.geometry._components[k].transform.translation[2] for k in ("head", "tail")]
        assert z[1] > z[0]                                # appended along +Z
        text = p.geometry.describe_layout()
        assert "Main axis: Z (default)" in text and "Mesh strategy: glued" in text
        # same name inside the chain: replaced in place, order kept
        p.create_primitive("rwg", name="head", a=0.02286, b=0.01016, L=0.03, maxh=0.01)
        assert list(p.parts) == ["head", "tail"]
        q = EMProject("two", base_dir=str(tmp_path))
        assert list(q.parts) == ["head", "tail"]

    def test_part_name_and_axis_persist(self, tmp_path):
        p = EMProject("named", base_dir=str(tmp_path), overwrite=True)
        p.import_geometry(STEP, name="guide", unit="m")
        p.main_axis = "x"
        q = EMProject("named", base_dir=str(tmp_path))
        assert list(q.parts) == ["guide"] and q.main_axis == "X"

    def test_create_importer_is_deprecated(self, tmp_path):
        p = EMProject("dep", base_dir=str(tmp_path), overwrite=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p.create_importer(STEP, unit="m")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_mesh_strategy(self, tmp_path):
        asm = Assembly()
        asm.add("a", _section())
        asm.add("b", _section())
        assert asm.resolved_mesh_strategy()[0] == "glued"
        asm.set_mesh_strategy("coupled")
        assert asm.resolved_mesh_strategy()[0] == "coupled"
        rep = Assembly()
        rep.add("cell", _section(), n=2)
        assert rep.resolved_mesh_strategy()[0] == "coupled"
        rep.set_mesh_strategy("glued")
        with pytest.raises(ValueError, match="repeated"):
            rep.resolved_mesh_strategy()

    def test_flip_rotates_a_glued_part(self):
        asm = Assembly()
        asm.add("a", _section())
        asm.add("b", _section(), flip=True)
        assert asm._components["b"].transform.rotation == (180.0, 0.0, 0.0)


class TestProjectImport:

    def test_reference_and_copy_give_the_same_result(self, tmp_path):
        src = _solved_source(tmp_path)
        _, c_ref, S_ref = _module(tmp_path, src, "mod_ref", "reference")
        _, c_copy, S_copy = _module(tmp_path, src, "mod_copy", "copy")
        assert np.allclose(S_ref, S_copy, atol=1e-12)
        assert len(c_ref.structures) == 3
        # reference: nothing copied, the source is recorded
        assert not (tmp_path / "mod_ref" / "fds" / "foms" / "matrices" / "K_sec.h5").exists()
        refs = json.loads((tmp_path / "mod_ref" / "fds" / "imports.json").read_text())
        assert refs["sec"]["mode"] == "reference" and refs["sec"]["fingerprint"]
        # copy: the section's files live in the module
        assert (tmp_path / "mod_copy" / "fds" / "foms" / "matrices" / "K_sec.h5").exists()

    def test_source_is_never_written(self, tmp_path):
        src = _solved_source(tmp_path)
        before = {f: f.stat().st_mtime for f in src.rglob("*") if f.is_file()}
        _module(tmp_path, src, "mod_ref", "reference")
        after = {f: f.stat().st_mtime for f in src.rglob("*") if f.is_file()}
        assert after == before

    def test_localize_makes_the_module_stand_alone(self, tmp_path):
        src = _solved_source(tmp_path)
        m, _, _ = _module(tmp_path, src, "mod_loc", "reference")
        assert m.localize() == 1
        mod = tmp_path / "mod_loc"
        assert (mod / "fds" / "foms" / "matrices" / "K_sec.h5").exists()
        assert (mod / "fds" / "foms" / "roms" / "matrices" / "A_r_sec.h5").exists()
        assert json.loads((mod / "fds" / "imports.json").read_text())["sec"]["mode"] == "copy"
        shutil.rmtree(src)
        q = EMProject("mod_loc", base_dir=str(tmp_path))     # reopens without the source
        q.fds.solve(config=CFG)                               # and solves from its copy
        assert len(q.fds.foms.reduce(tol=1e-9).concatenate().structures) == 3

    def test_changed_source_is_reported_on_reopen(self, tmp_path, caplog):
        src = _solved_source(tmp_path)
        _module(tmp_path, src, "mod_stale", "reference")
        s = EMProject("source", base_dir=str(tmp_path))       # re-solve the source
        s.fds.solve(config=dict(CFG, nsamples=5), rerun=True)
        s.fds.fom.reduce(tol=1e-9)
        s.save()
        caplog.clear()
        EMProject("mod_stale", base_dir=str(tmp_path))
        assert "changed since this project was solved" in caplog.text

    def test_missing_reference_errors_helpfully(self, tmp_path):
        src = _solved_source(tmp_path)
        m = EMProject("mod_gone", base_dir=str(tmp_path), overwrite=True)
        m.import_project(src, name="sec", n=2)
        shutil.rmtree(src)
        with pytest.raises(FileNotFoundError, match="import_project"):
            m.fds.solve(config=CFG)

    def test_plan_reduces_a_source_without_a_reduced_model(self, tmp_path, caplog):
        src = EMProject("fom_only", base_dir=str(tmp_path), overwrite=True)
        src.geometry = _section()
        src.fds.solve(config=dict(CFG, store_snapshots=True))
        src.save()
        path = tmp_path / "fom_only"
        before = {f: f.stat().st_mtime for f in path.rglob("*") if f.is_file()}
        m = EMProject("mod_red", base_dir=str(tmp_path), overwrite=True)
        m.import_project(path, name="sec", n=2)
        m.fds.solve(config=CFG)
        assert "reduce" in caplog.text
        concat = m.fds.foms.reduce(tol=1e-9).concatenate()
        assert len(concat.structures) == 2
        assert (tmp_path / "mod_red" / "fds" / "foms" / "roms" / "matrices"
                / "A_r_sec.h5").exists()
        after = {f: f.stat().st_mtime for f in path.rglob("*") if f.is_file()}
        assert after == before                               # source untouched

    def test_plan_recomputes_a_part_that_does_not_fit(self, tmp_path):
        src = _solved_source(tmp_path)                       # trained on 1.8-2.4 GHz
        before = {f: f.stat().st_mtime for f in src.rglob("*") if f.is_file()}
        m = EMProject("mod_wide", base_dir=str(tmp_path), overwrite=True)
        m.import_project(src, name="sec", n=2)
        wide = dict(CFG, fmax=2.6)
        with pytest.raises(RuntimeError, match="rerun=True"):   # non-interactive
            m.fds.solve(config=wide)
        m.fds.solve(config=wide, rerun=True)
        concat = m.fds.foms.reduce(tol=1e-9).concatenate()
        band = concat.structures[0].training_band
        assert band["fmax_GHz"] == pytest.approx(2.6)
        after = {f: f.stat().st_mtime for f in src.rglob("*") if f.is_file()}
        assert after == before                               # source untouched

    def test_plan_recomputes_when_port_modes_are_short(self, tmp_path):
        src = _solved_source(tmp_path)                       # 1 mode per port
        m = EMProject("mod_modes", base_dir=str(tmp_path), overwrite=True)
        m.import_project(src, name="sec", n=2)
        m.fds.solve(config=dict(CFG, nportmodes=2), rerun=True)
        concat = m.fds.foms.reduce(tol=1e-9).concatenate()
        assert len(concat.structures[0].port_modes["port1"]) == 2

    def test_joins_use_the_facing_ports(self, tmp_path):
        src = _solved_source(tmp_path)
        _, concat, _ = _module(tmp_path, src, "mod_join", "reference", n=2)
        # a waveguide section has port1 at z=0 (facing -z) and port2 at z=L
        assert concat.connections == [((0, "port2"), (1, "port1"))]
