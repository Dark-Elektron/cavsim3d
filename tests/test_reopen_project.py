"""A reopened single-solid project behaves like the one that was solved.

    proj = EMProject(name, base_dir)          # reopen, no overwrite
    proj.fds.solve(<same request>)            # returns the stored S and Z
    proj.fds.fom.reduce(tol)                  # writes a complete ROM
"""
import json
import warnings

import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide

CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)


def _solved(tmp_path, name="reopen"):
    proj = EMProject(name, base_dir=str(tmp_path), overwrite=True)
    proj.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    res = proj.fds.solve(config=CFG)
    proj.fds.fom.reduce(tol=1e-9)
    return res


def test_reopened_solve_returns_stored_results(tmp_path):
    res = _solved(tmp_path)
    proj = EMProject("reopen", base_dir=str(tmp_path))
    again = proj.fds.solve(config=CFG)
    assert again["S"] is not None and again["Z"] is not None
    np.testing.assert_allclose(again["S"], res["S"])
    np.testing.assert_allclose(again["Z"], res["Z"])


def test_reopened_rom_keeps_port_modes(tmp_path):
    _solved(tmp_path)
    proj = EMProject("reopen", base_dir=str(tmp_path))
    rom = proj.fds.fom.rom
    assert rom.port_modes is not None
    assert rom.get_reduced_structure().n_full > rom.get_reduced_structure().r


def test_forced_remesh_discards_old_results(tmp_path):
    proj = EMProject("remesh", base_dir=str(tmp_path), overwrite=True)
    proj.create_primitive("rwg", name="g", a=0.1, b=0.05, L=0.1, maxh=0.06)
    coarse = proj.fds.solve(config=CFG)["S"]
    proj.generate_mesh(maxh=0.03, force=True)
    fine = proj.fds.solve(config=CFG)["S"]
    assert not np.allclose(fine, coarse, rtol=0, atol=1e-12)


def test_reopened_chain_has_its_mesh(tmp_path):
    proj = EMProject("glued", base_dir=str(tmp_path), overwrite=True)
    for name in ("a", "b"):
        proj.create_primitive("rwg", name=name, a=0.1, b=0.05, L=0.05, maxh=0.03)
    proj.generate_mesh(maxh=0.03)
    again = EMProject("glued", base_dir=str(tmp_path))
    assert again.geometry.mesh is not None
    assert again.geometry.mesh.ne == proj.mesh.ne


def test_chain_edits_persist_through_generate_mesh(tmp_path):
    proj = EMProject("chain", base_dir=str(tmp_path), overwrite=True)
    for name in ("a", "b", "c"):
        proj.create_primitive("rwg", name=name, a=0.1, b=0.05, L=0.05, maxh=0.03)
    proj.geometry.remove("c")
    proj.geometry.set_mesh_strategy("coupled")   # no project-level mesh
    proj.generate_mesh(maxh=0.03)
    again = EMProject("chain", base_dir=str(tmp_path))
    assert list(again.parts) == ["a", "b"]
    assert again.geometry.resolved_mesh_strategy()[0] == "coupled"


def test_reopened_reduce_writes_complete_rom(tmp_path):
    _solved(tmp_path)
    proj = EMProject("reopen", base_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        proj.fds.fom.reduce(tol=1e-9)
    meta = json.loads((proj.project_path / "fds" / "fom" / "rom"
                       / "structures.json").read_text())
    assert [s["domain"] for s in meta["structures"]] == ["global"]
