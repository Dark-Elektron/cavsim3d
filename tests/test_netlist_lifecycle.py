"""A chain of coupled parts across solves and sessions.

    proj.add("sec", part, n=2)          # repeated -> coupled through port modes
    proj.fds.solve(config)              # each unique section once, staged here
    proj.fds.foms.reduce(tol)           # any number of times, from staged files
    proj.fds.foms.roms.concatenate()    # the joined model

Reopening the project restores every stage; a second solve with the same
settings reuses the staged sections; the scratch projects never outlive a solve.
"""
import json
import tempfile

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver

CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)
SWEEP = dict(fmin=1.8, fmax=2.4, nsamples=5)


@pytest.fixture
def scratch_dir(tmp_path, monkeypatch):
    """The system temp folder, where the sections' scratch projects go."""
    d = tmp_path / "_tmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def _chain(root, name="chain"):
    p = EMProject(name, base_dir=root, overwrite=True)
    p.add("sec", RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06), n=2)
    return p


@pytest.fixture
def count_section_solves(monkeypatch):
    calls = []
    original = FrequencyDomainSolver._run_section_fom

    def counting(base, comp, cfg, project_root):
        calls.append(base)
        return original(base, comp, cfg, project_root)

    monkeypatch.setattr(FrequencyDomainSolver, "_run_section_fom", staticmethod(counting))
    return calls


def test_solving_leaves_no_scratch_folders(tmp_path, scratch_dir):
    p = _chain(tmp_path)
    p.fds.solve(config=CFG)
    assert not list(scratch_dir.glob("cavsim3d_section_*"))
    record = json.loads((tmp_path / "chain" / "fds" / "sections.json").read_text())
    assert set(record["sections"]) == {"sec"}
    assert record["sections"]["sec"]["rom_template"]["ports"] == ["port1", "port2"]


def test_a_failed_section_solve_removes_its_scratch_folder(tmp_path, scratch_dir, monkeypatch):
    p = _chain(tmp_path)
    original = FrequencyDomainSolver.solve

    def failing(self, *args, **kwargs):
        if self._checkpoint_dir is not None:          # the section's own solve
            raise RuntimeError("section solve failed")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(FrequencyDomainSolver, "solve", failing)
    with pytest.raises(RuntimeError, match="section solve failed"):
        p.fds.solve(config=CFG)
    assert not list(scratch_dir.glob("cavsim3d_section_*"))


def test_same_settings_reuse_the_staged_sections(tmp_path, scratch_dir, count_section_solves):
    p = _chain(tmp_path)
    p.fds.solve(config=CFG)
    assert count_section_solves == ["sec"]
    p.fds.solve(config=CFG)                              # same session
    EMProject("chain", base_dir=tmp_path).fds.solve(config=CFG)   # reopened
    assert count_section_solves == ["sec"]
    EMProject("chain", base_dir=tmp_path).fds.solve(config=dict(CFG, nsamples=5))
    assert count_section_solves == ["sec", "sec"]        # another sweep: solved
    EMProject("chain", base_dir=tmp_path).fds.solve(config=dict(CFG, nsamples=5),
                                                    rerun=True)
    assert count_section_solves == ["sec", "sec", "sec"]


def test_a_changed_geometry_is_solved_again(tmp_path, scratch_dir, count_section_solves):
    p = _chain(tmp_path)
    p.fds.solve(config=CFG)
    sections = tmp_path / "chain" / "fds" / "sections.json"
    data = json.loads(sections.read_text())
    data["sections"]["sec"]["signature"] = "a different geometry"
    sections.write_text(json.dumps(data))
    EMProject("chain", base_dir=tmp_path).fds.solve(config=CFG)
    assert count_section_solves == ["sec", "sec"]


def test_reduce_again_and_reopen_every_stage(tmp_path, scratch_dir, monkeypatch):
    p = _chain(tmp_path)
    p.fds.solve(config=CFG)
    flat = tmp_path / "chain" / "fds" / "foms" / "roms" / "structures.json"

    p.fds.foms.reduce(tol=1e-2)
    r_coarse = json.loads(flat.read_text())["structures"][0]["r"]
    roms = p.fds.foms.reduce(tol=1e-10)                  # used to raise KeyError
    entry = json.loads(flat.read_text())["structures"][0]
    assert entry["tol"] == 1e-10 and entry["r"] >= r_coarse
    S = roms.concatenate().solve(**SWEEP)["S"]

    # the same reduction again is not recomputed
    from cavsim3d.solvers import netlist_persistence as npz
    calls = []
    monkeypatch.setattr(npz, "reduce_staged_section",
                        lambda *a, **k: calls.append(a) or pytest.fail("re-reduced"))
    p.fds.foms.reduce(tol=1e-10)
    assert calls == []
    monkeypatch.undo()

    q = EMProject("chain", base_dir=tmp_path)
    assert q.fds.foms.keys == ["sec"]
    assert q.fds.foms["sec"].fom.S_dict is not None      # the section's own FOM
    concat = q.fds.foms.roms.concat                      # restored with its sweep
    assert np.allclose(concat._S_matrix, S)
    assert np.allclose(concat.solve(**SWEEP)["S"], S)
    # reducing after reopening works from the staged full-order files
    S6 = q.fds.foms.reduce(tol=1e-6).concatenate().solve(**SWEEP)["S"]
    assert np.allclose(S6, S, atol=1e-3)


def test_reducing_again_drops_the_stale_joined_model(tmp_path, scratch_dir):
    p = _chain(tmp_path)
    p.fds.solve(config=CFG)
    p.fds.foms.reduce(tol=1e-6).concatenate().solve(**SWEEP)
    p.fds.foms.reduce(tol=1e-2)
    q = EMProject("chain", base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="concatenate"):
        q.fds.foms.roms.concat
