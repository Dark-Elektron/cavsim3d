"""
Resumable frequency sweeps (cavsim3d.solvers.sweep_checkpoint).

Validates:
  - a sample round-trips through its checkpoint file; other sweeps are ignored
  - an interrupted solve() resumes: finished samples are read, not recomputed,
    and the result equals an uninterrupted solve
  - resuming also works after reopening the project
  - the checkpoint is removed once the full results are saved
"""

import logging

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.solvers import sweep_checkpoint as sc
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver

CFG = dict(fmin=1.6, fmax=3.0, nsamples=6, order=1, nportmodes=1, solver_type="direct")


class TestCheckpointFiles:
    def test_round_trip_and_other_sweeps(self, tmp_path):
        rhs = np.arange(12.0).reshape(6, 2)
        ck = sc.SweepCheckpoint(tmp_path, "global", "fp", sc.rhs_signature(rhs))
        x = np.ones((6, 2), dtype=complex)
        ck.write(3, np.eye(2), x, [4, 5], [1e-9, 2e-9], 1.5)
        got = ck.load()
        assert list(got) == [3]
        np.testing.assert_array_equal(got[3]["x"], x)
        assert got[3]["iters"].tolist() == [4, 5] and got[3]["time"] == 1.5

        # re-assembly noise in the RHS is tolerated, a real change is not
        noisy = sc.SweepCheckpoint(tmp_path, "global", "fp", sc.rhs_signature(rhs * (1 + 1e-13)))
        assert list(noisy.load()) == [3]
        flipped = rhs.copy()
        flipped[:, 1] *= -1                               # e.g. a port mode's sign
        other = sc.SweepCheckpoint(tmp_path, "global", "fp", sc.rhs_signature(flipped))
        assert other.load() == {}
        assert not list((tmp_path / "global").glob("sample_*.npz"))   # stale sample removed

    def test_partial_write_is_ignored(self, tmp_path):
        ck = sc.SweepCheckpoint(tmp_path, "global", "fp", np.ones(3))
        (tmp_path / "global").mkdir()
        (tmp_path / "global" / "sample_00000.tmp.npz").write_bytes(b"truncated")
        assert ck.load() == {}

    def test_disabled_without_project(self, tmp_path):
        ck = sc.SweepCheckpoint(None, "global")
        ck.write(0, np.eye(1), None, [0], [0.0], 0.1)        # no-op
        assert ck.load() == {}


class _Interrupt(Exception):
    pass


def _interrupt_after(monkeypatch, n):
    """Make the sweep stop right after its n-th sample is checkpointed."""
    real = sc.SweepCheckpoint.write
    count = [0]

    def write(self, *a, **k):
        real(self, *a, **k)
        count[0] += 1
        if count[0] == n:
            raise _Interrupt
    monkeypatch.setattr(sc.SweepCheckpoint, "write", write)


def _count_computed(monkeypatch):
    """Count the samples the sweep really computes (checkpoint writes)."""
    real = sc.SweepCheckpoint.write
    count = [0]

    def write(self, *a, **k):
        count[0] += 1
        real(self, *a, **k)
    monkeypatch.setattr(sc.SweepCheckpoint, "write", write)
    return count


def _project(tmp_path, name):
    p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
    p.create_primitive("rwg", name="guide", a=0.1, L=0.2, b=0.05, maxh=0.04)
    return p


class TestResume:
    def test_interrupted_solve_resumes(self, tmp_path, monkeypatch, caplog):
        ref = _project(tmp_path, "ref")
        ref.fds.solve(config=CFG)
        Z_ref = ref.fds._Z_matrix.copy()
        snaps_ref = ref.fds.snapshots["global"].copy()
        assert not (ref.project_path / "fds" / "checkpoint").exists()   # cleared

        p = _project(tmp_path, "cut")
        _interrupt_after(monkeypatch, 3)
        with pytest.raises(_Interrupt):
            p.fds.solve(config=CFG)
        assert len(list((p.project_path / "fds" / "checkpoint" / "global").glob("sample_*.npz"))) == 3

        monkeypatch.undo()
        computed = _count_computed(monkeypatch)
        with caplog.at_level(logging.INFO, logger="cavsim3d"):
            p.fds.solve(config=CFG)
        assert computed[0] == 3                                   # only the missing samples
        assert any("Resuming an interrupted sweep" in r.getMessage() for r in caplog.records)
        np.testing.assert_allclose(p.fds._Z_matrix, Z_ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(p.fds.snapshots["global"], snaps_ref, rtol=1e-10, atol=1e-14)
        assert not (p.project_path / "fds" / "checkpoint").exists()

    def test_resume_after_reopening(self, tmp_path, monkeypatch):
        p = _project(tmp_path, "reopen")
        _interrupt_after(monkeypatch, 2)
        with pytest.raises(_Interrupt):
            p.fds.solve(config=CFG)
        monkeypatch.undo()

        q = EMProject(name="reopen", base_dir=str(tmp_path))         # e.g. after a restart
        computed = _count_computed(monkeypatch)
        q.fds.solve(config=CFG)
        assert computed[0] == CFG["nsamples"] - 2
        assert q.fds._Z_matrix.shape[0] == CFG["nsamples"]

    def test_rerun_true_starts_over(self, tmp_path, monkeypatch):
        p = _project(tmp_path, "forced")
        _interrupt_after(monkeypatch, 2)
        with pytest.raises(_Interrupt):
            p.fds.solve(config=CFG)
        monkeypatch.undo()
        computed = _count_computed(monkeypatch)
        p.fds.solve(config=CFG, rerun=True)
        assert computed[0] == CFG["nsamples"]

    def test_changed_sweep_starts_over(self, tmp_path, monkeypatch):
        p = _project(tmp_path, "changed")
        _interrupt_after(monkeypatch, 2)
        with pytest.raises(_Interrupt):
            p.fds.solve(config=CFG)
        monkeypatch.undo()
        computed = _count_computed(monkeypatch)
        p.fds.solve(config=dict(CFG, fmax=3.2))                   # other frequencies
        assert computed[0] == CFG["nsamples"]


NET_CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=1, solver_type="direct")


def _chain(tmp_path, name):
    """Two unique parts, the first repeated: a netlist (each part solved on its own)."""
    p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
    p.create_primitive("rwg", name="a", n=2, a=0.1, L=0.1, b=0.05, maxh=0.05)
    p.create_primitive("rwg", name="b", a=0.1, L=0.15, b=0.05, maxh=0.05)
    return p


def _chain_S(p):
    concat = p.fds.foms.reduce(tol=1e-9).concatenate()
    return concat.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=7))["S"]


class TestNetlistResume:
    def test_interrupted_netlist_resumes(self, tmp_path, monkeypatch):
        ref = _chain(tmp_path, "net_ref")
        ref.fds.solve(config=NET_CFG)
        S_ref = _chain_S(ref)

        p = _chain(tmp_path, "net_cut")
        _interrupt_after(monkeypatch, NET_CFG["nsamples"] + 2)     # part 'b' after 2 samples
        with pytest.raises(_Interrupt):
            p.fds.solve(config=NET_CFG)
        sections = p.project_path / "fds" / "checkpoint" / "sections"
        assert len(list((sections / "a" / "global").glob("sample_*.npz"))) == NET_CFG["nsamples"]
        assert len(list((sections / "b" / "global").glob("sample_*.npz"))) == 2

        monkeypatch.undo()
        computed = _count_computed(monkeypatch)
        p.fds.solve(config=NET_CFG)
        assert computed[0] == NET_CFG["nsamples"] - 2        # only b's missing samples
        assert not (p.project_path / "fds" / "checkpoint").exists()
        np.testing.assert_allclose(_chain_S(p), S_ref, rtol=1e-8, atol=1e-10)


class TestBatchLine:
    def test_restored_samples_are_not_timed(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="cavsim3d"):
            FrequencyDomainSolver._report_batch(0, 4, 0.0, [0] * 5, [0.0] * 5, 1,
                                                "direct", {}, restored=5)
            FrequencyDomainSolver._report_batch(5, 9, 0.0, [0] * 10, [0.0] * 10, 1,
                                                "direct", {}, restored=2)
        a, b = [r.getMessage() for r in caplog.records[-2:]]
        assert a == "  \tsamples 1-5: read from the checkpoint"
        assert b.startswith("  \tsamples 6-10: ") and "2 read from the checkpoint" in b
