"""A reduction that fails says so, and leaves the stored reduced model as it was.

    pod_reduce(...)                 # raises FloatingPointError on a failed SVD
    proj.fds.fom.reduce(tol)        # ... and the saved ROM is not overwritten
    roms.concatenate()              # a 0-DOF section is named, not "Null space empty"
"""
import tempfile

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.rom.reduction import _real_pod_basis, pod_reduce

CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)


def _system(n=80, p=2, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    K = A @ A.T + n * np.eye(n)
    M = np.diag(rng.uniform(1.0, 2.0, n))
    B = rng.standard_normal((n, p))
    return K, M, B


def _nan_svd(original):
    def svd(a, *args, **kwargs):
        U, S, Vh = original(a, *args, **kwargs)
        return U, np.full_like(S, np.nan), Vh      # MKL gesdd without workspace
    return svd


def test_pod_basis_spans_the_svd_basis():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 12)) + 1j * rng.standard_normal((300, 12))
    X0 = X.copy()
    Q, U_R, S = _real_pod_basis(X)
    Xr = np.hstack([X.real, X.imag])
    U, S_ref, _ = np.linalg.svd(Xr, full_matrices=False)
    np.testing.assert_allclose(S, S_ref, rtol=1e-12)
    W = Q @ U_R[:, :10]
    np.testing.assert_allclose(W @ W.T, U[:, :10] @ U[:, :10].T, atol=1e-12)
    assert np.array_equal(X, X0)                   # the input is left alone


def test_nan_snapshots_raise():
    K, M, B = _system()
    X = np.ones((80, 6))
    X[3, 2] = np.nan
    with pytest.raises(FloatingPointError, match="NaN or infinite"):
        pod_reduce(K, M, B, X)


def test_all_zero_snapshots_raise():
    K, M, B = _system()
    with pytest.raises(FloatingPointError, match="all-zero"):
        pod_reduce(K, M, B, np.zeros((80, 6)))


def test_failed_svd_raises_instead_of_a_0_dof_model(monkeypatch):
    K, M, B = _system()
    X = np.random.default_rng(2).standard_normal((80, 6))
    monkeypatch.setattr(np.linalg, "svd", _nan_svd(np.linalg.svd))
    with pytest.raises(FloatingPointError, match="NaN singular values"):
        pod_reduce(K, M, B, X)


def test_failed_reduce_keeps_the_stored_model(tmp_path, monkeypatch):
    proj = EMProject("rom", base_dir=str(tmp_path), overwrite=True)
    proj.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    proj.fds.solve(config=CFG)
    good = proj.fds.fom.reduce(tol=1e-9)
    r, A_r = good._r_global, good._A_r_global.copy()
    rom_dir = tmp_path / "rom" / "fds" / "fom" / "rom"
    stamps = {f: f.stat().st_mtime_ns for f in rom_dir.rglob("*.h5")}

    monkeypatch.setattr(np.linalg, "svd", _nan_svd(np.linalg.svd))
    with pytest.raises(FloatingPointError, match="left as it was"):
        proj.fds.fom.reduce(tol=1e-6)
    monkeypatch.undo()

    assert proj.fds.fom.rom is good
    assert {f: f.stat().st_mtime_ns for f in rom_dir.rglob("*.h5")} == stamps
    again = EMProject("rom", base_dir=str(tmp_path)).fds.fom.rom
    assert again._r_global == r
    np.testing.assert_allclose(again._A_r_global, A_r)


def test_a_0_dof_section_is_named_at_the_join(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    p = EMProject("chain", base_dir=str(tmp_path), overwrite=True)
    p.add("sec", RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06), n=2)
    p.fds.solve(config=CFG)
    roms = p.fds.foms.reduce(tol=1e-6)
    # a reduced model as the old failure saved it: no basis vectors at all
    from cavsim3d.solvers.concatenation import ConcatenatedSystem
    original = ConcatenatedSystem.couple

    def couple_with_an_empty_section(self, *args, **kwargs):
        s = self.structures[0]
        s.Ard, s.Brd = np.zeros((0, 0)), np.zeros((0, np.shape(s.Brd)[1]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConcatenatedSystem, "couple", couple_with_an_empty_section)
    with pytest.raises(RuntimeError, match=r"'sec.*' has 0 DOFs"):
        roms.concatenate()
