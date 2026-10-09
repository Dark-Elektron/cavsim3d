"""The beam's blocks of reduced sections and their joins (cavsim3d.solvers.beam).

Validates:
  - S~ and Z~ convert into each other (stacked over the frequencies)
  - the join of segments is the same in one batch of frequencies or in many
  - the joins and the reduced beam blocks run on one BLAS thread: many small
    dense solves, which a multithreaded BLAS slows down by 10-100x
"""
import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.solvers import beam as bm


def _segments(rng, D=4, Pd=6, L=2, S=2, n_f=30):
    blocks = [0.3 * (rng.standard_normal((n_f, Pd + L, Pd + S))
                     + 1j * rng.standard_normal((n_f, Pd + L, Pd + S))) for _ in range(D)]
    modes = [list(range(d * Pd, (d + 1) * Pd)) for d in range(D)]
    h = Pd // 2                     # the upper half of segment d joins the lower of d + 1
    pairs = [(d * Pd + h + m, (d + 1) * Pd + m) for d in range(D - 1) for m in range(h)]
    external = list(range(h)) + list(range((D - 1) * Pd + h, D * Pd))
    col = [np.exp(-1j * rng.uniform(0, 3, (n_f, S))) for _ in range(D)]
    row = [np.exp(1j * rng.uniform(0, 3, (n_f, L))) for _ in range(D)]
    return blocks, modes, pairs, external, col, row


def test_s_tilde_and_z_tilde_convert_into_each_other():
    rng = np.random.default_rng(0)
    n_f, P, L, S = 12, 5, 2, 3
    Z = rng.standard_normal((n_f, P, P)) + 1j * rng.standard_normal((n_f, P, P))
    Z = Z + Z.transpose(0, 2, 1)
    kZ = rng.standard_normal((n_f, P, S)) + 1j * rng.standard_normal((n_f, P, S))
    hZ = rng.standard_normal((n_f, L, P)) + 1j * rng.standard_normal((n_f, L, P))
    zoc = rng.standard_normal((n_f, L, S)) + 1j * rng.standard_normal((n_f, L, S))
    Zref = np.array([np.diag(rng.uniform(50.0, 400.0, P)) for _ in range(n_f)]).astype(complex)
    St = bm.s_tilde(Z, kZ, hZ, zoc, Zref)
    np.testing.assert_allclose(bm.z_tilde_from_s_tilde(St, Zref), bm.z_tilde(Z, kZ, hZ, zoc),
                               rtol=0, atol=1e-11 * np.abs(Z).max())
    # one frequency at a time: the same matrices
    for i in (0, n_f - 1):
        one = bm.s_tilde(Z[i:i + 1], kZ[i:i + 1], hZ[i:i + 1], zoc[i:i + 1], Zref[i:i + 1])
        np.testing.assert_allclose(one[0], St[i], rtol=1e-13)


def test_join_in_one_batch_or_many(monkeypatch):
    args = _segments(np.random.default_rng(1))
    whole = bm.join_s_tilde(*args)
    monkeypatch.setattr(bm, "JOIN_CHUNK_BYTES", 1.0)            # one frequency per batch
    np.testing.assert_allclose(bm.join_s_tilde(*args), whole, rtol=1e-13, atol=1e-15)
    E, D, L, S = len(args[3]), len(args[0]), 2, 2
    assert whole.shape == (30, E + L, E + S)


def test_join_runs_on_one_blas_thread(monkeypatch):
    threadpoolctl = pytest.importorskip("threadpoolctl")
    if not any(i["user_api"] == "blas" for i in threadpoolctl.threadpool_info()):
        pytest.skip("threadpoolctl cannot see the BLAS library")
    seen = []
    solve = np.linalg.solve

    def recording_solve(*a, **k):
        seen.append({i["num_threads"] for i in threadpoolctl.threadpool_info()
                     if i["user_api"] == "blas"})
        return solve(*a, **k)

    monkeypatch.setattr(np.linalg, "solve", recording_solve)
    bm.join_s_tilde(*_segments(np.random.default_rng(2)))
    assert seen and all(s == {1} for s in seen)
