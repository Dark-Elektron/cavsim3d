"""
Generalised matrices with the beam and their conversions
(docs/theory/beam.md §9.7-9.9, docs/theory/beam_conversions.md).

A random complex-symmetric system stands in for the finite-element system, with
the port matrix B, the beam load f, the beam-voltage functional c and the part
r of the beam's own field that the port modes see.  Solved with the ports open
(currents prescribed), matched (incoming waves prescribed, a Robin condition)
and short-circuited (voltages prescribed, the currents as multipliers), it
gives Z~, S~_CSC and the short-circuit hybrid directly; the conversions of
cavsim3d.solvers.beam must reproduce them, and S~ and Y~ must satisfy their
definitions.
"""

import numpy as np
import pytest

import cavsim3d  # noqa: F401  (loads pythonocc before netgen)
from cavsim3d.solvers import beam as bm
from cavsim3d.solvers import netlist_persistence as npz

N, P, S = 30, 3, 2          # unknowns, port modes, beams
W = 1.3                     # angular frequency (any scale)
RB = 1.7                    # beam reference impedance


def _system(seed=7, L=0):
    """Toy system and its matrices at two frequencies; ``L`` extra voltage
    paths without current."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(2):
        G = rng.normal(size=(N, N)) + 1j * 0.05 * rng.normal(size=(N, N))
        A = G + G.T + 3 * np.eye(N)
        B = rng.normal(size=(N, P))
        f = rng.normal(size=(N, S)) + 1j * rng.normal(size=(N, S))
        c = rng.normal(size=(N, S + L)) + 1j * rng.normal(size=(N, S + L))
        r = rng.normal(size=(P, S)) + 1j * rng.normal(size=(P, S))
        zr = np.array([1.1, 0.7, 0.4j])                 # the last mode evanescent
        out.append(dict(A=A, B=B, f=f, c=c, r=r, zr=zr))
    return out


def _open(s):
    """Z~ from the open-port solve: [V; v] per [I; i]."""
    Ainv = np.linalg.inv(s['A'])
    Z = 1j * W * s['B'].T @ Ainv @ s['B']
    hZ = 1j * W * s['c'].T @ Ainv @ s['B']
    es = Ainv @ s['f']
    return bm.z_tilde(Z[None], (s['B'].T @ es - s['r'])[None], hZ[None], (s['c'].T @ es)[None])[0]


def _matched(s):
    """S~_CSC from the matched-port solve: [b; v] per [a; i]."""
    zr, B = s['zr'], s['B']
    izr, isq = np.diag(1 / zr), np.diag(1 / np.sqrt(zr))
    Am = s['A'] + 1j * W * B @ izr @ B.T
    cols = []
    for col in range(P + S):
        a, i = np.zeros(P, complex), np.zeros(S, complex)
        (a if col < P else i)[col % P if col < P else col - P] = 1
        x = np.linalg.solve(Am, 2j * W * B @ isq @ a + (s['f'] + 1j * W * B @ izr @ s['r']) @ i)
        V = B.T @ x - s['r'] @ i
        I = 2 * isq @ a - izr @ V
        cols.append(np.concatenate([0.5 * isq @ (V - np.diag(zr) @ I), s['c'].T @ x]))
    return np.array(cols).T


def _shorted(s):
    """The hybrid of the short-circuit solve: [I; v] per [V; i]."""
    K = np.block([[s['A'], -1j * W * s['B']], [s['B'].T, np.zeros((P, P))]])
    cols = []
    for col in range(P + S):
        V, i = np.zeros(P, complex), np.zeros(S, complex)
        (V if col < P else i)[col % P if col < P else col - P] = 1
        sol = np.linalg.solve(K, np.concatenate([s['f'] @ i, V + s['r'] @ i]))
        cols.append(np.concatenate([sol[N:], s['c'][:, :S].T @ sol[:N]]))
    return np.array(cols).T


def _stack(systems, fn):
    return np.array([fn(s) for s in systems])


def _zref(systems):
    return np.array([np.diag(s['zr']) for s in systems])


def test_csc_from_z_tilde_is_the_matched_port_solve():
    sys_ = _system()
    Zt, Zref = _stack(sys_, _open), _zref(sys_)
    Sc = bm.s_csc_from_z_tilde(Zt, Zref)
    np.testing.assert_allclose(Sc, _stack(sys_, _matched), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(bm.z_tilde_from_s_csc(Sc, Zref), Zt, rtol=1e-11, atol=1e-12)


def test_y_tilde_is_the_short_circuit_solve():
    sys_ = _system()
    Zt = _stack(sys_, _open)
    Yt = bm.y_tilde_from_z_tilde(Zt, P)
    for H, Y in zip(_stack(sys_, _shorted), Yt):
        # the hybrid [I; v] = H [V; i] with the beam row solved for i, u = -v
        Ysc, kY, hY, zsc = H[:P, :P], H[:P, P:], H[P:, :P], H[P:, P:]
        izsc = np.linalg.inv(zsc)
        expected = np.block([[Ysc - kY @ izsc @ hY, -kY @ izsc], [-izsc @ hY, -izsc]])
        np.testing.assert_allclose(Y, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(bm.z_tilde_from_y_tilde(Yt, P), Zt, rtol=1e-11, atol=1e-12)


def test_s_tilde_from_z_tilde_and_from_csc_agree_and_invert():
    sys_ = _system()
    Zt, Zref = _stack(sys_, _open), _zref(sys_)
    St = bm.s_tilde_from_z_tilde(Zt, Zref, RB)
    np.testing.assert_allclose(bm.s_tilde_from_s_csc(bm.s_csc_from_z_tilde(Zt, Zref), P, RB),
                               St, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(bm.z_tilde_from_s_tilde(St, Zref, RB), Zt, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(bm.s_csc_from_s_tilde(St, P, RB), bm.s_csc_from_z_tilde(Zt, Zref),
                               rtol=1e-11, atol=1e-12)


def test_definitions_of_s_tilde_and_y_tilde():
    """Any port and beam currents: [b; b_b] = S~ [a; a_b], [I; i] = Y~ [V; u],
    [b; v] = S~_CSC [a; i], with u = -v and the waves of §9.8."""
    sys_ = _system()
    Zt, Zref = _stack(sys_, _open), _zref(sys_)
    St, Yt, Sc = (bm.s_tilde_from_z_tilde(Zt, Zref, RB), bm.y_tilde_from_z_tilde(Zt, P),
                  bm.s_csc_from_z_tilde(Zt, Zref))
    rng = np.random.default_rng(3)
    for k, s in enumerate(sys_):
        Ii = rng.normal(size=P + S) + 1j * rng.normal(size=P + S)
        Vv = Zt[k] @ Ii
        V, v, I, i = Vv[:P], Vv[P:], Ii[:P], Ii[P:]
        u = -v
        sq = np.sqrt(s['zr'])
        a, b = (V + s['zr'] * I) / (2 * sq), (V - s['zr'] * I) / (2 * sq)
        ab, bb = (u + RB * i) / (2 * np.sqrt(RB)), (u - RB * i) / (2 * np.sqrt(RB))
        np.testing.assert_allclose(St[k] @ np.r_[a, ab], np.r_[b, bb], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(Yt[k] @ np.r_[V, u], np.r_[I, i], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(Sc[k] @ np.r_[a, i], np.r_[b, v], rtol=1e-11, atol=1e-12)
        # real reference impedances: the waves carry the power of V, I and u, i
        if np.isreal(s['zr']).all():
            assert np.isclose(0.5 * (abs(ab) ** 2 - abs(bb) ** 2), 0.5 * (u * np.conj(i)).real)


def test_beam_entry_measures_the_beam_impedance():
    """One beam, no ports: s_bb = (Z_par - R_b) / (Z_par + R_b), Z_par = -z_b."""
    zb = np.array([[[0.0]], [[-3.0 + 2.0j]], [[-RB]]])           # z_b = v / i
    St = bm.s_tilde_from_s_csc(zb, 0, RB)
    zpar = -zb[:, 0, 0]
    np.testing.assert_allclose(St[:, 0, 0], (zpar - RB) / (zpar + RB), atol=1e-15)
    assert St[0, 0, 0] == pytest.approx(-1)          # no beam impedance: a short
    assert abs(St[1, 0, 0]) < 1                      # Re Z_par > 0
    assert St[2, 0, 0] == pytest.approx(0)           # Z_par = R_b: matched


def test_paths_without_current_stay_in_z_tilde_and_csc():
    """A voltage path without current adds a row to Z~ and S~_CSC; S~ and Y~
    are square over the port modes and the beams."""
    sys_ = _system(L=1)
    Zt, Zref = _stack(sys_, _open), _zref(sys_)
    assert Zt.shape == (2, P + S + 1, P + S)
    Sc = bm.s_csc_from_z_tilde(Zt, Zref)
    assert Sc.shape == Zt.shape
    sq = Zt[:, :P + S, :]
    np.testing.assert_allclose(bm.y_tilde_from_z_tilde(Zt, P), bm.y_tilde_from_z_tilde(sq, P))
    np.testing.assert_allclose(bm.s_tilde_from_z_tilde(Zt, Zref), bm.s_tilde_from_z_tilde(sq, Zref))
    np.testing.assert_allclose(bm.s_tilde_from_s_csc(Sc, P), bm.s_tilde_from_z_tilde(sq, Zref),
                               rtol=1e-11, atol=1e-12)


def test_y_tilde_of_a_beam_without_impedance_is_nan():
    Zt = np.zeros((2, 1, 1), dtype=complex)
    Zt[1] = 2.0
    Yt = bm.y_tilde_from_z_tilde(Zt, 0)
    assert np.isnan(Yt[0]).all() and Yt[1, 0, 0] == pytest.approx(-0.5)


def test_legacy_s_tilde_folder_is_read_and_staged_as_csc(tmp_path):
    """Older versions saved S~_CSC in s_tilde/: it is found, and staging a part
    copies it into s_csc/ under the section's name."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "s_tilde").mkdir(parents=True)
    (src / "s_tilde" / "s_tilde_global.h5").write_bytes(b"x")
    assert bm.find_csc_file(src, "global") == src / "s_tilde" / "s_tilde_global.h5"
    npz._copy_legacy_beam_dirs(src, dst, "global", "cell")
    assert (dst / "s_csc" / "s_csc_cell.h5").read_bytes() == b"x"
    bm.remove_legacy_csc(src, "global")
    assert not (src / "s_tilde").exists() and bm.find_csc_file(src, "global") is None
