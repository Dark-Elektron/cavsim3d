"""Sweeps of reduced and joined models: any frequencies, and no stored states.

    rom.solve(frequencies=f)               # any array [GHz], e.g. on narrow peaks
    concat.solve(frequencies=f)
    rom.solve(..., store_snapshots=False)  # the reduced solution is not kept
    concat.solve(..., store_snapshots=False)

A field at one frequency is then solved again when asked for; it equals the
stored one.
"""
import tempfile

import h5py
import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide

CFG = dict(fmin=1.8, fmax=2.4, nsamples=6, nportmodes=1, order=2)


@pytest.fixture(scope="module")
def rom(tmp_path_factory):
    root = tmp_path_factory.mktemp("rom")
    p = EMProject("rom", base_dir=str(root), overwrite=True)
    p.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    p.fds.solve(config=CFG)
    return p.fds.fom.reduce(tol=1e-9)


@pytest.fixture(scope="module")
def concat(tmp_path_factory):
    root = tmp_path_factory.mktemp("chain")
    old = tempfile.tempdir
    tempfile.tempdir = str(root)                 # the sections' scratch projects
    try:
        p = EMProject("chain", base_dir=str(root), overwrite=True)
        p.add("sec", RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06), n=2)
        p.fds.solve(config=CFG)
        c = p.fds.foms.reduce(tol=1e-9).concatenate()
    finally:
        tempfile.tempdir = old
    return c


def test_rom_takes_any_frequencies(rom):
    S_grid = rom.solve(1.8, 2.4, 7, rerun=True)["S"].copy()
    f = np.linspace(1.8, 2.4, 7)
    # unsorted, with a repeat: solved sorted, once each
    res = rom.solve(frequencies=[f[3], f[0], f[6], f[1], f[2], f[5], f[4], f[3]])
    np.testing.assert_allclose(rom.frequencies, f * 1e9)
    np.testing.assert_allclose(res["S"], S_grid, rtol=1e-12, atol=1e-14)
    # a peak's neighbourhood, much finer than any grid
    fine = 2.1 + np.array([-1e-6, 0.0, 1e-6])
    rom.solve(frequencies=fine)
    np.testing.assert_allclose(rom.frequencies, fine * 1e9)


def test_frequencies_and_a_grid_are_exclusive(rom):
    with pytest.raises(ValueError, match="either frequencies= or fmin"):
        rom.solve(1.8, 2.4, 5, frequencies=[2.0, 2.1])
    with pytest.raises(ValueError, match="> 0 GHz"):
        rom.solve(frequencies=[0.0, 2.0])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        rom.solve(frequencies=[])
    # a full-order config may be reused: frequencies= replaces its grid
    rom.solve(config=CFG, frequencies=[2.0, 2.2])
    np.testing.assert_allclose(rom.frequencies, [2.0e9, 2.2e9])


def test_rom_without_stored_states(rom):
    rom.solve(1.8, 2.4, 5, rerun=True)
    domain = rom.domains[0]
    stored = [rom._get_snapshot_for_excitation(k, "port1", 0, domain) for k in range(5)]
    S = rom._S_matrix.copy()
    rom.solve(1.8, 2.4, 5, rerun=True, store_snapshots=False)
    assert rom._x_r_snapshots is None and rom.can_reconstruct()
    np.testing.assert_allclose(rom._S_matrix, S, rtol=1e-12, atol=1e-14)
    for k in range(5):
        np.testing.assert_allclose(rom._get_snapshot_for_excitation(k, "port1", 0, domain),
                                   stored[k], rtol=1e-10, atol=1e-14 * np.abs(stored[k]).max())
    snap = rom._results_dir() / "snapshots" / f"snapshots_{domain}.h5"
    with h5py.File(snap) as f:
        assert "x_r_snapshots" not in f              # not the earlier sweep's either


def test_concat_takes_any_frequencies(concat):
    f = np.linspace(1.8, 2.4, 7)
    S_grid = concat.solve(1.8, 2.4, 7, rerun=True)["S"].copy()
    res = concat.solve(frequencies=f[::-1])
    np.testing.assert_allclose(concat.frequencies, f * 1e9)
    np.testing.assert_allclose(res["S"], S_grid, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="either frequencies= or fmin"):
        concat.solve(fmin=1.8, frequencies=f)


def test_concat_without_stored_states(concat):
    concat.solve(1.8, 2.4, 5, rerun=True)
    stored = [concat._state_at(k).copy() for k in range(5)]
    rom2 = concat.reduce(tol=1e-9)
    concat.solve(1.8, 2.4, 5, rerun=True, store_snapshots=False)
    assert concat._snapshots is None and concat.has_solution and not concat.has_snapshots
    for k in range(5):
        np.testing.assert_allclose(concat._state_at(k), stored[k], rtol=1e-10,
                                   atol=1e-12 * np.abs(stored[k]).max())
    folder, _ = concat._results_dir()
    with h5py.File(folder / "snapshots" / "snapshots.h5") as f:
        assert "frequencies" in f and "coupled_snapshots" not in f
    # a further reduction solves the states it needs again
    again = concat.reduce(tol=1e-9)
    np.testing.assert_allclose(np.linalg.eigvalsh(again.A_coupled),
                               np.linalg.eigvalsh(rom2.A_coupled), rtol=1e-8)
