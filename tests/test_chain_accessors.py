"""
Tests for backward-compatible result chain accessors and persistence round-trip.

Validates:
  - FOMResult.rom (deprecated) DeprecationWarning + returns MOR
  - FOMCollection.roms (deprecated) returns ROMCollection
  - FOMCollection.concat (deprecated) returns ConcatenatedSystem
  - ROMCollection.concat (deprecated) returns ConcatenatedSystem
  - ConcatenatedSystem.rom (deprecated) returns ReducedConcatenatedSystem
  - Save/load round-trip preserves all state
"""

import numpy as np
import pytest
import warnings
import unittest.mock as mock
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers.results import FOMResult, FOMCollection, ROMCollection


# ---------------------------------------------------------------------------
# Fixtures: minimal mock data
# ---------------------------------------------------------------------------

def _make_frequencies(n=20):
    return np.linspace(1.0, 3.0, n) * 1e9


def _make_z_dict(n=20):
    freqs = _make_frequencies(n)
    Z11 = np.random.randn(n) + 1j * np.random.randn(n)
    Z12 = np.random.randn(n) + 1j * np.random.randn(n)
    Z21 = Z12.copy()
    Z22 = np.random.randn(n) + 1j * np.random.randn(n)
    return {
        'frequencies': freqs,
        '1(1)1(1)': Z11,
        '1(1)2(1)': Z12,
        '2(1)1(1)': Z21,
        '2(1)2(1)': Z22,
    }


def _make_fom(domain='default', n=20, solver_ref=None):
    freqs = _make_frequencies(n)
    z = _make_z_dict(n)
    s = {k: v * 0.5 for k, v in z.items() if k != 'frequencies'}
    s['frequencies'] = freqs
    return FOMResult(
        domain=domain,
        frequencies=freqs,
        Z_matrix=None,
        S_matrix=None,
        Z_dict=z,
        S_dict=s,
        n_ports=2,
        ports=['port1', 'port2'],
        _solver_ref=solver_ref,
    )


# ---------------------------------------------------------------------------
# FOMResult.rom (backward-compatible)
# ---------------------------------------------------------------------------

class TestFOMResultRomAccessor:
    """FOMResult.rom should call reduce() and emit DeprecationWarning."""

    def test_rom_without_solver_raises(self):
        """Without solver, .rom should raise (because reduce() raises)."""
        fom = _make_fom()
        with pytest.raises(RuntimeError, match="no solver reference"):
            _ = fom.rom

    def test_rom_emits_deprecation_warning(self):
        """When solver exists, .rom should emit DeprecationWarning."""
        mock_solver = mock.MagicMock()
        fom = _make_fom(solver_ref=mock_solver)

        # Mock reduce() so we don't need real MOR
        fake_mor = mock.MagicMock()
        fom.reduce = mock.MagicMock(return_value=fake_mor)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fom.rom
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

        assert result is fake_mor

    def test_rom_is_cached(self):
        """Second access should return cached result, no new warning."""
        mock_solver = mock.MagicMock()
        fom = _make_fom(solver_ref=mock_solver)
        fake_mor = mock.MagicMock()
        fom.reduce = mock.MagicMock(return_value=fake_mor)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _ = fom.rom

        # Second access — reduce() should NOT be called again
        fom.reduce.reset_mock()
        result2 = fom.rom
        fom.reduce.assert_not_called()
        assert result2 is fake_mor


# ---------------------------------------------------------------------------
# FOMCollection.roms (backward-compatible)
# ---------------------------------------------------------------------------

class TestFOMCollectionRomsAccessor:
    """FOMCollection.roms should call reduce() and emit DeprecationWarning."""

    def test_roms_without_fds_raises(self):
        fom0 = _make_fom('d0')
        fom1 = _make_fom('d1')
        coll = FOMCollection([fom0, fom1])
        with pytest.raises(RuntimeError, match="no reference"):
            _ = coll.roms

    def test_roms_emits_deprecation_warning(self):
        fom0 = _make_fom('d0')
        fom1 = _make_fom('d1')
        coll = FOMCollection([fom0, fom1], _fds_ref=mock.MagicMock())

        fake_roms = mock.MagicMock(spec=ROMCollection)
        coll.reduce = mock.MagicMock(return_value=fake_roms)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = coll.roms
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1
            assert "deprecated" in str(dep_warnings[0].message).lower()

        assert result is fake_roms

    def test_roms_is_cached(self):
        fom0 = _make_fom('d0')
        fom1 = _make_fom('d1')
        coll = FOMCollection([fom0, fom1], _fds_ref=mock.MagicMock())
        fake_roms = mock.MagicMock(spec=ROMCollection)
        coll.reduce = mock.MagicMock(return_value=fake_roms)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _ = coll.roms

        coll.reduce.reset_mock()
        result2 = coll.roms
        coll.reduce.assert_not_called()
        assert result2 is fake_roms


# ---------------------------------------------------------------------------
# FOMCollection.concat (backward-compatible)
# ---------------------------------------------------------------------------

class TestFOMCollectionConcatAccessor:
    """FOMCollection.concat should call concatenate() and emit DeprecationWarning."""

    def test_concat_without_fds_raises(self):
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        with pytest.raises(RuntimeError, match="no reference"):
            _ = coll.concat

    def test_concat_emits_deprecation_warning(self):
        coll = FOMCollection([_make_fom('a'), _make_fom('b')], _fds_ref=mock.MagicMock())
        fake_cs = mock.MagicMock()
        coll.concatenate = mock.MagicMock(return_value=fake_cs)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = coll.concat
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

        assert result is fake_cs

    def test_concat_is_cached(self):
        coll = FOMCollection([_make_fom('a'), _make_fom('b')], _fds_ref=mock.MagicMock())
        fake_cs = mock.MagicMock()
        coll.concatenate = mock.MagicMock(return_value=fake_cs)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _ = coll.concat

        coll.concatenate.reset_mock()
        result2 = coll.concat
        coll.concatenate.assert_not_called()
        assert result2 is fake_cs


# ---------------------------------------------------------------------------
# ROMCollection.concat (backward-compatible)
# ---------------------------------------------------------------------------

class TestROMCollectionConcatAccessor:
    """ROMCollection.concat should call concatenate() and emit DeprecationWarning."""

    def test_concat_emits_deprecation_warning(self):
        mor = mock.MagicMock()
        mor.domains = ['d0', 'd1']
        mor.n_domains = 2
        roms = ROMCollection(_fds_ref=mock.MagicMock(), _mor_ref=mor)

        fake_cs = mock.MagicMock()
        roms.concatenate = mock.MagicMock(return_value=fake_cs)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = roms.concat
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1

        assert result is fake_cs

    def test_concat_is_cached(self):
        mor = mock.MagicMock()
        mor.domains = ['d0', 'd1']
        mor.n_domains = 2
        roms = ROMCollection(_fds_ref=mock.MagicMock(), _mor_ref=mor)
        fake_cs = mock.MagicMock()
        roms.concatenate = mock.MagicMock(return_value=fake_cs)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _ = roms.concat

        roms.concatenate.reset_mock()
        result2 = roms.concat
        roms.concatenate.assert_not_called()
        assert result2 is fake_cs


# ---------------------------------------------------------------------------
# Full chain: fds.fom.rom, fds.foms.roms.concat.rom (mocked)
# ---------------------------------------------------------------------------

class TestChainCalls:
    """
    Verify that the full chain-call patterns work with mocks.
    The chains all use deprecation-warning properties that delegate to
    reduce() / concatenate().
    """

    def test_fom_rom_chain(self):
        """fds.fom.rom should work (FOMResult -> MOR via reduce)."""
        fom = _make_fom(solver_ref=mock.MagicMock())
        fake_mor = mock.MagicMock()
        fom.reduce = mock.MagicMock(return_value=fake_mor)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert fom.rom is fake_mor

    def test_foms_roms_chain(self):
        """fds.foms.roms should produce ROMCollection."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')], _fds_ref=mock.MagicMock())
        fake_roms = mock.MagicMock(spec=ROMCollection)
        coll.reduce = mock.MagicMock(return_value=fake_roms)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = coll.roms
            assert result is fake_roms

    def test_foms_roms_concat_chain(self):
        """fds.foms.roms.concat should produce ConcatenatedSystem."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')], _fds_ref=mock.MagicMock())

        fake_concat = mock.MagicMock()
        fake_roms = mock.MagicMock(spec=ROMCollection)
        fake_roms.concat = mock.PropertyMock(return_value=fake_concat)
        # Since concat is a property on ROMCollection, we mock it as an attribute
        type(fake_roms).concat = mock.PropertyMock(return_value=fake_concat)

        coll.reduce = mock.MagicMock(return_value=fake_roms)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            roms = coll.roms
            cs = roms.concat
            assert cs is fake_concat

    def test_foms_roms_concat_rom_chain(self):
        """fds.foms.roms.concat.rom should produce ReducedConcatenatedSystem."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')], _fds_ref=mock.MagicMock())

        fake_reduced = mock.MagicMock()
        fake_concat = mock.MagicMock()
        type(fake_concat).rom = mock.PropertyMock(return_value=fake_reduced)

        fake_roms = mock.MagicMock(spec=ROMCollection)
        type(fake_roms).concat = mock.PropertyMock(return_value=fake_concat)

        coll.reduce = mock.MagicMock(return_value=fake_roms)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = coll.roms.concat.rom
            assert result is fake_reduced

    def test_foms_concat_rom_chain(self):
        """fds.foms.concat.rom should produce reduced system."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')], _fds_ref=mock.MagicMock())

        fake_reduced = mock.MagicMock()
        fake_concat = mock.MagicMock()
        type(fake_concat).rom = mock.PropertyMock(return_value=fake_reduced)

        coll.concatenate = mock.MagicMock(return_value=fake_concat)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = coll.concat.rom
            assert result is fake_reduced


# ---------------------------------------------------------------------------
# Persistence round-trip: FOMResult save/load
# ---------------------------------------------------------------------------

class TestFOMResultPersistence:
    """Save/load of FOMResult preserves all data."""

    def test_fom_result_round_trip(self, tmp_path):
        freqs = np.linspace(1e9, 2e9, 10)
        Z = np.random.randn(10, 2, 2) + 1j * np.random.randn(10, 2, 2)
        S = Z * 0.1

        fom = FOMResult(
            domain='test',
            frequencies=freqs,
            Z_matrix=Z,
            S_matrix=S,
            Z_dict=None,
            S_dict=None,
            n_ports=2,
            ports=['P1', 'P2']
        )

        save_path = tmp_path / "fom_test"
        fom.save(save_path)

        loaded = FOMResult.load(save_path)

        assert loaded.domain == 'test'
        assert loaded.n_ports == 2
        np.testing.assert_array_almost_equal(loaded.frequencies, freqs)
        np.testing.assert_array_almost_equal(loaded._Z_matrix, Z)
        np.testing.assert_array_almost_equal(loaded._S_matrix, S)

        # Verify lazy reconstruction of S_dict
        assert loaded.S_dict is not None
        assert '1(1)1(1)' in loaded.S_dict

    def test_fom_collection_round_trip(self, tmp_path):
        freqs = np.linspace(1e9, 2e9, 5)
        fom1 = FOMResult(
            domain='D1', frequencies=freqs,
            Z_matrix=np.random.randn(5, 1, 1), S_matrix=None,
            Z_dict=None, S_dict=None, n_ports=1, ports=['P1']
        )
        fom2 = FOMResult(
            domain='D2', frequencies=freqs,
            Z_matrix=np.random.randn(5, 1, 1), S_matrix=None,
            Z_dict=None, S_dict=None, n_ports=1, ports=['P2']
        )

        coll = FOMCollection([fom1, fom2])
        save_path = tmp_path / "coll_test"
        coll.save(save_path)

        loaded = FOMCollection.load(save_path)

        assert len(loaded) == 2
        assert loaded[0].domain == 'D1'
        assert loaded[1].domain == 'D2'
        np.testing.assert_array_almost_equal(loaded[0]._Z_matrix, fom1._Z_matrix)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
