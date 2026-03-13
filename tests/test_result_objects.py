"""
Unit tests for result wrapper objects.

These tests use mock data — no NGSolve dependency required.
They verify:
  - FOMResult / FOMCollection / ROMCollection expose correct Z_dict, S_dict, frequencies
  - PlotMixin.plot_s / plot_z return (fig, ax) and accept an existing ax
  - FOMCollection / ROMCollection indexing, len, iter
  - DataExtractor recognises new result types
  - FOMResult.reduce() and .concatenate() guards
"""

import numpy as np
import pytest
import warnings
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for CI
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers.results import FOMResult, FOMCollection, ROMCollection
from utils.visualization import DataExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_frequencies(n=50):
    return np.linspace(1.0, 3.0, n) * 1e9


def _make_z_dict(n=50):
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


def _make_fom(domain='default', n=50):
    freqs = _make_frequencies(n)
    z = _make_z_dict(n)
    # Build a dummy S_dict too
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
    )


# ---------------------------------------------------------------------------
# FOMResult tests
# ---------------------------------------------------------------------------

class TestFOMResult:
    def test_init(self):
        fom = _make_fom()
        assert fom.domain == 'default'
        assert fom.n_ports == 2
        assert len(fom.frequencies) == 50

    def test_z_dict(self):
        fom = _make_fom()
        assert '1(1)1(1)' in fom.Z_dict
        assert '2(1)2(1)' in fom.Z_dict

    def test_s_dict(self):
        fom = _make_fom()
        assert '1(1)1(1)' in fom.S_dict

    def test_repr(self):
        fom = _make_fom()
        r = repr(fom)
        assert 'FOMResult' in r
        assert 'default' in r

    def test_reduce_without_solver_raises(self):
        """reduce() without a solver reference should raise RuntimeError."""
        fom = _make_fom()
        with pytest.raises(RuntimeError, match="no solver reference"):
            fom.reduce()

    def test_concatenate_single_warns(self):
        """concatenate() on single-solid FOM should warn."""
        fom = _make_fom()
        with pytest.warns(UserWarning, match="not available on a single FOMResult"):
            result = fom.concatenate()
        assert result is None


# ---------------------------------------------------------------------------
# PlotMixin tests (via FOMResult)
# ---------------------------------------------------------------------------

class TestPlotMixin:
    def test_plot_s_creates_figure(self):
        fom = _make_fom()
        fig, ax = fom.plot_s()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_z_creates_figure(self):
        fom = _make_fom()
        fig, ax = fom.plot_z()
        assert fig is not None
        plt.close(fig)

    def test_plot_s_accepts_existing_ax(self):
        fom = _make_fom()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = fom.plot_s(ax=ax1)
        assert ax2 is ax1
        assert fig2 is fig1
        plt.close(fig1)

    def test_plot_s_specific_params(self):
        fom = _make_fom()
        fig, ax = fom.plot_s(params=['1(1)1(1)'])
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close(fig)

    def test_plot_s_overlay_two_foms(self):
        """Verify overlaying two FOMResults on the same axes."""
        fom1 = _make_fom('domain_a')
        fom2 = _make_fom('domain_b')
        fig, ax = fom1.plot_s(params=['1(1)1(1)'], label='A')
        fig, ax = fom2.plot_s(params=['1(1)1(1)'], ax=ax, label='B')
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)

    def test_plot_z_various_types(self):
        fom = _make_fom()
        for pt in ['db', 'mag', 'phase', 're', 'im']:
            fig, ax = fom.plot_z(plot_type=pt, params=['1(1)1(1)'])
            plt.close(fig)


# ---------------------------------------------------------------------------
# FOMCollection tests
# ---------------------------------------------------------------------------

class TestFOMCollection:
    def test_indexing(self):
        fom0 = _make_fom('cell0')
        fom1 = _make_fom('cell1')
        coll = FOMCollection([fom0, fom1])
        assert coll[0].domain == 'cell0'
        assert coll[1].domain == 'cell1'

    def test_len(self):
        coll = FOMCollection([_make_fom(), _make_fom()])
        assert len(coll) == 2

    def test_iter(self):
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        names = [f.domain for f in coll]
        assert names == ['a', 'b']

    def test_plot_s_overlays(self):
        coll = FOMCollection([_make_fom('c0'), _make_fom('c1')])
        fig, ax = coll.plot_s(params=['1(1)1(1)'])
        lines = ax.get_lines()
        assert len(lines) == 2   # one per domain
        plt.close(fig)

    def test_repr(self):
        coll = FOMCollection([_make_fom('x'), _make_fom('y')])
        assert 'x' in repr(coll)
        assert 'y' in repr(coll)

    def test_reduce_without_fds_raises(self):
        """reduce() without FDS reference should raise."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        with pytest.raises(RuntimeError, match="no reference"):
            coll.reduce()

    def test_concatenate_without_fds_raises(self):
        """concatenate() without FDS reference should raise."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        with pytest.raises(RuntimeError, match="no reference"):
            coll.concatenate()


# ---------------------------------------------------------------------------
# DataExtractor recognises new types
# ---------------------------------------------------------------------------

class TestDataExtractorNewTypes:
    def test_fom_result_is_solver(self):
        fom = _make_fom()
        assert DataExtractor.get_source_type(fom) == 'solver'

    def test_fom_result_label(self):
        fom = _make_fom()
        assert DataExtractor.get_label(fom) == 'FOM'

    def test_fom_result_style(self):
        fom = _make_fom()
        style = DataExtractor.get_style(fom)
        assert 'linestyle' in style

    def test_extract_z_from_fom_result(self):
        fom = _make_fom()
        freqs, z = DataExtractor.extract_z_parameters(fom)
        assert len(z) == 50

    def test_extract_s_from_fom_result(self):
        fom = _make_fom()
        freqs, s = DataExtractor.extract_s_parameters(fom)
        assert len(s) == 50


# ---------------------------------------------------------------------------
# Regression: .rom / .roms / .concat must NOT auto-trigger computation
# ---------------------------------------------------------------------------

class TestNoAutoCompute:
    """
    Regression guard: accessing .rom, .roms, or .concat when no cached
    result exists must raise RuntimeError — never silently trigger an
    expensive reduce() or concatenate() call.
    """

    def test_fom_rom_does_not_auto_reduce(self):
        """FOMResult.rom must raise when no ROM has been computed."""
        fom = _make_fom()
        assert fom._rom_cache is None
        with pytest.raises(RuntimeError, match="Call fom.reduce"):
            _ = fom.rom

    def test_fom_rom_returns_cached(self):
        """FOMResult.rom returns the cached ROM when set explicitly."""
        fom = _make_fom()
        sentinel = object()
        fom._rom_cache = sentinel
        assert fom.rom is sentinel

    def test_fom_collection_roms_does_not_auto_reduce(self):
        """FOMCollection.roms must raise when no ROMs have been computed."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        assert coll._roms_cache is None
        with pytest.raises(RuntimeError, match="Call foms.reduce"):
            _ = coll.roms

    def test_fom_collection_roms_returns_cached(self):
        """FOMCollection.roms returns cached ROMCollection when set."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        sentinel = object()
        coll._roms_cache = sentinel
        assert coll.roms is sentinel

    def test_fom_collection_concat_does_not_auto_concatenate(self):
        """FOMCollection.concat must raise when not computed."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        assert coll._concat_cache is None
        with pytest.raises(RuntimeError, match="Call foms.concatenate"):
            _ = coll.concat

    def test_fom_collection_concat_returns_cached(self):
        """FOMCollection.concat returns cached system when set."""
        coll = FOMCollection([_make_fom('a'), _make_fom('b')])
        sentinel = object()
        coll._concat_cache = sentinel
        assert coll.concat is sentinel

    def test_rom_collection_concat_does_not_auto_concatenate(self):
        """ROMCollection.concat must raise when not computed."""
        import unittest.mock as mock
        mor = mock.MagicMock()
        mor.domains = ['a']
        mor.n_domains = 1
        roms = ROMCollection(_mor_ref=mor)
        with pytest.raises(RuntimeError, match="Call roms.concatenate"):
            _ = roms.concat

    def test_rom_collection_concat_returns_cached(self):
        """ROMCollection.concat returns cached system when set."""
        import unittest.mock as mock
        mor = mock.MagicMock()
        mor.domains = ['a']
        mor.n_domains = 1
        roms = ROMCollection(_mor_ref=mor)
        sentinel = object()
        roms._concat_cache = sentinel
        assert roms.concat is sentinel


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
