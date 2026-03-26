"""
Tests for the result wrapper hierarchy and chain accessors.

Merged from: test_result_objects.py, test_chain_accessors.py

Validates:
  - FOMResult / FOMCollection / ROMCollection creation and data access
  - PlotMixin plotting methods
  - DataExtractor recognition of result types
  - No-auto-compute guards (.rom, .roms, .concat)
  - Backward-compatible deprecation-warning chain accessors
  - Full mock chain: fds.fom.rom, fds.foms.roms.concat.rom
"""

import numpy as np
import pytest
import unittest.mock as mock
import matplotlib.pyplot as plt
from cavsim3d.solvers.results import FOMResult, FOMCollection, ROMCollection
from cavsim3d.utils.visualization import DataExtractor
from tests.helpers import make_fom




# ===========================================================================
# FOMResult tests
# ===========================================================================

class TestFOMResult:
    def test_init(self):
        fom = make_fom()
        assert fom.domain == 'default'
        assert fom.n_ports == 2
        assert len(fom.frequencies) == 50

    def test_z_dict(self):
        fom = make_fom()
        assert '1(1)1(1)' in fom.Z_dict
        assert '2(1)2(1)' in fom.Z_dict

    def test_s_dict(self):
        fom = make_fom()
        assert '1(1)1(1)' in fom.S_dict

    def test_repr(self):
        fom = make_fom()
        r = repr(fom)
        assert 'FOMResult' in r
        assert 'default' in r

    def test_reduce_without_solver_raises(self):
        fom = make_fom()
        with pytest.raises(RuntimeError, match="no solver reference"):
            fom.reduce()

    def test_concatenate_single_warns(self):
        fom = make_fom()
        with pytest.warns(UserWarning, match="not available on a single FOMResult"):
            result = fom.concatenate()
        assert result is None


# ===========================================================================
# PlotMixin tests (via FOMResult)
# ===========================================================================

class TestPlotMixin:
    def test_plot_s_creates_figure(self):
        fom = make_fom()
        fig, ax = fom.plot_s()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_z_creates_figure(self):
        fom = make_fom()
        fig, ax = fom.plot_z()
        assert fig is not None
        plt.close(fig)

    def test_plot_s_accepts_existing_ax(self):
        fom = make_fom()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = fom.plot_s(ax=ax1)
        assert ax2 is ax1
        assert fig2 is fig1
        plt.close(fig1)

    def test_plot_s_specific_params(self):
        fom = make_fom()
        fig, ax = fom.plot_s(params=['1(1)1(1)'])
        lines = ax.get_lines()
        assert len(lines) >= 1
        plt.close(fig)

    def test_plot_s_overlay_two_foms(self):
        fom1 = make_fom('domain_a')
        fom2 = make_fom('domain_b')
        fig, ax = fom1.plot_s(params=['1(1)1(1)'], label='A')
        fig, ax = fom2.plot_s(params=['1(1)1(1)'], ax=ax, label='B')
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)

    def test_plot_z_various_types(self):
        fom = make_fom()
        for pt in ['db', 'mag', 'phase', 're', 'im']:
            fig, ax = fom.plot_z(plot_type=pt, params=['1(1)1(1)'])
            plt.close(fig)


# ===========================================================================
# FOMCollection tests
# ===========================================================================

class TestFOMCollection:
    def test_indexing(self):
        coll = FOMCollection([make_fom('cell0'), make_fom('cell1')])
        assert coll[0].domain == 'cell0'
        assert coll[1].domain == 'cell1'

    def test_len(self):
        coll = FOMCollection([make_fom(), make_fom()])
        assert len(coll) == 2

    def test_iter(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        names = [f.domain for f in coll]
        assert names == ['a', 'b']

    def test_plot_s_overlays(self):
        coll = FOMCollection([make_fom('c0'), make_fom('c1')])
        fig, ax = coll.plot_s(params=['1(1)1(1)'])
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)

    def test_repr(self):
        coll = FOMCollection([make_fom('x'), make_fom('y')])
        assert 'x' in repr(coll)
        assert 'y' in repr(coll)

    def test_reduce_without_fds_raises(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        with pytest.raises(RuntimeError, match="no reference"):
            coll.reduce()

    def test_concatenate_without_fds_raises(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        with pytest.raises(RuntimeError, match="no reference"):
            coll.concatenate()


# ===========================================================================
# DataExtractor recognition
# ===========================================================================

class TestDataExtractorNewTypes:
    def test_fom_result_is_solver(self):
        assert DataExtractor.get_source_type(make_fom()) == 'solver'

    def test_fom_result_label(self):
        assert DataExtractor.get_label(make_fom()) == 'FOM'

    def test_fom_result_style(self):
        style = DataExtractor.get_style(make_fom())
        assert 'linestyle' in style

    def test_extract_z_from_fom_result(self):
        freqs, z = DataExtractor.extract_z_parameters(make_fom())
        assert len(z) == 50

    def test_extract_s_from_fom_result(self):
        freqs, s = DataExtractor.extract_s_parameters(make_fom())
        assert len(s) == 50


# ===========================================================================
# No-auto-compute guards
# ===========================================================================

class TestNoAutoCompute:
    """
    Accessing .rom, .roms, or .concat when no cached result exists must
    raise RuntimeError -- never silently trigger expensive computation.
    """

    def test_fom_rom_does_not_auto_reduce(self):
        fom = make_fom()
        assert fom._rom_cache is None
        with pytest.raises(RuntimeError, match="Call fom.reduce"):
            _ = fom.rom

    def test_fom_rom_returns_cached(self):
        fom = make_fom()
        sentinel = object()
        fom._rom_cache = sentinel
        assert fom.rom is sentinel

    def test_fom_collection_roms_does_not_auto_reduce(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        assert coll._roms_cache is None
        with pytest.raises(RuntimeError, match="Call foms.reduce"):
            _ = coll.roms

    def test_fom_collection_roms_returns_cached(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        sentinel = object()
        coll._roms_cache = sentinel
        assert coll.roms is sentinel

    def test_fom_collection_concat_does_not_auto_concatenate(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        assert coll._concat_cache is None
        with pytest.raises(RuntimeError, match="Call foms.concatenate"):
            _ = coll.concat

    def test_fom_collection_concat_returns_cached(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        sentinel = object()
        coll._concat_cache = sentinel
        assert coll.concat is sentinel

    def test_rom_collection_concat_does_not_auto_concatenate(self):
        mor = mock.MagicMock()
        mor.domains = ['a']
        mor.n_domains = 1
        mor._concatenated = None  # Prevent MagicMock auto-attribute
        roms = ROMCollection(_mor_ref=mor)
        with pytest.raises(RuntimeError, match="Call roms.concatenate"):
            _ = roms.concat

    def test_rom_collection_concat_returns_cached(self):
        mor = mock.MagicMock()
        mor.domains = ['a']
        mor.n_domains = 1
        roms = ROMCollection(_mor_ref=mor)
        sentinel = object()
        roms._concat_cache = sentinel
        assert roms.concat is sentinel


# ===========================================================================
# Backward-compatible deprecation-warning chain accessors
# ===========================================================================

# ===========================================================================
# Cache accessor behavior (current API: no auto-trigger, no deprecation)
# ===========================================================================

class TestCacheAccessors:
    """Test that .rom, .roms, .concat return cached values when set."""

    def test_fom_rom_returns_cached(self):
        fom = make_fom()
        sentinel = mock.MagicMock()
        fom._rom_cache = sentinel
        assert fom.rom is sentinel

    def test_fom_collection_roms_returns_cached(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        sentinel = mock.MagicMock()
        coll._roms_cache = sentinel
        assert coll.roms is sentinel

    def test_fom_collection_concat_returns_cached(self):
        coll = FOMCollection([make_fom('a'), make_fom('b')])
        sentinel = mock.MagicMock()
        coll._concat_cache = sentinel
        assert coll.concat is sentinel

    def test_rom_collection_concat_returns_cached(self):
        mor = mock.MagicMock()
        mor.domains = ['d0', 'd1']
        mor.n_domains = 2
        mor._concatenated = None
        roms = ROMCollection(_fds_ref=mock.MagicMock(), _mor_ref=mor)
        sentinel = mock.MagicMock()
        roms._concat_cache = sentinel
        assert roms.concat is sentinel


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
