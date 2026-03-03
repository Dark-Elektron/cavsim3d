"""
Unit tests for result wrapper objects.

These tests use mock data — no NGSolve dependency required.
They verify:
  - FOMResult / ROMResult / ConcatResult expose correct Z_dict, S_dict, frequencies
  - PlotMixin.plot_s / plot_z return (fig, ax) and accept an existing ax
  - FOMCollection / ROMCollection indexing, len, iter
  - DataExtractor recognises new result types
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for CI
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solvers.results import FOMResult, ROMResult, ConcatResult, FOMCollection, ROMCollection
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


def _make_rom(domain='default', n=50):
    freqs = _make_frequencies(n)
    z = _make_z_dict(n)
    s = {k: v * 0.5 for k, v in z.items() if k != 'frequencies'}
    s['frequencies'] = freqs
    return ROMResult(
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

    def test_rom_without_factory_raises(self):
        fom = _make_fom()
        with pytest.raises(RuntimeError, match="ROM factory"):
            _ = fom.rom


# ---------------------------------------------------------------------------
# ROMResult tests
# ---------------------------------------------------------------------------

class TestROMResult:
    def test_init(self):
        rom = _make_rom()
        assert rom.domain == 'default'
        assert rom.n_ports == 2

    def test_eigenvalues_without_ref_raises(self):
        rom = _make_rom()
        with pytest.raises(RuntimeError, match="Eigenvalues"):
            rom.get_eigenvalues()

    def test_repr(self):
        rom = _make_rom()
        assert 'ROMResult' in repr(rom)


# ---------------------------------------------------------------------------
# PlotMixin tests  (via FOMResult / ROMResult)
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

    def test_plot_s_overlay_fom_rom(self):
        """Verify overlaying FOM and ROM on the same axes."""
        fom = _make_fom()
        rom = _make_rom()
        fig, ax = fom.plot_s(params=['1(1)1(1)'], label='FOM')
        fig, ax = rom.plot_s(params=['1(1)1(1)'], ax=ax, label='ROM')
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


# ---------------------------------------------------------------------------
# ROMCollection tests
# ---------------------------------------------------------------------------

class TestROMCollection:
    def test_indexing(self):
        r0 = _make_rom('d0')
        r1 = _make_rom('d1')
        coll = ROMCollection([r0, r1])
        assert coll[0].domain == 'd0'

    def test_plot_z_overlays(self):
        coll = ROMCollection([_make_rom('a'), _make_rom('b')])
        fig, ax = coll.plot_z(params=['1(1)1(1)'])
        assert len(ax.get_lines()) == 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# DataExtractor recognises new types
# ---------------------------------------------------------------------------

class TestDataExtractorNewTypes:
    def test_fom_result_is_solver(self):
        fom = _make_fom()
        assert DataExtractor.get_source_type(fom) == 'solver'

    def test_rom_result_is_solver(self):
        rom = _make_rom()
        assert DataExtractor.get_source_type(rom) == 'solver'

    def test_fom_result_label(self):
        fom = _make_fom()
        assert DataExtractor.get_label(fom) == 'FOM'

    def test_rom_result_label(self):
        rom = _make_rom()
        assert DataExtractor.get_label(rom) == 'ROM'

    def test_fom_result_style(self):
        fom = _make_fom()
        style = DataExtractor.get_style(fom)
        assert 'linestyle' in style

    def test_extract_z_from_fom_result(self):
        fom = _make_fom()
        freqs, z = DataExtractor.extract_z_parameters(fom)
        assert len(z) == 50

    def test_extract_s_from_rom_result(self):
        rom = _make_rom()
        freqs, s = DataExtractor.extract_s_parameters(rom)
        assert len(s) == 50


# ---------------------------------------------------------------------------
# ConcatResult tests
# ---------------------------------------------------------------------------

class TestConcatResult:
    def test_basic(self):
        freqs = _make_frequencies()
        z = _make_z_dict()
        s = {k: v * 0.5 for k, v in z.items() if k != 'frequencies'}
        s['frequencies'] = freqs
        cr = ConcatResult(
            concat_system=None,
            frequencies=freqs,
            Z_matrix=None,
            S_matrix=None,
            Z_dict=z,
            S_dict=s,
            n_ports=2,
            ports=['port1', 'port2'],
        )
        assert cr.n_ports == 2
        fig, ax = cr.plot_s(params=['1(1)1(1)'])
        assert fig is not None
        plt.close(fig)

    def test_repr(self):
        freqs = _make_frequencies()
        cr = ConcatResult(
            concat_system=None,
            frequencies=freqs,
            Z_matrix=None,
            S_matrix=None,
            Z_dict=_make_z_dict(),
            S_dict=_make_z_dict(),
            n_ports=2,
            ports=['port1', 'port2'],
        )
        assert 'ConcatResult' in repr(cr)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
