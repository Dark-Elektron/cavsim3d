"""
Model-comparison utilities (cavsim3d.analysis.network), the solver convergence
plot (PlotMixin.plot_residual) and the per-batch timing line of the FOM solve.

Validates:
  - network_matrix reads both key conventions (excitation first / CST)
  - keep_port_modes == open-circuiting the dropped modes (Z' = Z_kk)
  - band_difference, plot_matrix, plot_entries on arrays and result objects
  - plot_residual layouts, largest-residual summary, maxsteps marker
  - _report_batch prints one tab deeper, with GMRES stats when iterative
"""

import logging
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cavsim3d.solvers.frequency_domain  # noqa: F401  (import order)
from cavsim3d.analysis import (band_difference, keep_port_modes, network_matrix,
                               plot_entries, plot_matrix, port_mode_labels)
from cavsim3d.analytical.cst_result import CSTResult
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from cavsim3d.utils.plot_mixin import PlotMixin

LABELS = ["1(1)", "1(2)", "2(1)", "3(1)"]


def _network(n_freq=7, seed=0):
    """Random reciprocal network: symmetric Z, pseudo-wave S with unit reference."""
    rng = np.random.default_rng(seed)
    n = len(LABELS)
    Z = rng.normal(size=(n_freq, n, n)) + 1j * rng.normal(size=(n_freq, n, n))
    Z = Z + np.swapaxes(Z, 1, 2) + 4 * np.eye(n)
    S = (Z - np.eye(n)) @ np.linalg.inv(Z + np.eye(n))
    return Z, S


def _as_dict(A, excitation_first=True):
    d = {}
    for r, lr in enumerate(LABELS):
        for c, lc in enumerate(LABELS):
            d[(lc + lr) if excitation_first else (lr + lc)] = A[:, r, c]
    return d


def _model(Z, S, freq_hz):
    return SimpleNamespace(S_dict={**_as_dict(S), "frequencies": freq_hz},
                           Z_dict={**_as_dict(Z), "frequencies": freq_hz},
                           frequencies=freq_hz)


class TestNetworkMatrix:
    def test_excitation_first_keys(self):
        Z, S = _network()
        S[:, 0, 2] += 1.0                      # break symmetry: orientation matters
        model = _model(Z, S, np.linspace(1e9, 2e9, 7))
        assert port_mode_labels(model) == LABELS
        np.testing.assert_allclose(network_matrix(model), S)
        sub = network_matrix(model, ["2(1)", "1(1)"])
        np.testing.assert_allclose(sub[:, 1, 0], S[:, 0, 2])

    def test_cst_keys_are_response_first(self):
        Z, S = _network()
        S[:, 0, 2] += 1.0
        cst = CSTResult.__new__(CSTResult)
        cst._S_dict = _as_dict(S, excitation_first=False)
        np.testing.assert_allclose(network_matrix(cst, LABELS), S)

    def test_labels_sort_numerically(self):
        d = {"10(1)2(1)": 0, "2(1)10(1)": 0, "2(1)2(1)": 0, "10(1)10(1)": 0}
        assert port_mode_labels(d) == ["2(1)", "10(1)"]

    def test_missing_entry_names_available_modes(self):
        Z, S = _network()
        with pytest.raises(KeyError, match="available port modes"):
            network_matrix(_model(Z, S, np.arange(7.0)), ["1(1)", "9(1)"])


class TestKeepPortModes:
    def test_equals_open_circuit_termination(self):
        Z, S = _network()
        keep = ["2(1)", "1(1)"]                     # also reorders
        S_k, Z_k = keep_port_modes(_model(Z, S, np.arange(7.0)), keep)
        idx = [LABELS.index(k) for k in keep]
        Z_kk = Z[:, idx][:, :, idx]
        np.testing.assert_allclose(Z_k, Z_kk)
        # I = 0 on the dropped modes: the kept network is Z_kk itself
        I2 = np.eye(len(keep))
        np.testing.assert_allclose(S_k, (Z_kk - I2) @ np.linalg.inv(Z_kk + I2), atol=1e-12)

    def test_keeping_everything_is_identity(self):
        Z, S = _network()
        S_k, _ = keep_port_modes(_model(Z, S, np.arange(7.0)), LABELS)
        np.testing.assert_allclose(S_k, S)


class TestComparison:
    def test_band_difference(self):
        f = np.arange(1, 11) / 10                 # 0.1 ... 1.0 GHz
        A = np.zeros((10, 2, 2))
        B = np.zeros((10, 2, 2))
        B[f > 0.55, 0, 0] = 0.4
        out = band_difference(A, B, f, [0.1, 0.55, 1.0])
        assert list(out) == ["0.100-0.550", "0.550-1.000"]
        assert out["0.100-0.550"] == 0.0
        assert out["0.550-1.000"] == pytest.approx(0.1)          # 0.4 on 1 of 4 entries
        assert band_difference(A, B, f, [0.55, 1.0], ports=[0])["0.550-1.000"] == pytest.approx(0.4)

    def test_band_difference_needs_same_grid(self):
        with pytest.raises(ValueError, match="frequency grid"):
            band_difference(np.zeros((5, 2, 2)), np.zeros((5, 3, 3)), np.arange(5.0), [0, 4])

    def test_plots_mix_arrays_and_results(self):
        Z, S = _network()
        f_hz = np.linspace(1e9, 2e9, 7)
        models = {"ref": S, "model": _model(Z, S, f_hz)}
        fig, axs = plot_matrix(models, freq=f_hz / 1e9, ports=[0, 2], labels=["1", "1", "2", "3"])
        assert axs.shape == (2, 2)
        assert axs[0, 1].get_title() == "$S_{12}$"
        lines = axs[0, 0].get_lines()
        assert [l.get_label() for l in lines] == ["ref", "model"]
        assert lines[0].get_linewidth() > lines[1].get_linewidth()     # reference drawn wide
        fig2, axs2 = plot_entries(models, [(0, 2), (1, 1, "own title")], freq=f_hz / 1e9, kind="S")
        assert axs2[1].get_title() == "own title"
        plt.close("all")

    def test_array_models_need_freq(self):
        with pytest.raises(ValueError, match="freq"):
            plot_matrix({"a": np.zeros((3, 2, 2))})


class _Residuals(PlotMixin):
    def __init__(self, iters, res, maxsteps=None):
        self.domain = "global"
        self.mode_labels = [(1, 1), (2, 1)]
        self._residual_data = {
            "frequencies": np.linspace(1e9, 2e9, len(iters)),
            "iterations": iters.mean(axis=1), "residuals": res.min(axis=1),
            "iterations_per_excitation": iters, "residuals_per_excitation": res,
            "solver_type": "iterative",
        }
        if maxsteps is not None:
            self._residual_data["maxsteps"] = maxsteps


class TestPlotResidual:
    iters = np.array([[10, 12], [40, 50], [95, 100]])
    res = np.array([[1e-9, 1e-8], [2e-9, 3e-9], [1e-7, 1e-9]])

    def test_both_is_two_stacked_panels(self):
        fig, (ax_it, ax_res) = _Residuals(self.iters, self.res).plot_residual()
        assert ax_it is not ax_res and ax_it.get_shared_x_axes().joined(ax_it, ax_res)
        # summary curves: mean steps, LARGEST residual (unconverged solves stay visible)
        np.testing.assert_allclose(ax_it.get_lines()[0].get_ydata(), self.iters.mean(axis=1))
        np.testing.assert_allclose(ax_res.get_lines()[0].get_ydata(),
                                   self.res.max(axis=1) + 1e-30)
        plt.close("all")

    def test_per_excitation_labels_and_colours(self):
        fig, (ax_it, ax_res) = _Residuals(self.iters, self.res).plot_residual(per_excitation=True)
        it_lines, res_lines = ax_it.get_lines(), ax_res.get_lines()
        assert [l.get_label() for l in it_lines] == ["1(1)", "2(1)"]
        assert [l.get_color() for l in it_lines] == [l.get_color() for l in res_lines]
        plt.close("all")

    def test_maxsteps_line_only_when_close(self):
        _, (ax_it, _) = _Residuals(self.iters, self.res, maxsteps=100).plot_residual()
        assert "maxsteps" in [l.get_label() for l in ax_it.get_lines()]
        _, (ax_it, _) = _Residuals(self.iters, self.res, maxsteps=500).plot_residual()
        assert "maxsteps" not in [l.get_label() for l in ax_it.get_lines()]
        plt.close("all")

    def test_single_quantity(self):
        fig, ax = _Residuals(self.iters, self.res).plot_residual("residual", per_excitation=True)
        assert ax.get_yscale() == "log" and len(ax.get_lines()) == 2
        with pytest.raises(ValueError):
            _Residuals(self.iters, self.res).plot_residual("steps")
        plt.close("all")


class TestBatchReport:
    def test_iterative_line(self, caplog):
        iters = [10, 12, 500, 20]                 # 2 samples x 2 excitations
        res = [1e-9, 2e-9, 3e-4, 1e-9]
        with caplog.at_level(logging.DEBUG, logger="cavsim3d"):
            FrequencyDomainSolver._report_batch(0, 1, 0.0, iters, res, 2, "iterative",
                                                {"maxsteps": 500})
        msg = caplog.records[-1].getMessage()
        assert msg.startswith("  \tsamples 1-2: ")
        assert "s/sample" in msg and "GMRES 136 steps/solve (max 500)" in msg
        assert "residual <= 3.0e-04" in msg and "1 solve(s) stopped at maxsteps=500" in msg

    def test_direct_line(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="cavsim3d"):
            FrequencyDomainSolver._report_batch(5, 9, 0.0, [0] * 10, [0.0] * 10, 1, "direct", {})
        msg = caplog.records[-1].getMessage()
        assert msg.startswith("  \tsamples 6-10: ") and "GMRES" not in msg
