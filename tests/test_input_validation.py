"""Inputs that used to be ignored or accepted silently are rejected clearly."""
from pathlib import Path

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.solvers.base import ParameterConverter
from cavsim3d.solvers.options import validate_sweep

CFG = dict(fmin=1.8, fmax=2.4, nsamples=3, nportmodes=1, order=1)
STEP = (Path(__file__).resolve().parents[1] / "docs" / "example_models"
        / "rectangular_waveguide.step")


def _guide():
    return RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)


@pytest.fixture
def solved(tmp_path):
    p = EMProject("solved", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    p.fds.solve(config=CFG)
    return p


# --- solve() options --------------------------------------------------------

def test_solve_rejects_a_misspelt_option_and_suggests_the_right_one(tmp_path):
    p = EMProject("opt", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    with pytest.raises(TypeError, match="did you mean 'nportmodes'"):
        p.fds.solve(fmin=1.8, fmax=2.4, nsamples=3, n_port_modes=2)
    with pytest.raises(TypeError, match="per_domian"):
        p.fds.solve(config=dict(CFG, per_domian=False))


def test_reduced_models_accept_a_reused_config_but_not_typos(solved):
    rom = solved.fds.fom.reduce(tol=1e-6)
    rom.solve(config=CFG)                 # full-order options: accepted, no effect
    with pytest.raises(TypeError, match="did you mean 'nsamples'"):
        rom.solve(fmin=1.8, fmax=2.4, nsample=3)


@pytest.mark.parametrize("fmin, fmax, nsamples, message", [
    (0, 2, 3, "fmin must be > 0"),
    (float("nan"), 2, 3, "finite"),
    (1, float("inf"), 3, "finite"),
    (2, 1, 3, "fmax"),
    (1, 2, 0, ">= 1"),
    (1, 2, 2.5, "whole number"),
    (1, 2, "3", "whole number"),
    (1, 2, True, "whole number"),
])
def test_a_sweep_that_cannot_be_solved_is_refused(fmin, fmax, nsamples, message):
    with pytest.raises(ValueError, match=message):
        validate_sweep(fmin, fmax, nsamples)


def test_a_whole_float_sample_count_is_accepted(tmp_path):
    p = EMProject("floaty", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    p.fds.solve(config=dict(CFG, nsamples=3.0))
    assert len(p.fds.frequencies) == 3


# --- materials ---------------------------------------------------------------

@pytest.mark.parametrize("props, message", [
    ({"eps_r": 2.2 - 0.01j}, "tan_delta"),
    ({"eps_r": "2.2"}, "real number"),
    ({"eps_r": True}, "real number"),
    ({"mu_r": 0}, "> 0"),
    ({"eps_r": -1.0}, "> 0"),
    ({"sigma": -1.0}, ">= 0"),
    ({"tan_delta": float("nan")}, "finite"),
])
def test_unusable_material_values_are_refused(props, message):
    with pytest.raises(ValueError, match=message):
        _guide().set_materials({"*": props})


def test_the_importer_refuses_a_material_that_is_not_a_dict():
    from cavsim3d.geometry.importers import OCCImporter
    geo = OCCImporter(str(STEP), unit="m")
    with pytest.raises(ValueError, match="expected a dict"):
        geo.set_materials({"*": "copper"})
    with pytest.raises(ValueError, match="unknown properties"):
        geo.set_materials({"*": {"permittivity": 2.2}})


def test_the_solver_refuses_material_values_set_around_the_checks(tmp_path):
    g = _guide()
    p = EMProject("raw", base_dir=tmp_path, overwrite=True)
    p.geometry = g
    fds = p.fds                                           # built with valid materials
    g._materials = {"*": {"eps_r": 2.0 - 1.0j}}           # bypasses set_materials
    with pytest.raises(ValueError, match="real number"):
        fds._material_props("vacuum")


# --- reduction ---------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [dict(max_rank=0), dict(max_rank=-1),
                                    dict(max_rank=2.5), dict(tol=-1e-6),
                                    dict(tol=float("nan"))])
def test_reduction_arguments_are_checked(solved, kwargs):
    with pytest.raises(ValueError, match="max_rank|tol"):
        solved.fds.fom.reduce(**{"tol": 1e-6, **kwargs})
    from cavsim3d.solvers.concatenation import reduce_concatenated_system
    with pytest.raises(ValueError, match="max_rank|tol"):
        reduce_concatenated_system(None, **{"tol": 1e-6, **kwargs})


# --- export --------------------------------------------------------------------

def test_touchstone_export_takes_a_path_and_is_exact_at_50_ohm(solved, tmp_path):
    written = solved.fds.export_touchstone(tmp_path / "guide")
    assert written.endswith(".s2p")
    lines = Path(written).read_text().splitlines()
    assert "# GHz S MA R 50.0" in lines
    first = next(ln for ln in lines if ln and ln[0].isdigit()).split()
    s21_mag = float(first[3])                   # 2-port order: S11 S21 S12 S22
    S50 = ParameterConverter.z_to_s(solved.fds.fom._Z_matrix, 50.0)
    assert s21_mag == pytest.approx(abs(S50[0, 1, 0]), rel=1e-6)
    with pytest.warns(UserWarning, match="R 50"):
        solved.fds.export_touchstone(tmp_path / "as_solved", z0=None)


# --- analysis helpers ------------------------------------------------------------

def test_band_difference_covers_the_whole_band_by_default():
    from cavsim3d.analysis import band_difference
    f = np.linspace(1.0, 2.0, 5)
    A = np.ones((5, 2, 2))
    assert band_difference(A, 0.5 * A, freq=f) == {"1.000-2.000": 0.5}
    empty = band_difference(A, 0.5 * A, freq=f, edges=[3.0, 4.0])
    assert np.isnan(empty["3.000-4.000"])


def test_plot_helpers_refuse_empty_input():
    from cavsim3d.analysis import plot_entries, plot_matrix
    with pytest.raises(ValueError, match="models is empty"):
        plot_matrix({})
    with pytest.raises(ValueError, match="entries is empty"):
        plot_entries({"a": np.ones((2, 2, 2))}, [], freq=[1.0, 2.0])
