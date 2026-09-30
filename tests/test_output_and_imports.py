"""Importing cavsim3d, and what it prints."""
import io
import logging
import subprocess
import sys
from pathlib import Path

import pytest

import cavsim3d.utils.printing as pr

ROOT = Path(__file__).resolve().parents[1]


def _python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-W", "ignore", "-c", code], cwd=ROOT,
                          capture_output=True, text=True)


@pytest.mark.parametrize("module", [
    "cavsim3d.rom", "cavsim3d.rom.reduction", "cavsim3d.rom.structures",
    "cavsim3d.solvers", "cavsim3d.solvers.concatenation", "cavsim3d.solvers.results",
    "cavsim3d.geometry", "cavsim3d.analysis", "cavsim3d.core.em_project",
])
def test_every_package_imports_first_in_a_fresh_interpreter(module):
    result = _python(f"import {module}")
    assert result.returncode == 0, result.stderr[-2000:]


def test_an_opencascade_that_netgen_cannot_share_is_reported(monkeypatch):
    # pythonocc-core 8 next to netgen's OpenCASCADE 7.8 fails in netgen's
    # import (Windows) or crashes reading a STEP file (Linux): say why instead
    from importlib import metadata

    import OCC
    from cavsim3d.geometry import importers
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(metadata, "version", lambda name: "7.8.1")
    monkeypatch.setattr(OCC, "VERSION", "8.0.1", raising=False)
    with pytest.raises(ImportError, match=r"pythonocc-core=7\.9"):
        importers._check_occt_pairing()
    monkeypatch.setattr(OCC, "VERSION", "7.9.3", raising=False)
    importers._check_occt_pairing()                 # same major version: fine


def test_the_project_class_is_exported_and_loaded_on_first_use():
    result = _python(
        "import sys, cavsim3d\n"
        "assert 'cavsim3d.core.em_project' not in sys.modules\n"
        "assert cavsim3d.EMProject.__name__ == 'EMProject'\n")
    assert result.returncode == 0, result.stderr[-2000:]


def test_importing_leaves_the_host_stdout_alone():
    result = _python(
        "import sys\n"
        "before = (sys.stdout.encoding, sys.stdout.errors)\n"
        "import cavsim3d.core.em_project\n"
        "assert (sys.stdout.encoding, sys.stdout.errors) == before, before\n")
    assert result.returncode == 0, result.stderr[-2000:]


def test_messages_go_to_the_current_stdout_once_and_without_colour(capsys):
    assert logging.getLogger("cavsim3d").propagate is False
    pr.milestone("hello from cavsim3d")
    out = capsys.readouterr().out
    assert out.count("hello from cavsim3d") == 1
    assert "\x1b[" not in out                   # captured output is not a terminal


def test_colour_follows_force_color_and_no_color(capsys, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    pr.milestone("coloured")
    assert "\x1b[" in capsys.readouterr().out
    monkeypatch.setenv("NO_COLOR", "1")
    pr.milestone("plain")
    assert "\x1b[" not in capsys.readouterr().out


def test_output_never_fails_on_characters_the_stream_cannot_encode(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    pr.echo("σ ─ ✅")
    pr.warning("⚠ done")
    stream.flush()
    text = raw.getvalue().decode("cp1252")
    assert "?" in text and "done" in text


def test_verbose_applies_to_one_solve_only(tmp_path):
    from cavsim3d.core.em_project import EMProject
    from cavsim3d.geometry.primitives import RectangularWaveguide
    p = EMProject("verb", base_dir=tmp_path, overwrite=True)
    p.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    pr.set_verbosity(True)
    try:
        p.fds.solve(fmin=1.8, fmax=2.4, nsamples=3, order=1, verbose=False)
        assert pr.console_handler.level == pr.VERBOSE     # restored after the solve
        p.fds.solve(fmin=1.8, fmax=2.4, nsamples=3, order=1, rerun=True)
        assert pr.console_handler.level == pr.VERBOSE     # not given: left as set
    finally:
        pr.set_verbosity(False)


def test_a_solve_prints_no_ngsolve_dof_warnings(tmp_path, capfd):
    from cavsim3d.core.em_project import EMProject
    from cavsim3d.geometry.primitives import RectangularWaveguide
    p = EMProject("quiet", base_dir=tmp_path, overwrite=True)
    p.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    p.fds.solve(fmin=1.8, fmax=2.4, nsamples=3, order=1)
    captured = capfd.readouterr()
    assert "used dof inconsistency" not in captured.out + captured.err
