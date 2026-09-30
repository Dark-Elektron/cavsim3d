"""Questions to the user: without an answer, nothing is changed."""
import shutil
from pathlib import Path

import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.utils import io_utils

STEP = (Path(__file__).resolve().parents[1] / "docs" / "example_models"
        / "rectangular_waveguide.step")


@pytest.fixture
def interactive(monkeypatch):
    monkeypatch.setattr(io_utils, "is_interactive", lambda: True)


def _no_frontend(prompt=""):
    from IPython.core.error import StdinNotImplementedError
    raise StdinNotImplementedError("raw_input was called, but this frontend "
                                   "does not support input requests.")


@pytest.mark.parametrize("failure", [_no_frontend, EOFError, OSError])
def test_a_question_nobody_can_answer_is_declined(interactive, monkeypatch, failure):
    def fail(prompt=""):
        if isinstance(failure, type):
            raise failure()
        failure(prompt)
    monkeypatch.setattr("builtins.input", fail)
    assert io_utils.get_user_confirmation("Delete all results?", default=True) is False
    assert io_utils.ask_choice("Choice", ("1", "2"), default="2") is None


def test_answers_are_read_when_someone_is_there(interactive, monkeypatch):
    answers = iter(["", "n", "maybe", "3", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert io_utils.get_user_confirmation("Go on?", default=True) is True
    assert io_utils.get_user_confirmation("Go on?", default=True) is False
    assert io_utils.ask_choice("Choice", ("1", "2"), default="2") == "1"


def _project_with_linked_step(tmp_path):
    source = tmp_path / "cad" / "guide.step"
    source.parent.mkdir()
    shutil.copy(STEP, source)
    p = EMProject("linked", base_dir=tmp_path / "projects", overwrite=True)
    p.import_geometry(source, name="guide", unit="m")
    p.fds.solve(fmin=1.8, fmax=2.4, nsamples=3, order=1)
    return source, tmp_path / "projects" / "linked"


def _change(source: Path):
    source.write_bytes(source.read_bytes() + b"\n/* edited */\n")


def test_a_changed_cad_file_is_left_alone_without_an_answer(tmp_path):
    source, project = _project_with_linked_step(tmp_path)
    _change(source)
    history = (project / "geometry" / "history.json").read_text()
    reopened = EMProject("linked", base_dir=project.parent)
    assert reopened.has_results()                    # nothing deleted
    assert (project / "geometry" / "history.json").read_text() == history


def test_reading_another_project_never_asks(tmp_path, interactive, monkeypatch):
    source, project = _project_with_linked_step(tmp_path)
    _change(source)
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": pytest.fail("asked a question"))
    from cavsim3d.geometry.base import BaseGeometry
    BaseGeometry.load_geometry(project, check_source=False)
    EMProject("linked", base_dir=project.parent, _read_only=True, _announce=False)


def test_updating_to_a_changed_cad_file_rebuilds_from_it(tmp_path, interactive, monkeypatch):
    source, project = _project_with_linked_step(tmp_path)
    _change(source)
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")     # [1] Update
    reopened = EMProject("linked", base_dir=project.parent)
    geo = reopened.geometry
    assert geo._source_hash == geo._file_hash(project / "geometry" / "source_model.step")
    assert geo._source_hash == geo._file_hash(source)
    assert not (project / "mesh" / "mesh.pkl").exists()     # mesh of the old file
    assert not reopened.has_results()
