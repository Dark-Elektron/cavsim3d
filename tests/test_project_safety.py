"""Project folders: names, overwrite, and reopening after partial deletion.

    EMProject(name, base_dir)                 # base_dir / name is the project
    EMProject(name, base_dir, overwrite=True) # replaces a PROJECT, nothing else
"""
import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.primitives import RectangularWaveguide

CFG = dict(fmin=1.8, fmax=2.4, nsamples=3, nportmodes=1, order=1)


def _guide():
    return RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)


def _files(folder):
    return {f: f.stat().st_mtime_ns for f in folder.rglob("*") if f.is_file()}


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "a/b", "a\\b", "C:/Windows",
                                  "bad:name", "why?", "trailing.", " padded"])
def test_a_name_that_is_not_one_folder_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="(?i)project name"):
        EMProject(name, base_dir=tmp_path, overwrite=True)


def test_overwrite_never_deletes_a_folder_that_is_not_a_project(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "results.csv").write_text("precious")
    with pytest.raises(FileExistsError, match="not a cavsim3d project"):
        EMProject("data", base_dir=tmp_path, overwrite=True)
    assert (folder / "results.csv").read_text() == "precious"


def test_overwrite_replaces_a_project(tmp_path):
    p = EMProject("proj", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    assert (tmp_path / "proj" / "project.json").exists()
    q = EMProject("proj", base_dir=tmp_path, overwrite=True)
    assert q.geometry is None
    # a project that was only created (no project.json yet) is a project too
    EMProject("fresh", base_dir=tmp_path)
    EMProject("fresh", base_dir=tmp_path, overwrite=True)


def test_a_file_in_place_of_the_project_folder_is_refused(tmp_path):
    (tmp_path / "taken").write_text("not a folder")
    with pytest.raises(FileExistsError, match="is a file"):
        EMProject("taken", base_dir=tmp_path)


@pytest.mark.parametrize("invalidate", ["invalidate_results", "invalidate_mesh"])
def test_project_reopens_after_invalidation(tmp_path, invalidate):
    p = EMProject("inv", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    p.fds.solve(config=CFG)
    getattr(p, invalidate)()
    q = EMProject("inv", base_dir=tmp_path)
    assert not q.has_results()
    q.fds.solve(config=CFG)
    assert q.has_results()


def test_a_project_saved_before_its_first_solve_reopens_and_solves(tmp_path):
    p = EMProject("early", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    p.fds.print_info()        # the solver exists, its port modes are not solved yet
    p.save()
    q = EMProject("early", base_dir=tmp_path)
    q.fds.solve(config=CFG)
    assert q.fds.fom._S_matrix.shape == (3, 2, 2)


@pytest.mark.parametrize("damage", ["missing", "garbled"])
def test_project_opens_when_its_solver_state_is_unreadable(tmp_path, damage):
    p = EMProject("broken", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    p.fds.solve(config=CFG)
    config = tmp_path / "broken" / "fds" / "config.json"
    if damage == "missing":
        config.unlink()
    else:
        config.write_text("{ not json")
    q = EMProject("broken", base_dir=tmp_path)
    assert q.geometry is not None
    q.fds.solve(config=CFG)
    assert q.fds.fom.S_dict is not None


def test_load_opens_existing_projects_only(tmp_path):
    EMProject("there", base_dir=tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        EMProject.load("there", base_dir=tmp_path, overwrite=True)
    assert (tmp_path / "there").is_dir()
    assert EMProject.load("there", base_dir=tmp_path).name == "there"
    with pytest.raises(FileNotFoundError):
        EMProject.load("absent", base_dir=tmp_path)
    assert not (tmp_path / "absent").exists()


def test_n_port_modes_is_deprecated(tmp_path):
    p = EMProject("npm", base_dir=tmp_path)
    with pytest.warns(DeprecationWarning, match="nportmodes"):
        p.n_port_modes = 3


def test_a_project_opened_read_only_is_never_written(tmp_path):
    p = EMProject("src", base_dir=tmp_path, overwrite=True)
    p.geometry = _guide()
    p.fds.solve(config=CFG)
    before = _files(tmp_path / "src")
    q = EMProject("src", base_dir=tmp_path, _read_only=True, _announce=False)
    q.save()
    q.geometry = _guide()                     # the setter saves: not here
    assert _files(tmp_path / "src") == before
    with pytest.raises(ValueError, match="read-only"):
        EMProject("src", base_dir=tmp_path, overwrite=True, _read_only=True)
