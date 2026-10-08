"""A reduced or joined sweep saves its own results, not the whole project.

The full-order results and the reduced (or coupled) matrices change only with
solve() / reduce() / concatenate(); a sweep of a reduced or joined model
writes its S, Z, snapshots and timing, and leaves every other file as it was.
Reopening gives the last sweep.
"""
import numpy as np

from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.primitives import RectangularWaveguide
from cavsim3d.rom.reduction import ModelOrderReduction

CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2)


def _stamps(root, exclude):
    """{path: (mtime, size)} of every file under ``root`` but those whose path
    (relative, with '/') starts with one of ``exclude``."""
    out = {}
    for f in root.rglob("*"):
        rel = f.relative_to(root).as_posix()
        if (f.is_file() and f.suffix != ".log"              # a solve writes its log
                and not any(rel.startswith(e) for e in exclude)):
            out[rel] = (f.stat().st_mtime_ns, f.stat().st_size)
    return out


def test_reduced_sweep_writes_its_results_only(tmp_path):
    proj = EMProject("single", base_dir=str(tmp_path), overwrite=True)
    proj.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    proj.fds.solve(config=CFG)
    rom = proj.fds.fom.reduce(tol=1e-9)
    rom.solve(1.8, 2.4, 5)
    root = tmp_path / "single"
    # the results folders of the reduced model, and the timing, may change
    mine = tuple(f"fds/fom/rom/{d}" for d in ("s/", "z/", "snapshots/", "metadata.json"))
    before = _stamps(root, exclude=mine + ("timing.json",))
    rom.solve(1.9, 2.3, 7)
    after = _stamps(root, exclude=mine + ("timing.json",))
    assert after == before                       # matrices, FOM, mesh: untouched

    again = EMProject("single", base_dir=str(tmp_path))
    r2 = again.fds.fom.rom
    np.testing.assert_allclose(r2.frequencies, rom.frequencies)
    np.testing.assert_allclose(r2._S_matrix, rom._S_matrix)
    r2.solve(1.9, 2.3, 7)                        # the same request: the stored sweep
    np.testing.assert_allclose(r2._S_matrix, rom._S_matrix)


def test_reduced_model_the_project_does_not_hold_is_not_written(tmp_path):
    proj = EMProject("held", base_dir=str(tmp_path), overwrite=True)
    proj.geometry = RectangularWaveguide(a=0.1, L=0.06667, maxh=0.06)
    proj.fds.solve(config=CFG)
    rom = proj.fds.fom.reduce(tol=1e-9)
    rom.solve(1.8, 2.4, 5)
    root = tmp_path / "held"
    before = _stamps(root, exclude=("timing.json",))
    other = ModelOrderReduction(proj.fds)        # not proj.fds.fom.rom
    other.reduce(tol=1e-3)
    before = _stamps(root, exclude=("timing.json",))
    other.solve(1.8, 2.4, 9)
    assert _stamps(root, exclude=("timing.json",)) == before


def test_joined_sweep_writes_its_results_only(tmp_path):
    """A chain of a repeated part: the joined model's sweep writes its own
    results in fds/foms/roms/concat/, the coupled matrices stay."""
    proj = EMProject("chain", base_dir=str(tmp_path), overwrite=True)
    proj.create_primitive("rwg", name="g", a=0.1, b=0.05, L=0.05, maxh=0.04, n=2)
    proj.fds.solve(config=CFG)
    concat = proj.fds.foms.reduce(tol=1e-9).concatenate()
    concat.solve(1.8, 2.4, 5)
    root = tmp_path / "chain"
    keep = tuple(f"fds/foms/roms/concat/{d}/" for d in ("s", "z", "snapshots")) + ("timing.json",)
    before = _stamps(root, exclude=keep)
    concat.solve(1.9, 2.3, 7)
    assert _stamps(root, exclude=keep) == before
    again = EMProject("chain", base_dir=str(tmp_path))
    c2 = again.fds.foms.roms.concat
    np.testing.assert_allclose(c2.frequencies, concat.frequencies)
    np.testing.assert_allclose(c2._S_matrix, concat._S_matrix)


def test_glued_joined_sweep_is_saved(tmp_path):
    """Glued parts reduced per domain: the joined model's sweep is saved with
    the reduced models (fds/foms/roms/concat/) and read back on reopening."""
    proj = EMProject("glued", base_dir=str(tmp_path), overwrite=True)
    for name in ("a", "b"):
        proj.create_primitive("rwg", name=name, a=0.1, b=0.05, L=0.05, maxh=0.04)
    proj.generate_mesh(maxh=0.04)
    proj.fds.solve(config=dict(CFG, per_domain=True, global_method=None))
    concat = proj.fds.foms.reduce(tol=1e-9).concatenate()
    concat.solve(1.9, 2.3, 7)
    assert (tmp_path / "glued" / "fds" / "foms" / "roms" / "concat" / "z" / "z.h5").exists()
    again = EMProject("glued", base_dir=str(tmp_path))
    c2 = again.fds.foms.roms.concat
    np.testing.assert_allclose(c2.frequencies, concat.frequencies)
    np.testing.assert_allclose(c2._S_matrix, concat._S_matrix)
