"""
Nedelec element kind (solve setting ``nedelec='second' | 'first'``).

Validates:
  - 'first' uses NGSolve's type-1 space (fewer unknowns, same curls) and gives
    the same Z as 'second' to discretisation accuracy
  - the kind is saved: a reopened project rebuilds type-1 spaces and port modes
  - changing the kind recomputes the sweep (rerun policy)
  - reduction and a repeated-part netlist (concatenation) run with first kind
"""

import numpy as np
import pytest

from cavsim3d.core.em_project import EMProject
from cavsim3d.solvers.nedelec import check_kind, hcurl_flags, kind_of

CFG = dict(fmin=1.8, fmax=2.4, nsamples=4, nportmodes=1, order=2, solver_type="direct")
GUIDE = dict(a=0.1, L=0.2, b=0.05, maxh=0.04)


def _proj(tmp_path, name, **cfg):
    p = EMProject(name=name, base_dir=str(tmp_path), overwrite=True)
    p.create_primitive("rwg", name="guide", **GUIDE)
    p.fds.solve(config=dict(CFG, **cfg))
    return p


def test_helpers():
    assert hcurl_flags("second") == {} and hcurl_flags("first") == {"type1": True}
    with pytest.raises(ValueError):
        check_kind("third")


def test_first_kind_fewer_unknowns_and_accurate(tmp_path):
    from cavsim3d.analytical.rectangular_waveguide import RWGAnalytical
    ana = RWGAnalytical(a=0.1, b=0.05, L=0.2)
    band = dict(fmin=1.1 * ana.fc, fmax=2.0 * ana.fc)
    err, ndof = {}, {}
    for kind in ("second", "first"):
        p = EMProject(name=kind, base_dir=str(tmp_path), overwrite=True)
        p.create_primitive("rwg", name="guide", **dict(GUIDE, maxh=0.02))
        res = p.fds.solve(config=dict(CFG, **band, nsamples=6, nedelec=kind))
        fes = p.fds._fes_global
        assert kind_of(fes) == kind
        assert all(kind_of(gf.space) == kind
                   for modes in p.fds.port_modes.values() for gf in modes.values())
        Za = ana.z_parameters(p.fds.frequencies / 1e9)
        Zd = res["Z_dict"]
        err[kind] = max(np.max(np.abs(Zd["1(1)1(1)"] - Za["Z11"]) / np.abs(Za["Z11"])),
                        np.max(np.abs(Zd["1(1)2(1)"] - Za["Z21"]) / np.abs(Za["Z21"])))
        ndof[kind] = fes.ndof
    assert ndof["first"] < 0.75 * ndof["second"]
    assert err["first"] < 5e-2                       # measured 2.7e-2 (second kind: 1.5e-1)


def test_kind_is_saved(tmp_path, monkeypatch):
    from cavsim3d.solvers import sweep_checkpoint as sc
    _proj(tmp_path, "saved", nedelec="first")
    q = EMProject(name="saved", base_dir=str(tmp_path))
    assert q.fds.nedelec == "first"
    assert kind_of(q.fds._fes_global) == "first"
    assert all(kind_of(gf.space) == "first"
               for modes in q.fds.port_modes.values() for gf in modes.values())
    # same request: the stored results are reused, no sample is computed
    computed = []
    real = sc.SweepCheckpoint.write
    monkeypatch.setattr(sc.SweepCheckpoint, "write",
                        lambda self, *a, **k: (computed.append(1), real(self, *a, **k)))
    q.fds.solve(config=dict(CFG, nedelec="first"))
    assert not computed


def test_project_saved_before_the_setting_stays_second(tmp_path, monkeypatch):
    import json
    import pickle
    from cavsim3d.solvers import sweep_checkpoint as sc
    _proj(tmp_path, "legacy", nedelec="second")
    fds_dir = tmp_path / "legacy" / "fds"
    cfg = json.loads((fds_dir / "config.json").read_text())
    del cfg["nedelec"]                                   # the old file format
    (fds_dir / "config.json").write_text(json.dumps(cfg))
    pm = fds_dir / "port_modes" / "port_modes.pkl"
    data = pickle.loads(pm.read_bytes())
    data.pop("nedelec")
    pm.write_bytes(pickle.dumps(data))

    q = EMProject(name="legacy", base_dir=str(tmp_path))
    assert q.fds.nedelec == "second" and kind_of(q.fds._fes_global) == "second"
    computed = []
    real = sc.SweepCheckpoint.write
    monkeypatch.setattr(sc.SweepCheckpoint, "write",
                        lambda self, *a, **k: (computed.append(1), real(self, *a, **k)))
    q.fds.solve(config=CFG)                              # no setting: keeps its kind
    assert not computed


def test_default_is_first_and_changing_recomputes(tmp_path):
    p = _proj(tmp_path, "change")
    assert p.fds.nedelec == "first"
    n1 = p.fds._fes_global.ndof
    p.fds.solve(config=dict(CFG, nedelec="second"))
    assert p.fds.nedelec == "second" and p.fds._fes_global.ndof > n1
    assert p.fds.snapshots["global"].shape[0] == p.fds._fes_global.ndof


def test_rom_and_netlist_first_kind(tmp_path):
    p = _proj(tmp_path, "rom", nedelec="first")
    rom = p.fds.fom.reduce(tol=1e-9)
    res = rom.solve(fmin=1.8, fmax=2.4, nsamples=7)
    assert res["Z"].shape[0] == 7

    q = EMProject(name="chain", base_dir=str(tmp_path), overwrite=True)
    q.create_primitive("rwg", name="cell", n=2, **GUIDE)
    q.fds.solve(config=dict(CFG, nedelec="first"))
    concat = q.fds.foms.reduce(tol=1e-9).concatenate()
    S = concat.solve(config=dict(fmin=1.8, fmax=2.4, nsamples=7))["S"]
    assert np.all(np.abs(S[:, 1, 0]) > 0.9)          # matched guide: transmits
