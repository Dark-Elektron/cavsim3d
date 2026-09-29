"""Load-if-exists / run-if-not resolution of a section's reduced model.

A "section" (a single solid, a multi-solid model, or a sub-assembly) is turned
into reduced structures (ready to concatenate) by either:

  * importing a previously-run project's ROM from disk (no recompute), or
  * running the FOM + ROM once and saving it, so the next call reuses it.

These are standalone portability helpers ("IKEA screws"): they let a saved
reduced model be loaded — or produced once — without the original solver.
The netlist pipeline itself stages artifacts by copy (see
``cavsim3d.solvers.netlist_persistence``); it does not go through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def load_or_run_reduced(
    project_path,
    geometry=None,
    fom_config: Optional[dict] = None,
    rom_tol: float = 1e-9,
    order: int = 3,
    force: bool = False,
) -> Tuple[list, Optional[object]]:
    """Return ``(structures, impedance_func)`` for a section, reusing on disk.

    Parameters
    ----------
    project_path : path-like
        Where the section's project lives (or should live).  If it already
        holds a reduced model it is imported; otherwise it is created/run here.
    geometry : BaseGeometry, optional
        The section geometry — required only when the ROM must be computed
        (i.e. no saved result and ``force=False``, or ``force=True``).
    fom_config : dict, optional
        Config for the full-order solve (``fmin``/``fmax``/``nsamples``/
        ``nportmodes``/``solver_type`` ...).
    rom_tol : float
        ROM truncation tolerance.
    order : int
        H(curl) order used when the FOM has to be run.
    force : bool
        Recompute even if a saved ROM exists.

    Returns
    -------
    (structures, impedance_func)
        As from :func:`cavsim3d.rom.reduction.import_reduced_structures`.
    """
    from cavsim3d.rom.reduction import import_reduced_structures

    project_path = Path(project_path)

    if not force:
        try:
            return import_reduced_structures(project_path)
        except FileNotFoundError:
            pass  # fall through and compute

    if geometry is None:
        raise ValueError(
            f"No saved reduced model at {project_path} and no geometry given "
            "to run one. Pass geometry= to compute it."
        )

    from cavsim3d.core.em_project import EMProject

    proj = EMProject(name=project_path.name,
                     base_dir=str(project_path.parent),
                     overwrite=force)
    proj.order = order
    proj.geometry = geometry
    cfg = dict(fom_config or {})
    cfg.setdefault("nportmodes", 1)
    proj.fds.solve(config=cfg)

    # Reduce through the standard fluent path (single- vs multi-solid),
    # which persists the ROM (+ standalone structure metadata) to disk.
    if getattr(proj.fds, "is_compound", False):
        proj.fds.foms.reduce(tol=rom_tol)
    else:
        proj.fds.fom.reduce(tol=rom_tol)

    return import_reduced_structures(project_path)


class ImportedModel:
    """Another project used as a part of this one.

    Created with ``proj.import_project(path, mode=...)`` (the method cannot be
    called ``import``: that is a reserved Python keyword) and added to the
    project's parts like any geometry::

        cavity = proj.import_project("path/to/cavity_project", n=2)

    ``mode='reference'`` reads the source project's saved results where they
    are; ``mode='copy'`` copies them into this project, so it stands alone.
    The source project is never written to: anything it lacks (a reduced
    model, a wider band, more port modes) is computed into this project.

    The source needs at least one of: a reduced model, full-order results, or
    a geometry to compute them from.
    """

    # BaseGeometry duck-typing stubs so project bookkeeping treats the handle
    # as an inert component (nothing to build, mesh, or replay).
    geo = None
    mesh = None
    MODES = ('reference', 'copy')

    def __init__(self, project_path, mode: str = 'copy'):
        from cavsim3d.solvers import netlist_persistence as npz

        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise FileNotFoundError(f"Project folder not found: {self.project_path}")
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

        try:
            self.rom_dir = npz.find_rom_dir(self.project_path)
        except FileNotFoundError:
            self.rom_dir = None
        try:
            self.fom_dir = npz.find_fom_dir(self.project_path)
        except FileNotFoundError:
            self.fom_dir = None
        self.has_geometry = (self.project_path / "geometry" / "history.json").exists()
        if self.rom_dir is None and self.fom_dir is None and not self.has_geometry:
            raise FileNotFoundError(
                f"Nothing to import from {self.project_path}: no reduced model, "
                "no full-order results and no geometry. Run the project first "
                "(fds.solve(), then fom/foms.reduce()), or give it a geometry.")

        self.ports, self.port_modes, self.training_band = [], {}, None
        if self.rom_dir is not None:
            import json
            with open(self.rom_dir / "structures.json") as fh:
                meta = json.load(fh)
            self.ports = [p for s in meta.get("structures", []) for p in s["ports"]]
            self.port_modes = {p: s["port_modes"][p]
                               for s in meta.get("structures", [])
                               for p in s["port_modes"]}
            self.training_band = meta.get("band") or next(
                (s.get("band") for s in meta.get("structures", []) if s.get("band")),
                None)

    @property
    def available(self) -> str:
        """What the source holds: 'rom', 'fom' or 'geometry' (most complete)."""
        if self.rom_dir is not None:
            return "rom"
        if self.fom_dir is not None:
            return "fom"
        return "geometry"

    def fingerprint(self) -> str:
        """Hash of the source's saved results (detects a re-solved source)."""
        import hashlib
        h = hashlib.sha256()
        for d in (self.rom_dir, self.fom_dir):
            if d is None:
                continue
            for f in sorted(Path(d).rglob("*")):
                if not f.is_file() or "concat" in f.relative_to(d).parts:
                    continue
                st = f.stat()
                h.update(f"{f.relative_to(d).as_posix()}|{st.st_size}".encode())
                if f.name == "structures.json" or f.name == "metadata.json" or (
                        f.suffix == ".h5" and st.st_size < 50_000_000):
                    h.update(f.read_bytes())
        return h.hexdigest()

    def get_history(self):
        # Recorded so an assembly that references this model can be rebuilt
        # when its project is reopened.
        return [{'op': 'import_model', 'project_path': str(self.project_path),
                 'mode': self.mode}]

    def __repr__(self):
        band = (f", band=[{self.training_band['fmin_GHz']:.3g}, "
                f"{self.training_band['fmax_GHz']:.3g}] GHz"
                if self.training_band else "")
        return (f"ImportedModel('{self.project_path.name}', mode='{self.mode}', "
                f"has={self.available}, ports={self.ports}{band})")
