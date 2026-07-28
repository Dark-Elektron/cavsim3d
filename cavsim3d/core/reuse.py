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
    """Handle to an ALREADY-RUN project's saved results (a portable part).

    Created via ``proj.fds.import_model(path)`` (named ``import_model`` because
    ``import`` is a reserved Python keyword).  The handle validates at import
    time that the project actually holds a saved reduced model, and can be
    added to an :class:`~cavsim3d.geometry.assembly.Assembly` netlist exactly
    like a geometry::

        hom = proj.fds.import_model("path/to/hom_coupler_project")
        asm.add("hom", hom, after="cavity")

    The saved FOM/ROM is then LOADED (never recomputed) when the assembly's
    ROMs are concatenated.
    """

    # BaseGeometry duck-typing stubs so project bookkeeping treats the handle
    # as an inert component (nothing to build, mesh, or replay).
    geo = None
    mesh = None

    def __init__(self, project_path):
        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise FileNotFoundError(f"Project folder not found: {self.project_path}")

        # Fail fast: locate the saved reduced-model metadata.
        candidates = [
            self.project_path,
            self.project_path / "fds" / "foms" / "roms",
            self.project_path / "fds" / "fom" / "rom",
            self.project_path / "foms" / "roms",
        ]
        rom_dir = next((d for d in candidates
                        if (d / "structures.json").exists()), None)
        if rom_dir is None:
            hits = sorted(self.project_path.rglob("structures.json"))
            rom_dir = hits[0].parent if hits else None
        if rom_dir is None:
            raise FileNotFoundError(
                f"No saved reduced model found under {self.project_path}. "
                "Run the project first (fds.solve() then fom/foms.reduce()) "
                "so its results can be imported."
            )
        self.rom_dir = rom_dir

        import json
        with open(rom_dir / "structures.json") as fh:
            meta = json.load(fh)
        self.ports = [p for s in meta.get("structures", []) for p in s["ports"]]
        self.port_modes = {p: s["port_modes"][p]
                           for s in meta.get("structures", [])
                           for p in s["port_modes"]}
        self.training_band = meta.get("band")

    def get_history(self):
        return []

    def __repr__(self):
        band = (f", band=[{self.training_band['fmin_GHz']:.3g}, "
                f"{self.training_band['fmax_GHz']:.3g}] GHz"
                if self.training_band else "")
        return (f"ImportedModel('{self.project_path.name}', "
                f"ports={self.ports}{band})")
