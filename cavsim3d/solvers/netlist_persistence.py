"""Stage a single-section project's artifacts into a parent netlist project.

A netlist project has ONE ``fds`` and every unique section is a domain inside
that single flat tree.  Staging is a folder-to-matching-folder copy, with each
file **renamed to the section's index** in the current project — so there is
NO distinction on disk between a computed section and an imported one:

    already_solved/fds/fom/matrices/K.h5        -> fds/foms/matrices/K_<sec>.h5
    already_solved/fds/fom/s/s_global.h5        -> fds/foms/s/s_<sec>.h5
    already_solved/fds/fom/rom/matrices/A_r.h5  -> fds/foms/roms/matrices/A_r_<sec>.h5
    already_solved/mesh/mesh.pkl                -> mesh/mesh_<sec>.pkl
    already_solved/geometry/*.step              -> geometry/components/<sec>.step

Canonical folder contents (see CLAUDE.md): ``fom(s)``/``rom(s)``/``concat``
hold ONLY ``matrices, eigenmodes, s, z, snapshots`` (+ their nested stage
folders); one ``mesh/`` and one ``geometry/`` folder per project, at the top
level next to ``fds/``.  Live sections are computed once in a throwaway
scratch project and staged through the exact same copy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

FOM_MATS = ("K", "M", "B")
ROM_MATS = ("A_r", "B_r", "W", "Q_L_inv")
RESULT_DIRS = ("s", "z", "eigenmodes", "snapshots")


# --------------------------------------------------------------------------- #
# Locating the pieces inside a source project
# --------------------------------------------------------------------------- #
def find_fom_dir(project: Path) -> Path:
    """Locate a source project's FOM directory (single-section)."""
    project = Path(project)
    for c in (project / "fds" / "fom", project / "fds" / "foms",
              project / "fom", project / "foms", project):
        if (c / "metadata.json").exists() and (c / "matrices").exists():
            return c
    hits = sorted(project.rglob("metadata.json"))
    for h in hits:
        if (h.parent / "matrices").exists() and "rom" not in h.parent.parts:
            return h.parent
    raise FileNotFoundError(f"No FOM directory found under {project}")


def find_rom_dir(project: Path) -> Path:
    """Locate a source project's ROM directory (has structures.json)."""
    project = Path(project)
    for c in (project / "fds" / "fom" / "rom", project / "fds" / "foms" / "roms",
              project / "fom" / "rom", project / "foms" / "roms",
              project / "roms", project / "rom", project):
        if (c / "structures.json").exists():
            return c
    hits = sorted(project.rglob("structures.json"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(
        f"No reduced model (structures.json) found under {project}. "
        "Reduce the section in its own project first.")


def _pick(folder: Path, base: str, src_domain: str) -> Optional[Path]:
    """Pick ``base_<src_domain>.h5`` or ``base.h5`` if present."""
    for name in (f"{base}_{src_domain}.h5", f"{base}.h5"):
        f = folder / name
        if f.exists():
            return f
    return None


def _copy_result_dir(src: Path, dst: Path, src_domain: str, domain: str) -> None:
    """Copy s/z/eigenmodes/snapshots files, renaming the domain suffix."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        stem = f.stem
        # rename trailing _<src_domain> -> _<domain>; else append _<domain>
        if stem.endswith(f"_{src_domain}"):
            new = stem[: -len(src_domain) - 1] + f"_{domain}"
        elif "_" not in stem:
            new = f"{stem}_{domain}"
        else:
            new = stem
        shutil.copy2(f, dst / (new + f.suffix))


def _stage_mesh(source_project: Path, domain: str, project_root: Path) -> None:
    """already_solved/mesh/* -> <project>/mesh/<stem>_<domain><suffix>."""
    src = Path(source_project) / "mesh"
    if not src.exists():
        return
    dst = Path(project_root) / "mesh"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f"{f.stem}_{domain}{f.suffix}")


def _stage_geometry(source_project: Path, domain: str, project_root: Path) -> None:
    """already_solved/geometry/*.step -> <project>/geometry/components/<domain>.step."""
    src = Path(source_project) / "geometry"
    if not src.exists():
        return
    steps = sorted(src.rglob("*.step")) + sorted(src.rglob("*.stp"))
    if not steps:
        return
    dst = Path(project_root) / "geometry" / "components"
    dst.mkdir(parents=True, exist_ok=True)
    if len(steps) == 1:
        shutil.copy2(steps[0], dst / f"{domain}{steps[0].suffix}")
    else:
        for i, f in enumerate(steps, 1):
            shutil.copy2(f, dst / f"{domain}_{i}{f.suffix}")


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #
def stage_fom(source_project: Path, domain: str, project_root: Path) -> None:
    """Copy a section's FOM artifacts into the parent project, renamed to
    ``domain``: matrices/results into the flat ``fds/foms`` tree, mesh files
    into the project's single ``mesh/``, geometry into ``geometry/components/``.
    """
    source_project = Path(source_project)
    project_root = Path(project_root)
    foms_dir = project_root / "fds" / "foms"
    fom = find_fom_dir(source_project)
    src_domain = _source_domain(fom)

    (foms_dir / "matrices").mkdir(parents=True, exist_ok=True)
    for base in FOM_MATS:
        f = _pick(fom / "matrices", base, src_domain)
        if f is not None:
            shutil.copy2(f, foms_dir / "matrices" / f"{base}_{domain}.h5")
    for rd in RESULT_DIRS:
        _copy_result_dir(fom / rd, foms_dir / rd, src_domain, domain)

    _stage_mesh(source_project, domain, project_root)
    _stage_geometry(source_project, domain, project_root)


def stage_rom(source_project: Path, domain: str, project_root: Path) -> dict:
    """Copy a section's ROM matrices/results into ``fds/foms/roms`` (renamed to
    ``domain``) and return its per-structure metadata entry (fingerprints, band
    and impedance folded IN, so the flat merge keeps sections distinct)."""
    source_project = Path(source_project)
    roms_dir = Path(project_root) / "fds" / "foms" / "roms"
    rom = find_rom_dir(source_project)
    with open(rom / "structures.json") as fh:
        meta = json.load(fh)
    if not meta.get("structures"):
        raise ValueError(f"Empty structures.json in {rom}")
    sm = dict(meta["structures"][0])          # single-section
    src_domain = sm["domain"]

    (roms_dir / "matrices").mkdir(parents=True, exist_ok=True)
    for base in ROM_MATS:
        f = _pick(rom / "matrices", base, src_domain)
        if f is not None:
            shutil.copy2(f, roms_dir / "matrices" / f"{base}_{domain}.h5")
    for rd in RESULT_DIRS:
        _copy_result_dir(rom / rd, roms_dir / rd, src_domain, domain)

    # Fold shared metadata INTO the structure entry, rekeyed to this domain.
    sm["domain"] = domain
    if "fingerprints" not in sm:
        sm["fingerprints"] = meta.get("fingerprints", {})
    if "band" not in sm:
        sm["band"] = meta.get("band")
    if "impedance" not in sm:
        sm["impedance"] = meta.get("impedance")
    return sm


def write_flat_structures(project_root: Path, entries: list) -> None:
    """Write the merged ``fds/foms/roms/structures.json`` for all sections."""
    roms_dir = Path(project_root) / "fds" / "foms" / "roms"
    roms_dir.mkdir(parents=True, exist_ok=True)
    with open(roms_dir / "structures.json", "w") as fh:
        json.dump({"structures": entries}, fh, indent=2)


def _source_domain(fom_dir: Path) -> str:
    """Domain suffix used inside a source FOM dir (e.g. 'global')."""
    meta = fom_dir / "metadata.json"
    if meta.exists():
        try:
            with open(meta) as fh:
                d = json.load(fh)
            dom = d.get("domain")
            if dom:
                return dom
        except Exception:
            pass
    # Infer from a matrices filename: K_<domain>.h5
    for f in (fom_dir / "matrices").glob("K_*.h5"):
        return f.stem[2:]
    return "global"
