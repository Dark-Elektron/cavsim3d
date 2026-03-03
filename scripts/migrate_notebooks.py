"""
Migrate cavsim3d example notebooks to use the new result-object API.

For each notebook this script:
  1. Finds the cell that creates the ROM and the comparison plotting cells
  2. Adds NEW cells right after them that demonstrate the new API
  3. Replaces `rom._concatenated` internal access with `fds.foms.roms.concat`
  4. Saves the notebook in-place (a .bak copy is created first)

Usage:
    python migrate_notebooks.py
"""

import json
import shutil
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / 'examples'


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def load_notebook(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(path: Path, nb: dict):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def make_markdown_cell(source: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source.splitlines(keepends=True)
    }


def make_code_cell(source: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source.splitlines(keepends=True)
    }


def find_cell_index(cells, substring):
    """Return index of first code cell containing substring, or -1."""
    for i, c in enumerate(cells):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            if substring in src:
                return i
    return -1


def replace_in_cells(cells, old, new):
    """Replace text in all code cells."""
    count = 0
    for c in cells:
        if c['cell_type'] == 'code':
            new_src = []
            for line in c['source']:
                if old in line:
                    line = line.replace(old, new)
                    count += 1
                new_src.append(line)
            c['source'] = new_src
    return count


# -------------------------------------------------------------------------
# New API cell templates
# -------------------------------------------------------------------------

NEW_API_HEADER = """\
## New Result-Object API

The new API provides a navigable object graph:
- `fds.fom` — global FOM result
- `fds.fom.rom` — ROM of the global FOM
- `fds.foms` — per-domain FOMs (multi-solid only)
- `fds.foms.roms.concat.rom` — full ROM concatenation chain

Each result object has `.plot_s()`, `.plot_z()`, `.plot_eigenvalues()` methods.\
"""

SINGLE_SOLID_CELLS = """\
# === New API: result-object navigation ===

# FOM result — same data as fds, but as a standalone result object
fom_result = fds.fom
print(fom_result)

# Plot S-parameters directly from the FOM result
fig, ax = fom_result.plot_s(params=['1(1)1(1)', '1(1)2(1)'], label='FOM')

# ROM result — navigate from FOM
rom_result = fds.fom.rom
print(rom_result)

# Overlay ROM onto the same axes
fig, ax = rom_result.plot_s(params=['1(1)1(1)', '1(1)2(1)'], ax=ax, label='ROM')
plt.show()\
"""

MULTI_SOLID_CELLS = """\
# === New API: multi-solid result-object navigation ===

# Global FOM (coupled solve over entire mesh)
print("Global FOM:", fds.fom)

# Per-domain FOMs
print("Per-domain FOMs:", fds.foms)
for i, fom_i in enumerate(fds.foms):
    print(f"  Domain {i}: {fom_i}")

# Plot per-domain S-parameters overlaid
fig, ax = fds.foms.plot_s(params=['1(1)1(1)'], title='Per-domain S11')
plt.show()

# Full chain: per-domain → ROM each → concatenate → further reduce
# This is the 1→2→3→4→5 path from the architecture diagram
concat_rom = fds.foms.roms.concat.rom
print("Concatenated ROM:", concat_rom)
fig, ax = concat_rom.plot_s(params=['1(1)1(1)'], label='Concat ROM')
plt.show()\
"""

CONCAT_INTERNALS_REPLACEMENT = """\
# === Concatenation info (new API) ===
concat_result = fds.foms.roms.concat
print(f"Concat result: {concat_result}")
print(f"N ports: {concat_result.n_ports}")
print(f"Ports: {concat_result.ports}")
fig, ax = concat_result.plot_z(params=['1(1)1(1)'], label='Concat FOM')
plt.show()\
"""


# -------------------------------------------------------------------------
# Per-notebook migration logic
# -------------------------------------------------------------------------

def is_multi_solid(nb_path: Path) -> bool:
    """Heuristic: notebooks in *_split directories are multi-solid."""
    return 'split' in nb_path.parent.name


def migrate_notebook(nb_path: Path):
    print(f"\nMigrating: {nb_path.relative_to(EXAMPLES_DIR)}")
    nb = load_notebook(nb_path)
    cells = nb['cells']
    multi = is_multi_solid(nb_path)
    inserted = 0

    # 1. Replace rom._concatenated internal access
    n_replaced = 0
    for c in cells:
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            if 'rom._concatenated' in src:
                # Replace the entire cell with the new API equivalent
                c['source'] = CONCAT_INTERNALS_REPLACEMENT.splitlines(keepends=True)
                # Ensure last line has newline
                if c['source'] and not c['source'][-1].endswith('\n'):
                    c['source'][-1] += '\n'
                n_replaced += 1
    if n_replaced:
        print(f"  Replaced {n_replaced} cell(s) accessing rom._concatenated")

    # 2. Find the last comparison-plot cell and insert new API cells after it
    insert_after = -1
    for i in range(len(cells) - 1, -1, -1):
        if cells[i]['cell_type'] == 'code':
            src = ''.join(cells[i]['source'])
            if 'plot_s_comparison' in src or 'plot_z_comparison' in src:
                insert_after = i
                break

    # If no comparison plot found, look for rom.solve
    if insert_after < 0:
        insert_after = find_cell_index(cells, 'rom.solve')

    if insert_after >= 0:
        new_cells = [
            make_markdown_cell(NEW_API_HEADER),
            make_code_cell(MULTI_SOLID_CELLS if multi else SINGLE_SOLID_CELLS),
        ]
        for j, nc in enumerate(new_cells):
            cells.insert(insert_after + 1 + j, nc)
            inserted += 1

    nb['cells'] = cells

    # 3. Save with backup
    backup = nb_path.with_suffix('.ipynb.bak')
    shutil.copy2(nb_path, backup)
    save_notebook(nb_path, nb)
    print(f"  Inserted {inserted} new cell(s), backup at {backup.name}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    notebooks = sorted(EXAMPLES_DIR.rglob('*.ipynb'))
    print(f"Found {len(notebooks)} notebook(s) to migrate:")
    for nb in notebooks:
        print(f"  {nb.relative_to(EXAMPLES_DIR)}")

    for nb in notebooks:
        migrate_notebook(nb)

    print(f"\nDone! Migrated {len(notebooks)} notebooks.")
    print("Open each notebook and re-run to verify the new API cells work.")


if __name__ == '__main__':
    main()
