"""
Export NGSolve WebGUI scenes from notebooks as standalone HTML files.

This script processes tutorial notebooks, identifies cells that produce
WebGUI widget outputs, re-executes them to generate standalone HTML files,
and patches the notebook outputs with an <iframe> fallback so they render
in static documentation (MkDocs).

Usage:
    python scripts/export_webgui.py [notebook_path ...]

If no paths are given, processes all notebooks in docs/tutorials/.
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_webgui_cells(nb_data):
    """Find cells whose outputs contain WebGUI widget data."""
    webgui_cells = []
    for i, cell in enumerate(nb_data['cells']):
        if cell['cell_type'] != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.get('output_type') in ('display_data', 'execute_result'):
                data = output.get('data', {})
                if 'application/vnd.jupyter.widget-view+json' in data:
                    text = data.get('text/plain', '')
                    if isinstance(text, list):
                        text = ''.join(text)
                    if 'WebGuiWidget' in text:
                        webgui_cells.append(i)
                        break
    return webgui_cells


def export_webgui_from_notebook(nb_path):
    """Process a notebook and export WebGUI HTML files.

    Strategy:
    1. Read the notebook
    2. Find WebGUI cells
    3. Execute those cells to generate standalone HTML exports
    4. Patch the notebook with <iframe> + text/html output
    """
    nb_path = Path(nb_path)
    print(f"\nProcessing: {nb_path.name}")

    with open(nb_path) as f:
        nb = json.load(f)

    webgui_cells = find_webgui_cells(nb)
    if not webgui_cells:
        print("  No WebGUI cells found.")
        return False

    print(f"  Found {len(webgui_cells)} WebGUI cell(s): {webgui_cells}")

    # Build execution context by collecting all code cells up to and including
    # each WebGUI cell
    tutorials_dir = nb_path.parent
    modified = False

    for cell_idx in webgui_cells:
        cell = nb['cells'][cell_idx]
        source = ''.join(cell['source'])

        # Generate output filename
        html_filename = f"{nb_path.stem}_webgui_cell{cell_idx}.html"
        html_path = tutorials_dir / html_filename

        print(f"  Cell {cell_idx}: '{source.strip()[:60]}...'")
        print(f"    → Exporting to: {html_filename}")

        # Build the execution script
        # Collect all prior code cells for context
        code_lines = []
        code_lines.append("import sys, os")
        code_lines.append(f"sys.path.insert(0, {str(PROJECT_ROOT)!r})")
        code_lines.append(f"os.chdir({str(PROJECT_ROOT)!r})")

        for j in range(cell_idx + 1):
            c = nb['cells'][j]
            if c['cell_type'] == 'code':
                cell_source = ''.join(c['source'])
                # Skip magic commands
                filtered_lines = []
                for line in cell_source.split('\n'):
                    if line.strip().startswith('%') or line.strip().startswith('!'):
                        continue
                    filtered_lines.append(line)
                code_lines.append('\n'.join(filtered_lines))

        # Replace the Draw/show call with a filename export
        # Detect the call pattern
        last_code = code_lines[-1].strip()

        if '.show(' in last_code:
            # Replace geo.show('mesh') with Draw(geo.mesh, filename=...)
            # We need to extract the object and the 'what' parameter
            code_lines[-1] = f"""
# Export WebGUI to standalone HTML
import re
_src = {source.strip()!r}
# Execute the original but capture the scene
from ngsolve.webgui import Draw as _Draw
_orig_show = type(geo).show

def _patched_show(self, what='geometry', **kwargs):
    if what.lower() == 'mesh':
        scene = _Draw(self.mesh, show=False, **kwargs)
    elif what.lower() in ('geometry', 'geo'):
        scene = _Draw(self.geo, show=False, **kwargs)
    scene.GenerateHTML(filename={str(html_path)!r})
    print(f"Exported WebGUI to {html_path.name!r}")

type(geo).show = _patched_show
{source.strip()}
type(geo).show = _orig_show
"""
        elif 'Draw(' in last_code:
            # Direct Draw() call - add filename parameter
            code_lines[-1] = last_code.rstrip(')')
            code_lines[-1] += f", filename={str(html_path)!r})"
        else:
            print(f"    ⚠ Cannot auto-patch cell source. Manual export needed.")
            continue

        # Execute to generate the HTML file
        full_script = '\n\n'.join(code_lines)
        script_path = tutorials_dir / '_webgui_export_temp.py'

        try:
            with open(script_path, 'w') as f:
                f.write(full_script)

            import subprocess
            python_exe = sys.executable
            result = subprocess.run(
                [python_exe, str(script_path)],
                capture_output=True, text=True, timeout=120,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode != 0:
                print(f"    ✗ Execution failed:")
                print(f"      {result.stderr[:500]}")
                continue

            if not html_path.exists():
                print(f"    ✗ HTML file not generated")
                continue

            print(f"    ✓ Generated {html_filename} ({html_path.stat().st_size // 1024} KB)")

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue
        finally:
            if script_path.exists():
                script_path.unlink()

        # Patch the notebook output with an HTML fallback
        iframe_html = (
            f'<div style="width:100%;height:500px;border:1px solid #ccc;border-radius:12px;overflow:hidden;">'
            f'<iframe src="{html_filename}" style="width:100%;height:100%;border:none;"></iframe>'
            f'</div>'
        )

        # Find and update the widget output
        for output in cell['outputs']:
            if output.get('output_type') in ('display_data', 'execute_result'):
                data = output.get('data', {})
                if 'application/vnd.jupyter.widget-view+json' in data:
                    # Add text/html fallback for static rendering
                    data['text/html'] = iframe_html
                    modified = True
                    print(f"    ✓ Patched notebook output with iframe fallback")
                    break

    if modified:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✓ Saved patched notebook")

    return modified


def main():
    if len(sys.argv) > 1:
        notebooks = [Path(p) for p in sys.argv[1:]]
    else:
        tutorials_dir = PROJECT_ROOT / 'docs' / 'tutorials'
        notebooks = sorted(tutorials_dir.glob('*.ipynb'))

    if not notebooks:
        print("No notebooks found.")
        return

    print(f"Processing {len(notebooks)} notebook(s)...")

    for nb_path in notebooks:
        try:
            export_webgui_from_notebook(nb_path)
        except Exception as e:
            print(f"  ✗ Error processing {nb_path.name}: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
