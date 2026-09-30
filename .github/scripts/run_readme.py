"""Run the Python code blocks of README.md in order, as one script.

Used by CI so the walkthrough on the project's front page keeps working.  The
blocks run in a temporary working directory (the walkthrough writes a project
folder there) with a non-interactive matplotlib backend.
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

readme = Path(__file__).resolve().parents[2] / "README.md"
blocks = re.findall(r"```python\n(.*?)```", readme.read_text(encoding="utf-8"), re.S)
if not blocks:
    sys.exit("README.md has no ```python blocks")

namespace = {"__name__": "__readme__"}
start = os.getcwd()
with tempfile.TemporaryDirectory() as work:
    os.chdir(work)
    try:
        for i, code in enumerate(blocks, 1):
            print(f"--- README block {i}/{len(blocks)}", flush=True)
            exec(compile(code, f"README.md[block {i}]", "exec"), namespace)
    finally:
        os.chdir(start)          # Windows cannot delete the working directory
print("README walkthrough ran without errors")
