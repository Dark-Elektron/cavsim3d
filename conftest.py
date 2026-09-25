# conftest.py (at repo root)
import sys
from pathlib import Path

# Make the repository root importable so `import cavsim3d` works without an
# install.  (Only the root: putting cavsim3d/ itself on sys.path would expose
# its subpackages as top-level `core`, `utils`, `geometry`, ... -- a second
# copy of every module, and a shadow for any dependency importing those names.)
_root = str(Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
