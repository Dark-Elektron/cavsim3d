# conftest.py (at repo root)
import sys
from pathlib import Path

# Add cavsim3d/ to path so tests can import geometry, solvers, etc.
sys.path.insert(0, str(Path(__file__).parent / "cavsim3d"))
