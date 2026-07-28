"""cavsim3d interactive test bench (ngapp).

Run (after ``pip install -e testbench/`` from the cavsim3d repo root):

    python -m testbench            # serve the app (opens the browser)
    python -m testbench --dev      # hot reload on code changes

IMPORTANT: connect with the FULL URL the server prints (it carries the
websocket port + token); a plain http://localhost:8765 stays blank.
"""

from .appconfig import config
from .app import TestBench

__all__ = ["TestBench", "config"]
