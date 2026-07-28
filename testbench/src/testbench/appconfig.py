"""App configuration (metadata) — the ngapp entry point of the test bench."""

from ngapp.app import AppConfig

from .app import TestBench

config = AppConfig(
    python_class=TestBench,
    name="cavsim3d test bench",
    version="0.4.0",
    description="CST-style studio: geometry modelling with primitives + "
                "project import, meshing, BC editing, staged pipeline runs "
                "and ROM-reconstructed 3D fields.",
)
