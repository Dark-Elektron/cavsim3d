"""Position-based ports go on the flat end faces, also when extents tie.

circular_waveguide.step is a cylinder as long as it is wide (0.3 x 0.3 x 0.3
m), so the bounding box alone does not say which way it runs.
"""
from pathlib import Path

import pytest

from cavsim3d.core.em_project import EMProject  # noqa: F401  (import order)
from cavsim3d.geometry.importers import OCCImporter

MODELS = Path(__file__).resolve().parents[1] / "docs" / "example_models"


@pytest.mark.parametrize("name, axis", [
    ("circular_waveguide", "Z"),
    ("rectangular_waveguide", "Z"),
    ("pillbox", "X"),
])
def test_ports_on_end_faces(name, axis):
    geo = OCCImporter(str(MODELS / f"{name}.step"), unit="m")
    assert geo._port_axis == axis
    i = "XYZ".index(axis)
    lo, hi = geo.get_bounding_box()
    ports = {f.name: f.center[i] for f in geo.geo.faces if f.name.startswith("port")}
    assert ports["port1"] == pytest.approx(lo[i], abs=1e-6)
    assert ports["port2"] == pytest.approx(hi[i], abs=1e-6)
