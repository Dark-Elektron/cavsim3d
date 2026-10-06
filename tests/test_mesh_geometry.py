"""Line and face queries on mesh surfaces (no NGSolve point search)."""

from cavsim3d import EMProject  # noqa: F401  (loads pythonocc before netgen)
import numpy as np
import pytest
from netgen.occ import Box, Glue, OCCGeometry, Pnt, Sphere
from ngsolve import Mesh

from cavsim3d.utils.mesh_geometry import line_intervals, surface_triangles, line_crossings


@pytest.fixture(scope="module")
def sphere():
    R = 0.1
    m = Mesh(OCCGeometry(Sphere(Pnt(0, 0, 0), R)).GenerateMesh(maxh=0.04))
    m.Curve(4)
    return m, R


@pytest.mark.parametrize("xo", [0.0, 0.03, 0.07, 0.095])
def test_crossings_on_the_curved_surface(sphere, xo):
    """The stretch of a line through a curved sphere ends on the curved
    surface, not on its straight triangles (which are centimetres off)."""
    m, R = sphere
    yo = 0.011
    iv = line_intervals(m, (xo, yo, 0.0), 2)
    exact = np.sqrt(R ** 2 - xo ** 2 - yo ** 2)
    assert len(iv) == 1
    s0, s1, _, _ = iv[0]
    assert s0 == pytest.approx(-exact, abs=2e-6)
    assert s1 == pytest.approx(exact, abs=2e-6)


def test_line_missing_the_mesh(sphere):
    m, R = sphere
    assert line_intervals(m, (1.2 * R, 0.0, 0.0), 2) == []


def test_stretches_per_domain_and_face_names():
    """Two glued boxes: stretches of the whole mesh and of each domain, with
    the names of the faces where the line enters and leaves."""
    b1 = Box(Pnt(0, 0, 0), Pnt(1, 1, 1))
    b2 = Box(Pnt(0, 0, 1), Pnt(1, 1, 3))
    b1.mat('one')
    b2.mat('two')
    geo = Glue([b1, b2])
    for f in geo.faces:
        lo, hi = f.bounding_box
        if hi.z - lo.z < 1e-5:
            f.name = {0.0: 'bottom', 1.0: 'middle', 3.0: 'top'}[round(lo.z, 6)]
        else:
            f.name = 'wall'
    m = Mesh(OCCGeometry(geo).GenerateMesh(maxh=0.5))
    mats = list(m.GetMaterials())
    pt = (0.3, 0.6, 0.0)
    assert line_intervals(m, pt, 2) == [(0.0, 3.0, 'bottom', 'top')]
    one = mats.index('one') + 1
    two = mats.index('two') + 1
    assert line_intervals(m, pt, 2, domains=[one]) == [(0.0, 1.0, 'bottom', 'middle')]
    assert line_intervals(m, pt, 2, domains=[two]) == [(1.0, 3.0, 'middle', 'top')]
    # the internal face is crossed once
    tris, _ = surface_triangles(m, names=['middle'])
    s, _ = line_crossings(tris, pt, 2)
    assert np.allclose(s, 1.0)
