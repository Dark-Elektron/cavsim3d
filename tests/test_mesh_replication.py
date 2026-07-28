"""Exact rigid placement of H(curl) fields onto compound meshes (utility)."""

import numpy as np
import pytest

from cavsim3d.utils.mesh_replication import (
    Placement, replicate_mesh, block_dof_maps, assemble_compound_field,
)


class TestMeshReplication:

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_exact_placement_translation_and_rotation(self, order):
        from netgen.occ import Box, OCCGeometry, Pnt
        from ngsolve import Mesh, HCurl, GridFunction, CF, x, y, z

        ref = Mesh(OCCGeometry(Box(Pnt(0, 0, 0), Pnt(1, 1, 1))).GenerateMesh(maxh=0.5))
        fes = HCurl(ref, order=order)
        gf = GridFunction(fes); gf.Set(CF((y, z, x)))

        places = [
            Placement((0, 0, 0)),
            Placement((2, 0, 0)),
            Placement((4, 0, 0), rotation=((0, 0, 1), np.pi / 2, (0.5, 0.5, 0.5))),
        ]
        cm = replicate_mesh(ref, places)
        fesc = HCurl(cm, order=order)
        maps = block_dof_maps(fes, fesc, ref, cm, len(places))
        gfc = assemble_compound_field(fesc, [gf.vec.FV().NumPy()] * len(places), maps)

        xr = np.array([0.3, 0.4, 0.5]); Fxr = np.array([xr[1], xr[2], xr[0]])
        for pl in places:
            R, _c, _t = pl.matrix()
            got = np.array(gfc(cm(*pl.apply(xr))))
            assert np.allclose(got, R @ Fxr, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
