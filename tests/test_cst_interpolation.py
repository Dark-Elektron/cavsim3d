"""interpolate_to must preserve CST's exported Z-matrix.

CST references Z to each port's line impedance. interpolate_to used to discard
the exported Z and re-derive it from the interpolated S using the default
z0 = 50 ohm, which re-referenced every Z by a constant factor (95.84/50 = 1.92
on the coaxial test model) -- large enough to make a correct solver look ~93%
wrong on Z while its S comparison stayed perfect.
"""

import numpy as np
import pytest

from cavsim3d.analytical.cst_result import CSTResult

REF = (__import__('pathlib').Path(__file__).parent.parent
       / 'docs' / 'example_models' / 'cst_reference' / 'tem_2port_line')

skip_no_ref = pytest.mark.skipif(
    not REF.exists(), reason='CST reference data not available'
)


@skip_no_ref
class TestInterpolatePreservesZ:

    def test_identity_grid_leaves_z_unchanged(self):
        """Interpolating onto the existing grid must be a no-op for Z."""
        cst = CSTResult(str(REF))
        same = cst.interpolate_to(cst.frequencies)
        for key, arr in cst.Z_dict.items():
            assert np.allclose(np.abs(same.Z_dict[key]), np.abs(arr), rtol=1e-6), (
                f'{key} changed under identity interpolation'
            )

    def test_z_not_rereferenced_to_default_z0(self):
        """The old behaviour rescaled |Z| by ~1.9; guard the magnitude."""
        cst = CSTResult(str(REF))
        sub = cst.interpolate_to(cst.frequencies[::10])
        a = np.abs(cst.Z_dict['1(1)1(1)'][::10])
        b = np.abs(sub.Z_dict['1(1)1(1)'])
        m = a > 1e-9
        assert np.median(b[m] / a[m]) == pytest.approx(1.0, abs=2e-3)

    def test_exported_z_is_line_referenced(self):
        """Solving for the scalar reference implied by CST's own S and Z
        recovers the port line impedance (~95.84 ohm), not 50 ohm."""
        cst = CSTResult(str(REF))
        S = cst.S_matrix
        Z = cst.Z_matrix
        I = np.eye(S.shape[-1])
        implied = []
        for k in range(0, len(cst.frequencies), 25):
            W = (I + S[k]) @ np.linalg.inv(I - S[k])
            implied.append((np.vdot(W, Z[k]) / np.vdot(W, W)).real)
        assert np.median(implied) == pytest.approx(95.84, rel=1e-3)
