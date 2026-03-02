import numpy as np
from cavsim3d.helpers import build_F, permutation_matrix_from_list, build_global_excitation, z2s


def test_build_F_shapes_and_values():
    F = build_F(3, dtype=int)
    assert F.shape == (6, 3)
    # each column j should have 1 at 2j and -1 at 2j+1
    for j in range(3):
        assert F[2*j, j] == 1
        assert F[2*j+1, j] == -1


def test_permutation_matrix_from_list():
    perm = [2, 0, 1]
    PT, P = permutation_matrix_from_list(perm)
    # PT maps old->new positions: PT @ old = new
    old = np.array([10, 20, 30])
    new = PT @ old
    assert list(new) == [30, 10, 20]
    # P is transpose
    assert np.all(P == PT.T)


def test_build_global_excitation_padding_and_stack():
    a = np.array([1, 0])  # 1D -> column
    b = np.array([[0, 1], [1, 0]])
    I_global = build_global_excitation([a, b])
    # first block should become column, padded to 2 cols
    assert I_global.shape == (3, 2)
    assert I_global[0, 0] == 1


def test_z2s_zero_for_z_equals_z0_identity():
    Z0 = 50.0
    Z = Z0 * np.eye(2)
    S = z2s(Z, Z0)
    # If Z == Z0*I then scattering matrix should be zeros
    assert np.allclose(S, np.zeros_like(S), atol=1e-12)
