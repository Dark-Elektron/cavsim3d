
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import matplotlib
from matplotlib.colors import Normalize
from sksparse.cholmod import cholesky, CholmodError
import scipy.sparse as sp
import scipy.linalg as sl


def spy(A, cmap='tab20', markersize=10):
    """Visualise a sparse matrix (including complex) with colours representing entry magnitude."""
    A = A.tocoo()
    fig, ax = plt.subplots(figsize=(6,6))

    values = np.abs(A.data)  # take magnitudes for complex matrices
    norm = Normalize(vmin=values.min(), vmax=values.max())
    cmap_ = matplotlib.colormaps[cmap]

    sc = ax.scatter(A.col, A.row, marker='s', c=values, s=markersize,
                    cmap=cmap_, norm=norm)

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Coloured sparse matrix pattern (|value|)")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Row index")

    fig.colorbar(sc, ax=ax, label="|Value| magnitude")
    plt.show()

def test_and_factorize_sparse(M, name, test='Positive', tol=1e-12):
    """
    Performs symmetry and PD/PSD tests, accepts positive definite and positive semi-definite.
    Only fails if the matrix is not symmetric or has negative eigenvalues.
    """
    print(f"--- Testing Matrix {name} ---")

    if not (M - M.T).nnz == 0:
        diff = (M - M.conj().T).data
        max_diff = np.max(np.abs(diff)) if diff.size else 0
        if max_diff > tol:
            print(f"\x1b[31m SYMMETRY: Matrix is NOT symmetric (max diff {max_diff:.2e})\x1b[0m")
            return None, M, False
        else:
            print("\x1b[32m SYMMETRY: Matrix is symmetric.\x1b[0m")

    else:
        print("\x1b[32m SYMMETRY: Matrix is symmetric.\x1b[0m")

    M_csc = M.tocsc()

    try:
        factor = cholesky(M_csc)
        L = factor.L()
        P = factor.P()
        print(f"\x1b[32m Matrix is {test} Definite (PD). Cholesky succeeded.\n")
        R = L.T[P.argsort()][:, P.argsort()]
        return R, M, True

    except CholmodError as e:
        print("\t\x1b[33m Cholesky failed. Possibly semi-definite or singular.\x1b[0m")
        print("\tTrying regularised decomposition...")

        # Regularisation: shift matrix slightly by tol*I
        n = M.shape[0]
        M_reg = M_csc + sp.eye(n, format="csc") * tol
        try:
            factor = cholesky(M_reg)
            L = factor.L()
            P = factor.P()
            print(f"\t\x1b[32m Matrix is {test} Semi-Definite (PSD). Regularised Cholesky succeeded.\x1b[0m")
            R = L.T[P.argsort()][:, P.argsort()]
            return R, M_reg, True
        except CholmodError:
            print(f"\t\x1b[31m Matrix is NOT {test} Semi-Definite.\n\x1b[0m")
            return None, M, False

def check_singularity(K, tol=1e-12):
    """Safely test if sparse matrix K is singular or nearly singular."""
    K_csc = K.tocsc()
    try:
        lu = splu(K_csc)
        diagU = lu.U.diagonal()
        det = np.prod(diagU)
        if abs(det) < tol:
            print("Matrix is singular or nearly singular.")
        else:
            print("Matrix is non-singular.")
        print("Determinant (approx):", det)
        return det
    except RuntimeError as e:
        print("LU factorisation failed (matrix may be singular).")
        print("Error:", e)
        return 0.0

def calc_error(R, M, name):
    M_reconstructed = R.transpose() @ R
    diff = M.toarray() - M_reconstructed.toarray()
    max_err = np.max(np.abs(diff))
    rel_max_err = max_err / np.max(np.abs(M.toarray()))

    print(f"\n--- Verification for {name} ---")
    print(f"Max reconstruction error: {max_err:.2e}")
    print("Relative max error:", rel_max_err)
    print()

def run_tests(A, name):
    R, A, res = test_and_factorize_sparse(A, name, test='Positive') # in case A is regularised
    if not res:
        print('_'*80)
        print("Cholesky failed. Possibly negative-definite, negative semi-definite or singular.")
        print('_'*80)
        R, A, res = test_and_factorize_sparse(-A, name, test='Negative') # in case A is regularised

    if R is not None:
        calc_error(R, A, name)

    check_singularity(A)
#     spy(A)
    print('='*80)
    return R

def parametric_line(self, start, end, npts=10000):
    t = np.linspace(0, 1, npts)
    line_points = start[None, :] + t[:, None] * (end - start)[None, :]
    return line_points

def build_embedding_matrix(self, fes_full, fes_edge, region_name):
    """
    Build a sparse embedding/prolongation matrix that maps a solution
    from fes_edge (defined on a boundary) into fes_full.

    Parameters
    ----------
    fes_full : FESpace
        Full FE space (target space)
    fes_edge : FESpace
        Restricted FE space defined on a boundary (source space)
    region_name : str
        Name of the boundary region on which fes_edge is defined

    Returns
    -------
    E : csr_matrix (shape fes_full.ndof x fes_edge.ndof)
        Embedding matrix such that:
            x_full = E @ x_edge
    """

    m = fes_full.ndof
    n = fes_edge.ndof

    E = lil_matrix((m, n))

    # loop over elements in the specified boundary region
    for el in fes_edge.mesh.Boundaries(region_name).Elements():
        dofs_full = fes_full.GetDofNrs(el)
        dofs_edge = fes_edge.GetDofNrs(el)

        if len(dofs_full) != len(dofs_edge):
            raise ValueError("Mismatch in number of local DOFs between full and edge spaces")

        # insert 1's for corresponding DOFs
        for lf in range(len(dofs_edge)):
            df = dofs_full[lf]
            de = dofs_edge[lf]
            E[df, de] = 1

    return E.tocsr()

def get_point_on_boundary(mesh, boundary_label):
    """
    Finds the coordinates of a vertex belonging to the specified boundary label.

    Args:
        mesh: The NGSolve Mesh object.
        boundary_label (str): The name of the boundary (e.g., 'port1', 'port2').

    Returns:
        tuple or None: (x, y, z) coordinates of a vertex on the boundary, or None if not found.
    """
    # Iterate over all boundary elements
    for bel in mesh.Elements(BND):
        if bel.mat == boundary_label:
            # Pick the first vertex
            vertex_index = bel.vertices[0]
            coords = mesh[vertex_index].point  # mesh[index] gives the vertex
            return tuple(coords)

    return None


def get_boundary_normal(self, mesh, boundary_label):
    """
    Calculates the constant normal vector for a planar boundary face
    by integrating the normal vector CF over the surface and dividing by the area.

    Args:
        mesh: The NGSolve Mesh object.
        boundary_label (str): The name of the boundary (e.g., 'port1', 'port2').

    Returns:
        np.array or None: The (nx, ny, nz) normal vector, or None if the area is zero.
    """
    # 1. Define the normal vector as a special CoefficientFunction
    nhat = specialcf.normal(3)

    # 2. Integrate the normal vector (gives the area vector)
    # The result is a tuple (Int(nx), Int(ny), Int(nz))
    integral_of_n = Integrate(nhat,
                              mesh,
                              BND,
                              definedon=mesh.Boundaries(boundary_label))

    # 3. Integrate 1 (gives the total scalar area)
    face_area = Integrate(1,
                          mesh,
                          BND,
                          definedon=mesh.Boundaries(boundary_label))

    # Convert integral_of_n to a NumPy array for component-wise division
    integral_of_n = np.array(integral_of_n)

    # 4. Calculate the normal vector (Area Vector / Scalar Area)
    if abs(face_area) < 1e-12:
        print(f"Warning: Area for boundary '{boundary_label}' is effectively zero.")
        return None

    normal_vector = -integral_of_n / face_area

    # Round the result to clean up numerical noise
    return np.round(normal_vector, decimals=6)


def field1D_plot(self, gfu, mesh, curve, component='mag'):
    if component == 'mag':
        E1d = [Norm(gfu)(mesh(*pt)) for pt in curve]
    elif component == 'x':
        E1d = [gfu[0](mesh(*pt)) for pt in curve]
    elif component == 'y':
        E1d = [gfu[1](mesh(*pt)) for pt in curve]
    elif component == 'z':
        E1d = [gfu[2](mesh(*pt)) for pt in curve]
    else:
        E1d = [Norm(gfu)(mesh(*pt)) for pt in curve]

    return np.array(E1d)


def plot_field_on_axis(self):
    start = np.array([rwg.a/2, rwg.b/2, 0.0])   # start point
    end = np.array([rwg.a/2, rwg.b/2, rwg.L])   # end point
    line = self.parametric_line(start, end)

    e1d = self.field1D_plot(E, rwg.mesh, line, component='y')
    # e1da = self.field1D_plot(E_analytic, rwg.mesh, line, component='y')
    # e1da = self.analytic_(rwg, line)

    plt.close()
    plt.plot(line[:, 2], np.imag(e1d), label='numeric')
    # plt.plot(line[:, 2], np.imag(e1da), label='analytic')
    # plt.plot(line[:, 2], np.imag(e1da - e1d), label='error')

    # plt.yscale('log')
    # plt.ylabel(r'$\frac{|E(a/2, b/2, z)|}{max(|E(a/2, b/2, z)|)}$ []', fontsize=14)
    plt.ylabel(r'$imag[Ey(a/2, b/2, z)]$ [V/m]', fontsize=14)
    plt.xlabel('L [m]', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_rom(self, W, K, M, B, tol=1e-1, show=False):
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    if show:
        fig,ax = plt.subplot_mosaic([[1]], layout='constrained', figsize=(8,4))
        ax[1].plot(S)
        ax[1].set_yscale('log')
        plt.grid()
        plt.show()

    print('Decomp shapes', U.shape, S.shape, Vt.shape)
    r = len(S[S>tol])
    Sr = S[:r]
    Vtr = Vt[:r, :]
    Wr = U[:, :r]
    print('Trunc shapes', Wr.shape, Sr.shape, Vtr.shape, r)

    # self._test_reconstruction(W, Wr, Sr, Vtr)
    Kr, Mr = Wr.T@K@Wr, Wr.T@M@Wr

    Kr = (Kr + Kr.T)/2
    Mr = (Mr + Mr.T)/2
    check_reduced_model(Wr, Kr, Mr)

    lam, Q = sl.eigh(Mr)

    inv_sqrt_lam = 1/np.sqrt(lam)
    QLinv = Q@np.diag(inv_sqrt_lam)

    Ard = QLinv.T@Kr@QLinv
    Ard = (Ard + Ard.T)/2
    Brd = QLinv.T @ Wr.T @ B

    return Wr, Ard, QLinv, Brd

# check eigenvalues
def check_eigs_system(A, mode=None):
        lrp, xrp = sl.eigh(A)
        # sort eigenmode and eigenvalues
        idx_sort = np.argsort(np.abs(lrp))
        lrp = lrp[idx_sort]
        xrp = xrp[:, idx_sort]
        return lrp, xrp

def check_reduced_model(Wr, Rr, Kr, draw=False):
    print('============Checking reduced system eigenvalue============')
    # check eigenvalue sof reduced system
    evals_r, evecs_r = sl.eigh(Rr, Kr)
    # evals_r, evecs_r = sp.linalg.eigs(Rr, k=10, M=Mr)

    sort = np.argsort(evals_r)
    evals_r = evals_r[sort]
    evecs_r = evecs_r[sort]
    print('evals_rd', evals_r)

    mode = 0
    xr = Wr@evecs_r
    xrm = xr[:, mode]
    # if draw:
    #     fes = HCurl(self.mesh, order=self.fesorder, dirichlet=self.bc)
    #     Er = GridFunction(fes)
    #     Er.vec.data = BaseVector(xrm)
    #     Er = BoundaryFromVolumeCF(Er)
    #     Draw(Norm(Er), self.mesh, settings=settings)
    # print('============Done checking reduced system eigenvalue============\n')
    return xr
