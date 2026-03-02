"""Matrix utilities for sparse matrix operations and testing."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from typing import Tuple, Optional

try:
    from sksparse.cholmod import cholesky, CholmodError

    HAS_CHOLMOD = True
except ImportError:
    HAS_CHOLMOD = False
    CholmodError = Exception


class MatrixAnalyzer:
    """Utilities for analyzing and factorizing sparse matrices."""

    @staticmethod
    def check_symmetry(M: sp.spmatrix, tol: float = 1e-12) -> Tuple[bool, float]:
        """
        Check if matrix is symmetric.

        Returns
        -------
        is_symmetric : bool
        max_diff : float
            Maximum asymmetry
        """
        diff = M - M.T
        if diff.nnz == 0:
            return True, 0.0

        max_diff = np.max(np.abs(diff.data))
        return max_diff < tol, max_diff

    @staticmethod
    def check_positive_definite(
            M: sp.spmatrix,
            tol: float = 1e-12
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if matrix is positive definite via Cholesky.

        Returns
        -------
        is_pd : bool
        L : ndarray or None
            Cholesky factor if successful
        """
        if not HAS_CHOLMOD:
            # Fallback: check eigenvalues (expensive)
            eigs = sp.linalg.eigsh(M, k=min(6, M.shape[0] - 2), which='SA', return_eigenvectors=False)
            return np.all(eigs > tol), None

        M_csc = M.tocsc()
        try:
            factor = cholesky(M_csc)
            L = factor.L()
            return True, L.toarray()
        except CholmodError:
            return False, None

    @staticmethod
    def check_singularity(K: sp.spmatrix, tol: float = 1e-12) -> Tuple[bool, float]:
        """
        Check if matrix is singular via LU decomposition.

        Returns
        -------
        is_singular : bool
        det_approx : float
            Approximate determinant
        """
        K_csc = K.tocsc()
        try:
            lu = splu(K_csc)
            diag_U = lu.U.diagonal()
            det = np.prod(np.sign(diag_U)) * np.exp(np.sum(np.log(np.abs(diag_U))))
            return np.abs(det) < tol, det
        except RuntimeError:
            return True, 0.0

    @staticmethod
    def regularize(
            M: sp.spmatrix,
            reg: float = 1e-12
    ) -> sp.spmatrix:
        """Add small regularization to diagonal."""
        n = M.shape[0]
        return M + reg * sp.eye(n, format=M.format)

    @staticmethod
    def condition_number_estimate(M: sp.spmatrix, k: int = 6) -> float:
        """
        Estimate condition number using extreme eigenvalues.

        Parameters
        ----------
        M : spmatrix
            Symmetric matrix
        k : int
            Number of eigenvalues to compute
        """
        n = M.shape[0]
        k = min(k, n - 2)

        # Largest eigenvalues
        eigs_large = sp.linalg.eigsh(M, k=k, which='LM', return_eigenvectors=False)
        # Smallest eigenvalues
        eigs_small = sp.linalg.eigsh(M, k=k, which='SM', return_eigenvectors=False)

        lam_max = np.max(np.abs(eigs_large))
        lam_min = np.min(np.abs(eigs_small))

        if lam_min < 1e-14:
            return np.inf
        return lam_max / lam_min


def sparse_block_diag(*matrices) -> sp.spmatrix:
    """Create block diagonal sparse matrix."""
    return sp.block_diag(matrices, format='csr')


def extract_freedofs(
        M: sp.spmatrix,
        freedofs
) -> sp.spmatrix:
    """Extract submatrix corresponding to free DOFs."""
    # Convert NGSolve BitArray to numpy bool array
    free = np.array([freedofs[i] for i in range(len(freedofs))])
    indices = np.where(free)[0]
    return M[indices, :][:, indices]


class MatrixTester:
    """Comprehensive matrix testing suite."""

    def __init__(self, M: sp.spmatrix, name: str = "Matrix"):
        self.M = M
        self.name = name
        self.results = {}

    def run_all_tests(self, verbose: bool = True) -> dict:
        """Run all matrix tests."""
        analyzer = MatrixAnalyzer()

        # Symmetry
        is_sym, max_diff = analyzer.check_symmetry(self.M)
        self.results['symmetric'] = is_sym
        self.results['symmetry_error'] = max_diff

        # Positive definiteness
        is_pd, L = analyzer.check_positive_definite(self.M)
        self.results['positive_definite'] = is_pd

        # Singularity
        is_sing, det = analyzer.check_singularity(self.M)
        self.results['singular'] = is_sing
        self.results['determinant'] = det

        if verbose:
            self._print_results()

        return self.results

    def _print_results(self):
        """Print test results with colors."""
        print(f"\n{'=' * 60}")
        print(f"Matrix Analysis: {self.name}")
        print(f"{'=' * 60}")
        print(f"Shape: {self.M.shape}, NNZ: {self.M.nnz}")

        # Symmetry
        sym_status = "\033[32mYES\033[0m" if self.results['symmetric'] else "\033[31mNO\033[0m"
        print(f"Symmetric: {sym_status} (max diff: {self.results['symmetry_error']:.2e})")

        # PD
        pd_status = "\033[32mYES\033[0m" if self.results['positive_definite'] else "\033[31mNO\033[0m"
        print(f"Positive Definite: {pd_status}")

        # Singularity
        sing_status = "\033[31mYES\033[0m" if self.results['singular'] else "\033[32mNO\033[0m"
        print(f"Singular: {sing_status}")

        print(f"{'=' * 60}\n")