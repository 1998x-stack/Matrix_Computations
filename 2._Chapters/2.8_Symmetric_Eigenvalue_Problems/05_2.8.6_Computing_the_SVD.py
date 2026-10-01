# 05_2.8.6_Computing_the_SVD

"""

Lecture: 2._Chapters/2.8_Symmetric_Eigenvalue_Problems
Content: 05_2.8.6_Computing_the_SVD

"""

import numpy as np
from typing import Tuple

class SVDSolver:
    def __init__(self, matrix: np.ndarray):
        """
        Initialize the SVDSolver with a given matrix.

        Args:
        - matrix (np.ndarray): The matrix to decompose.
        """
        assert matrix.ndim == 2, "Input must be a 2D matrix"
        self.matrix = matrix
        self.m, self.n = matrix.shape

    def _bidiagonalize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bidiagonalize the matrix A using Householder reflections.

        Args:
        - A (np.ndarray): The input matrix to be bidiagonalized.

        Returns:
        - B (np.ndarray): The bidiagonal matrix.
        - U (np.ndarray): The orthogonal matrix U.
        - V (np.ndarray): The orthogonal matrix V.
        """
        U = np.eye(self.m)
        V = np.eye(self.n)
        B = A.copy()

        for i in range(min(self.m, self.n)):
            # Apply Householder transformation to rows
            x = B[i:, i]
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x) if x[0] == 0 else np.sign(x[0]) * np.linalg.norm(x)
            u = x + e1
            u /= np.linalg.norm(u)

            H = np.eye(self.m)
            H[i:, i:] -= 2.0 * np.outer(u, u)
            B = H @ B
            U = U @ H

            if i < self.n - 1:
                # Apply Householder transformation to columns
                x = B[i, i+1:]
                e1 = np.zeros_like(x)
                e1[0] = np.linalg.norm(x) if x[0] == 0 else np.sign(x[0]) * np.linalg.norm(x)
                u = x + e1
                u /= np.linalg.norm(u)

                H = np.eye(self.n)
                H[i+1:, i+1:] -= 2.0 * np.outer(u, u)
                B = B @ H
                V = V @ H

        return B, U, V

    def _qr_algorithm(self, B: np.ndarray, tol: float = 1e-10, max_iterations: int = 1000) -> np.ndarray:
        """
        Perform the QR algorithm with shifts on the bidiagonal matrix B.

        Args:
        - B (np.ndarray): The bidiagonal matrix.
        - tol (float): Tolerance for convergence.
        - max_iterations (int): Maximum number of iterations.

        Returns:
        - B (np.ndarray): The matrix B with singular values converged on the diagonal.
        """
        for _ in range(max_iterations):
            off_diagonal_sum = np.sum(np.abs(np.diag(B, k=1)))
            if off_diagonal_sum < tol:
                break

            mu = B[-1, -1]
            Q, R = np.linalg.qr(B - mu * np.eye(B.shape[0]))
            B = R @ Q + mu * np.eye(B.shape[0])

        return B

    def compute_svd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Singular Value Decomposition (SVD) of the matrix.

        Returns:
        - U (np.ndarray): The orthogonal matrix U.
        - S (np.ndarray): The singular values as a diagonal matrix.
        - V (np.ndarray): The orthogonal matrix V.
        """
        B, U, V = self._bidiagonalize(self.matrix)
        B = self._qr_algorithm(B)

        singular_values = np.diag(B)
        S = np.zeros_like(self.matrix)
        np.fill_diagonal(S, singular_values)

        return U, S, V.T

# 示例矩阵
matrix = np.array([
    [4, 1, 2],
    [1, 2, 0],
    [2, 0, 3]
])

solver = SVDSolver(matrix)
U, S, V = solver.compute_svd()

print("Matrix U:\n", U)
print("Singular values (diagonal of S):\n", np.diag(S))
print("Matrix V^T:\n", V)
