# 04_2.8.5_Jacobi_Methods

"""

Lecture: 2._Chapters/2.8_Symmetric_Eigenvalue_Problems
Content: 04_2.8.5_Jacobi_Methods

"""

import numpy as np
from typing import Tuple

class JacobiEigenSolver:
    def __init__(self, matrix: np.ndarray):
        """
        Initialize the JacobiEigenSolver with a given symmetric matrix.

        Args:
        - matrix (np.ndarray): The symmetric matrix to solve for eigenvalues and eigenvectors.
        """
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
        assert np.allclose(matrix, matrix.T), "Matrix must be symmetric"
        self.matrix = matrix
        self.n = matrix.shape[0]

    def _rotate(self, A: np.ndarray, p: int, q: int) -> np.ndarray:
        """
        Perform a Jacobi rotation to zero out the A[p, q] and A[q, p] elements.

        Args:
        - A (np.ndarray): The matrix to be rotated.
        - p (int): The row index of the element to be zeroed out.
        - q (int): The column index of the element to be zeroed out.

        Returns:
        - A (np.ndarray): The rotated matrix.
        """
        if A[p, q] != 0:
            theta = (A[q, q] - A[p, p]) / (2 * A[p, q])
            t = np.sign(theta) / (np.abs(theta) + np.sqrt(1 + theta ** 2))
            c = 1 / np.sqrt(1 + t ** 2)
            s = t * c
        else:
            c = 1
            s = 0

        for i in range(self.n):
            if i != p and i != q:
                Aip = c * A[i, p] - s * A[i, q]
                Aiq = s * A[i, p] + c * A[i, q]
                A[i, p] = Aip
                A[i, q] = Aiq
                A[p, i] = Aip
                A[q, i] = Aiq

        App = c ** 2 * A[p, p] + s ** 2 * A[q, q] - 2 * s * c * A[p, q]
        Aqq = s ** 2 * A[p, p] + c ** 2 * A[q, q] + 2 * s * c * A[p, q]
        Apq = 0

        A[p, p] = App
        A[q, q] = Aqq
        A[p, q] = Apq
        A[q, p] = Apq

        return A

    def solve(self, tol: float = 1e-10, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the eigenvalue problem using the Jacobi method.

        Args:
        - tol (float): Tolerance for convergence.
        - max_iterations (int): Maximum number of iterations.

        Returns:
        - eigenvalues (np.ndarray): The eigenvalues of the matrix.
        - eigenvectors (np.ndarray): The eigenvectors of the matrix.
        """
        A = self.matrix.copy()
        V = np.eye(self.n)

        for iteration in range(max_iterations):
            off_diagonal_sum = np.sum(np.abs(A) - np.diag(np.abs(A)))
            if off_diagonal_sum < tol:
                break

            for p in range(self.n - 1):
                for q in range(p + 1, self.n):
                    A = self._rotate(A, p, q)

        eigenvalues = np.diag(A)
        return eigenvalues, V

# 示例对称矩阵
symmetric_matrix = np.array([
    [4, 1, 2],
    [1, 2, 0],
    [2, 0, 3]
])

solver = JacobiEigenSolver(symmetric_matrix)
eigenvalues, eigenvectors = solver.solve()

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
