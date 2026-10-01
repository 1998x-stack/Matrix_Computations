# 02_2.8.3_The_Symmetric_QR_Algorithm

"""

Lecture: 2._Chapters/2.8_Symmetric_Eigenvalue_Problems
Content: 02_2.8.3_The_Symmetric_QR_Algorithm

"""

import numpy as np
from typing import Tuple

class SymmetricQRAlgorithm:
    def __init__(self, matrix: np.ndarray):
        """
        Initialize the SymmetricQRAlgorithm with a given symmetric matrix.

        Args:
        - matrix (np.ndarray): The symmetric matrix to solve for eigenvalues and eigenvectors.
        """
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
        self.matrix = matrix
        self.n = matrix.shape[0]

    def tridiagonalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tridiagonalize the symmetric matrix using Householder transformations.

        Returns:
        - T (np.ndarray): The tridiagonal matrix.
        - Q (np.ndarray): The orthogonal matrix used in the transformation.
        """
        T = self.matrix.copy()
        Q = np.eye(self.n)

        for k in range(self.n - 2):
            x = T[k+1:, k]
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x) if x[0] == 0 else np.sign(x[0]) * np.linalg.norm(x)
            u = x + e1
            u = u / np.linalg.norm(u)

            H = np.eye(self.n)
            H[k+1:, k+1:] -= 2.0 * np.outer(u, u)

            T = H @ T @ H
            Q = Q @ H

        return T, Q

    def qr_algorithm(self, T: np.ndarray, max_iterations: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform the QR algorithm with shifts to find the eigenvalues and eigenvectors of the tridiagonal matrix.

        Args:
        - T (np.ndarray): The tridiagonal matrix.
        - max_iterations (int): The maximum number of iterations.
        - tol (float): The tolerance for convergence.

        Returns:
        - eigenvalues (np.ndarray): The eigenvalues of the matrix.
        - eigenvectors (np.ndarray): The eigenvectors of the matrix.
        """
        n = T.shape[0]
        Q_total = np.eye(n)
        
        for _ in range(max_iterations):
            if np.all(np.abs(T[np.arange(1, n), np.arange(n - 1)]) < tol):
                break

            mu = T[-1, -1]
            Q, R = np.linalg.qr(T - mu * np.eye(n))
            T = R @ Q + mu * np.eye(n)
            Q_total = Q_total @ Q
        
        eigenvalues = np.diag(T)
        return eigenvalues, Q_total

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the eigenvalue problem for the symmetric matrix.

        Returns:
        - eigenvalues (np.ndarray): The eigenvalues of the matrix.
        - eigenvectors (np.ndarray): The eigenvectors of the matrix.
        """
        T, Q = self.tridiagonalize()
        eigenvalues, eigenvectors = self.qr_algorithm(T)
        eigenvectors = Q @ eigenvectors
        return eigenvalues, eigenvectors


if __name__ == "__main__":
    # 示例对称矩阵
    symmetric_matrix = np.array([
        [4, 1, 2],
        [1, 2, 0],
        [2, 0, 3]
    ])

    solver = SymmetricQRAlgorithm(symmetric_matrix)
    eigenvalues, eigenvectors = solver.solve()

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
