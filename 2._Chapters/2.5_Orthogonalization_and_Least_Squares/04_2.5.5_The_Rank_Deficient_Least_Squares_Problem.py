# 04_2.5.5_The_Rank-Deficient_Least_Squares_Problem

"""

Lecture: 2._Chapters/2.5_Orthogonalization_and_Least_Squares
Content: 04_2.5.5_The_Rank-Deficient_Least_Squares_Problem

"""

import numpy as np
from typing import Tuple

class RankDeficientLeastSquares:
    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        Solves the rank-deficient least squares problem for the system Ax = b.

        Parameters:
        A (np.ndarray): Coefficient matrix A of shape (m, n)
        b (np.ndarray): Right-hand side vector b of shape (m,)

        Attributes:
        A (np.ndarray): Coefficient matrix A
        b (np.ndarray): Right-hand side vector b
        """
        self.A = A
        self.b = b

    def solve_with_svd(self) -> np.ndarray:
        """
        Solves the least squares problem using Singular Value Decomposition (SVD).

        Returns:
        np.ndarray: Solution vector x
        """
        U, s, VT = np.linalg.svd(self.A, full_matrices=False)
        V = VT.T
        S_inv = np.diag(1 / s)
        x_svd = V @ S_inv @ U.T @ self.b
        return x_svd

    def solve_with_qr(self) -> np.ndarray:
        """
        Solves the least squares problem using QR decomposition with column pivoting.

        Returns:
        np.ndarray: Solution vector x
        """
        Q, R, P = np.linalg.qr(self.A, mode='complete')
        y = np.dot(Q.T, self.b)
        x_qr = np.linalg.solve(R[:R.shape[1]], y[:R.shape[1]])
        return x_qr

# Example usage:
if __name__ == "__main__":
    # Example matrix A and vector b (you can replace these with your own data)
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])
    b = np.array([1, 2, 3, 4])
    
    # Solve using SVD
    solver_svd = RankDeficientLeastSquares(A, b)
    x_svd = solver_svd.solve_with_svd()
    print("Solution using SVD:", x_svd)
    
    # Solve using QR with column pivoting
    solver_qr = RankDeficientLeastSquares(A, b)
    x_qr = solver_qr.solve_with_qr()
    print("Solution using QR with column pivoting:", x_qr)
