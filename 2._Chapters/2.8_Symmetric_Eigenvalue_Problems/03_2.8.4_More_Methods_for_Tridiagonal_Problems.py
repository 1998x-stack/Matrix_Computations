# 03_2.8.4_More_Methods_for_Tridiagonal_Problems

"""

Lecture: 2._Chapters/2.8_Symmetric_Eigenvalue_Problems
Content: 03_2.8.4_More_Methods_for_Tridiagonal_Problems

"""

import numpy as np
from typing import Callable, Tuple, List

class SymmetricEigenSolver:
    def __init__(self, diagonal: np.ndarray, off_diagonal: np.ndarray):
        """
        Initialize the symmetric tridiagonal eigenvalue solver.

        Args:
        - diagonal (np.ndarray): Main diagonal elements of the tridiagonal matrix.
        - off_diagonal (np.ndarray): Off-diagonal elements of the tridiagonal matrix.
        """
        self.diagonal = diagonal
        self.off_diagonal = off_diagonal
        self.n = len(diagonal)
    
    def solve_using_bisection(self, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the symmetric tridiagonal eigenvalue problem using the bisection method.

        Args:
        - tolerance (float): Tolerance for convergence.

        Returns:
        - eigenvalues (np.ndarray): Array of eigenvalues.
        - eigenvectors (np.ndarray): Array of corresponding eigenvectors.
        """
        def eigenvalue_function(x: float) -> float:
            """
            Compute the value of the characteristic polynomial at x.

            Args:
            - x (float): Input value.

            Returns:
            - value (float): Value of the polynomial at x.
            """
            b = self.diagonal - x
            d = np.ones(self.n)
            d[1:] -= (self.off_diagonal ** 2) / b[:-1]
            return np.linalg.norm(d, ord=np.inf)

        eigenvalues = np.zeros(self.n)
        eigenvectors = np.zeros((self.n, self.n))

        # Perform bisection for each eigenvalue
        for i in range(self.n):
            a, b = np.min(self.diagonal), np.max(self.diagonal)
            while b - a > tolerance:
                mid = (a + b) / 2.0
                if eigenvalue_function(mid) < tolerance:
                    b = mid
                else:
                    a = mid
            eigenvalues[i] = (a + b) / 2.0
            eigenvectors[:, i] = self._compute_eigenvector(eigenvalues[i])

        return eigenvalues, eigenvectors

    def _compute_eigenvector(self, eigenvalue: float) -> np.ndarray:
        """
        Compute the eigenvector corresponding to a given eigenvalue.

        Args:
        - eigenvalue (float): Eigenvalue.

        Returns:
        - eigenvector (np.ndarray): Eigenvector.
        """
        b = self.diagonal - eigenvalue
        eigenvector = np.ones(self.n)
        for i in range(1, self.n):
            eigenvector[i] = self.off_diagonal[i-1] / np.sqrt(b[i-1])
        eigenvector /= np.linalg.norm(eigenvector)
        return eigenvector

    def solve_using_sturm_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the symmetric tridiagonal eigenvalue problem using Sturm sequence method.

        Returns:
        - eigenvalues (np.ndarray): Array of eigenvalues.
        - eigenvectors (np.ndarray): Array of corresponding eigenvectors.
        """
        def count_eigenvalues(x: float) -> int:
            """
            Count the number of eigenvalues less than x.

            Args:
            - x (float): Input value.

            Returns:
            - count (int): Number of eigenvalues less than x.
            """
            sturm_sequence = np.zeros(self.n + 1)
            sturm_sequence[0] = 1
            sturm_sequence[1] = self.diagonal[0] - x

            for i in range(2, self.n + 1):
                sturm_sequence[i] = (self.diagonal[i-1] - x) * sturm_sequence[i-1] - (self.off_diagonal[i-2] ** 2) * sturm_sequence[i-2]
                if sturm_sequence[i-1] == 0 and sturm_sequence[i] == 0:
                    break

            count = 0
            for i in range(1, self.n + 1):
                if sturm_sequence[i] * sturm_sequence[i-1] < 0:
                    count += 1

            return count

        eigenvalues = np.zeros(self.n)
        eigenvectors = np.zeros((self.n, self.n))

        # Determine eigenvalues using Sturm sequence method
        for i in range(self.n):
            a, b = np.min(self.diagonal), np.max(self.diagonal)
            while b - a > tolerance:
                mid = (a + b) / 2.0
                if count_eigenvalues(mid) < i + 1:
                    a = mid
                else:
                    b = mid
            eigenvalues[i] = (a + b) / 2.0
            eigenvectors[:, i] = self._compute_eigenvector(eigenvalues[i])

        return eigenvalues, eigenvectors

# Example usage:
if __name__ == "__main__":
    # Example symmetric tridiagonal matrix
    diagonal = np.array([2.0, 3.0, 4.0])
    off_diagonal = np.array([1.0, 2.0])
    
    solver = SymmetricEigenSolver(diagonal, off_diagonal)
    
    # Solve using bisection method
    eigenvalues_bisection, eigenvectors_bisection = solver.solve_using_bisection()
    print("Eigenvalues (Bisection Method):", eigenvalues_bisection)
    
    # Solve using Sturm sequence method
    eigenvalues_sturm, eigenvectors_sturm = solver.solve_using_sturm_sequence()
    print("Eigenvalues (Sturm Sequence Method):", eigenvalues_sturm)
