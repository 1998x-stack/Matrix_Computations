# 05_2.5.6_Square_and_Underdetermined_Systems

"""

Lecture: 2._Chapters/2.5_Orthogonalization_and_Least_Squares
Content: 05_2.5.6_Square_and_Underdetermined_Systems

"""

import numpy as np
from typing import Tuple

class LinearSystemSolver:
    """
    线性系统求解器类，提供方形和欠定系统的求解方法
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        初始化线性系统求解器

        Args:
            A (np.ndarray): 系数矩阵
            b (np.ndarray): 右端项向量
        """
        self.A = A
        self.b = b

    def solve_square_system(self) -> np.ndarray:
        """
        使用QR分解求解方形系统 Ax = b

        Returns:
            np.ndarray: 解向量 x
        """
        Q, R = np.linalg.qr(self.A)
        x = np.linalg.solve(R, Q.T @ self.b)
        return x

    def solve_underdetermined_system(self) -> np.ndarray:
        """
        使用SVD求解欠定系统 Ax = b，找到最小范数解

        Returns:
            np.ndarray: 最小范数解向量 x
        """
        U, s, VT = np.linalg.svd(self.A, full_matrices=False)
        c = U.T @ self.b
        w = np.divide(c[:s.size], s, where=s != 0)
        x_min_norm = VT.T @ w
        return x_min_norm

def main():
    """
    主函数，用于示例方形和欠定系统的求解
    """
    # 示例方形系统
    A_square = np.array([[2, 1], [1, 3]], dtype=float)
    b_square = np.array([1, 2], dtype=float)
    
    solver_square = LinearSystemSolver(A_square, b_square)
    x_square = solver_square.solve_square_system()
    print("方形系统的解:")
    print(x_square)

    # 示例欠定系统
    A_underdetermined = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b_underdetermined = np.array([7, 8], dtype=float)
    
    solver_underdetermined = LinearSystemSolver(A_underdetermined, b_underdetermined)
    x_underdetermined = solver_underdetermined.solve_underdetermined_system()
    print("欠定系统的最小范数解:")
    print(x_underdetermined)

if __name__ == "__main__":
    main()