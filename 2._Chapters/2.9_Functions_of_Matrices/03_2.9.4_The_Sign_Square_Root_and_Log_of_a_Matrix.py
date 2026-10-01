# 03_2.9.4_The_Sign_Square_Root_and_Log_of_a_Matrix

"""

Lecture: 2._Chapters/2.9_Functions_of_Matrices
Content: 03_2.9.4_The_Sign_Square_Root_and_Log_of_a_Matrix

"""

import numpy as np
from numpy.linalg import norm, inv
from typing import Tuple

class MatrixFunctions:
    """
    计算矩阵的符号、平方根和对数的类。
    """

    def __init__(self, A: np.ndarray):
        """
        初始化矩阵 A。

        Args:
            A (np.ndarray): 输入矩阵。
        """
        self.A = A
        self.n = A.shape[0]

    def matrix_sign(self, tol: float = 1e-10) -> np.ndarray:
        """
        计算矩阵的符号函数。

        Args:
            tol (float): 迭代终止的容差。

        Returns:
            np.ndarray: 矩阵的符号函数。
        """
        S = self.A
        I = np.eye(self.n)
        while norm(S - inv(S)) > tol:
            S = 0.5 * (S + inv(S))
        return S

    def matrix_square_root(self, tol: float = 1e-10) -> np.ndarray:
        """
        计算矩阵的平方根。

        Args:
            tol (float): 迭代终止的容差。

        Returns:
            np.ndarray: 矩阵的平方根。
        """
        X = self.A
        I = np.eye(self.n)
        while norm(X @ X - self.A) > tol:
            X = 0.5 * (X + inv(X) @ self.A)
        return X

    def matrix_log(self, tol: float = 1e-10) -> np.ndarray:
        """
        计算矩阵的对数。

        Args:
            tol (float): 迭代终止的容差。

        Returns:
            np.ndarray: 矩阵的对数。
        """
        m = max(0, int(np.ceil(np.log2(norm(self.A, np.inf)))))
        A_scaled = self.A / (2**m)
        I = np.eye(self.n)
        L = np.zeros_like(self.A)
        for k in range(1, 100):
            term = (-1)**(k+1) * (A_scaled - I)**k / k
            L += term
            if norm(term) < tol:
                break
        return L * (2**m)

    def verify_results(self):
        """
        验证计算结果的正确性。
        """
        sign_A = self.matrix_sign()
        sqrt_A = self.matrix_square_root()
        log_A = self.matrix_log()

        # 验证符号函数
        assert np.allclose(sign_A @ sign_A, np.eye(self.n)), "Sign function verification failed."
        # 验证平方根
        assert np.allclose(sqrt_A @ sqrt_A, self.A), "Square root function verification failed."
        # 验证对数函数
        assert np.allclose(np.exp(log_A), self.A), "Logarithm function verification failed."

        print("所有结果均已验证正确！")

# 示例使用
if __name__ == "__main__":
    # 定义矩阵 A
    A = np.array([[4, 0], [0, 9]])

    # 创建 MatrixFunctions 实例
    matrix_funcs = MatrixFunctions(A)

    # 计算并打印矩阵的符号函数
    sign_A = matrix_funcs.matrix_sign()
    print(f"矩阵的符号函数:\n{sign_A}")

    # 计算并打印矩阵的平方根
    sqrt_A = matrix_funcs.matrix_square_root()
    print(f"矩阵的平方根:\n{sqrt_A}")

    # 计算并打印矩阵的对数
    log_A = matrix_funcs.matrix_log()
    print(f"矩阵的对数:\n{log_A}")

    # 验证结果
    matrix_funcs.verify_results()
