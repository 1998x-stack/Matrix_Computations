# 01_2.6.2_Constrained_Least_Squares

"""

Lecture: 2._Chapters/2.6_Modified_Least_Squares_Problems_and_Methods
Content: 01_2.6.2_Constrained_Least_Squares

"""

import numpy as np
from typing import Tuple

class ConstrainedLeastSquares:
    """
    带约束的最小二乘问题求解器
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, B: np.ndarray = None, d: np.ndarray = None):
        """
        初始化带约束的最小二乘问题求解器

        Args:
            A (np.ndarray): 系数矩阵 A，形状为 (m, n)
            b (np.ndarray): 右侧向量 b，形状为 (m,)
            B (np.ndarray, optional): 约束矩阵 B，形状为 (p, n)，默认为 None
            d (np.ndarray, optional): 约束右侧向量 d，形状为 (p,)，默认为 None
        """
        self.A = A
        self.b = b
        self.B = B
        self.d = d

    def solve_lsqi(self, alpha: float) -> np.ndarray:
        """
        解带不等式约束的最小二乘问题

        Args:
            alpha (float): 约束范数

        Returns:
            np.ndarray: 解向量 x
        """
        U, s, VT = np.linalg.svd(self.A, full_matrices=False)
        c = U.T @ self.b
        w = np.divide(c[:s.size], s, where=s != 0)
        x_ls = VT.T @ w

        if np.linalg.norm(x_ls) <= alpha:
            return x_ls

        # 拉格朗日乘子法求解在约束边界上的解
        def f(lambda_):
            return np.sum((s**2 / (s**2 + lambda_)) * c**2) - alpha**2

        lambda_ = self._find_root(f, 0, np.max(s)**2)
        w = s / (s**2 + lambda_) * c
        x_constrained = VT.T @ w
        return x_constrained

    def solve_lse(self) -> np.ndarray:
        """
        解带等式约束的最小二乘问题

        Returns:
            np.ndarray: 解向量 x
        """
        Q, R = np.linalg.qr(self.B.T)
        p = self.B.shape[0]
        Q1 = Q[:, :p]
        Q2 = Q[:, p:]
        
        y = np.linalg.solve(R.T, self.d)
        b_hat = self.b - self.A @ Q1 @ y
        z = np.linalg.lstsq(self.A @ Q2, b_hat, rcond=None)[0]
        x = Q1 @ y + Q2 @ z
        return x

    def _find_root(self, func, a, b, tol=1e-10) -> float:
        """
        使用二分法求解方程的根

        Args:
            func (callable): 方程
            a (float): 区间左端点
            b (float): 区间右端点
            tol (float): 容差

        Returns:
            float: 方程的根
        """
        fa = func(a)
        fb = func(b)
        while (b - a) > tol:
            c = (a + b) / 2
            fc = func(c)
            if fc == 0:
                return c
            elif fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        return (a + b) / 2

# 示例用法
if __name__ == "__main__":
    # 带不等式约束的最小二乘问题
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)
    alpha = 1.0
    solver = ConstrainedLeastSquares(A, b)
    x_lsqi = solver.solve_lsqi(alpha)
    print("带不等式约束的最小二乘解 x_lsqi:")
    print(x_lsqi)

    # 带等式约束的最小二乘问题
    B = np.array([[1, 0], [0, 1]], dtype=float)
    d = np.array([1, 1], dtype=float)
    solver = ConstrainedLeastSquares(A, b, B, d)
    x_lse = solver.solve_lse()
    print("带等式约束的最小二乘解 x_lse:")
    print(x_lse)