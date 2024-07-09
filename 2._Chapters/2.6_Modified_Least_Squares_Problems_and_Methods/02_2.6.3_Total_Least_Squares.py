# 02_2.6.3_Total_Least_Squares

"""

Lecture: 2._Chapters/2.6_Modified_Least_Squares_Problems_and_Methods
Content: 02_2.6.3_Total_Least_Squares

"""
import numpy as np
from typing import Tuple

class TotalLeastSquares:
    """
    总最小二乘法（TLS）求解器
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        """
        初始化总最小二乘法求解器

        Args:
            A (np.ndarray): 系数矩阵 A，形状为 (m, n)
            b (np.ndarray): 观测向量 b，形状为 (m,)
        """
        self.A = A
        self.b = b

    def solve(self) -> np.ndarray:
        """
        求解总最小二乘问题

        Returns:
            np.ndarray: 解向量 x
        """
        # 构建扩展矩阵 [A | b]
        m, n = self.A.shape
        Z = np.hstack((self.A, self.b.reshape(-1, 1)))

        # 对扩展矩阵进行SVD分解
        U, s, VT = np.linalg.svd(Z, full_matrices=False)
        V = VT.T

        # 最小奇异值对应的右奇异向量
        v_min = V[:, -1]

        # 计算TLS解
        x_tls = -v_min[:n] / v_min[n]

        return x_tls

def main():
    """
    主函数，用于示例总最小二乘问题的求解
    """
    # 示例数据
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    b = np.array([7.0, 8.0, 9.0])

    # 创建TLS求解器实例
    tls_solver = TotalLeastSquares(A, b)

    # 求解总最小二乘问题
    x_tls = tls_solver.solve()

    # 打印结果
    print("总最小二乘解 x_tls:")
    print(x_tls)

if __name__ == "__main__":
    main()

