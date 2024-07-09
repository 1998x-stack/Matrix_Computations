# 00_2.4.1.1_Recursive_Block_Structures

"""

Lecture: 2._Chapters/2.4_Special_Linear_Systems/2.4.1_Diagonal_Dominance_and_Symmetry
Content: 00_2.4.1.1_Recursive_Block_Structures

"""

import numpy as np
from typing import Tuple

class RecursiveBlockCholesky:
    """
    递归块Cholesky分解类
    """

    def __init__(self, threshold: int = 32) -> None:
        """
        初始化递归块Cholesky分解器

        Args:
            threshold (int): 矩阵大小小于等于该值时使用直接Cholesky分解
        """
        self.threshold = threshold

    def cholesky_decomposition(self, A: np.ndarray) -> np.ndarray:
        """
        对矩阵A进行递归块Cholesky分解

        Args:
            A (np.ndarray): 输入的对称正定矩阵

        Returns:
            np.ndarray: Cholesky分解得到的下三角矩阵L
        """
        n = A.shape[0]
        if n <= self.threshold:
            return np.linalg.cholesky(A)

        m = n // 2

        # 分块
        A11 = A[:m, :m]
        A12 = A[:m, m:]
        A21 = A[m:, :m]
        A22 = A[m:, m:]

        # 递归分解
        L11 = self.cholesky_decomposition(A11)
        L21 = np.dot(A21, np.linalg.inv(L11.T))
        A22_new = A22 - np.dot(L21, L21.T)
        L22 = self.cholesky_decomposition(A22_new)

        # 构建结果矩阵
        L = np.zeros_like(A)
        L[:m, :m] = L11
        L[m:, :m] = L21
        L[m:, m:] = L22

        return L

def main():
    """
    主函数，用于示例递归块Cholesky分解
    """
    # 生成一个对称正定矩阵
    np.random.seed(0)
    A = np.random.rand(8, 8)
    A = np.dot(A, A.T)  # 保证矩阵是对称正定的

    # 初始化递归块Cholesky分解器
    solver = RecursiveBlockCholesky(threshold=2)
    
    # 对矩阵进行Cholesky分解
    L = solver.cholesky_decomposition(A)
    
    # 打印结果
    print("输入矩阵A:")
    print(A)
    print("\nCholesky分解得到的下三角矩阵L:")
    print(L)
    print("\n验证L @ L.T是否等于A:")
    print(np.allclose(np.dot(L, L.T), A))

if __name__ == "__main__":
    main()
