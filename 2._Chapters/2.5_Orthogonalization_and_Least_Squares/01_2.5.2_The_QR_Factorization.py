# 01_2.5.2_The_QR_Factorization

"""

Lecture: 2._Chapters/2.5_Orthogonalization_and_Least_Squares
Content: 01_2.5.2_The_QR_Factorization

"""

import numpy as np
from typing import Tuple

class QRDecomposition:
    """
    实现QR分解的类，提供Householder和Givens两种方法
    """

    @staticmethod
    def householder_reflection(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Householder变换对矩阵A进行QR分解

        Args:
            A (np.ndarray): 输入矩阵

        Returns:
            Tuple[np.ndarray, np.ndarray]: Q和R矩阵
        """
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy()

        for i in range(n):
            # 计算Householder向量
            x = R[i:, i]
            alpha = -np.sign(x[0]) * np.linalg.norm(x)
            v = x.copy()
            v[0] -= alpha
            v /= np.linalg.norm(v)

            # 计算Householder矩阵
            H = np.eye(m)
            H[i:, i:] -= 2.0 * np.outer(v, v)

            # 更新R和Q
            R = H @ R
            Q = Q @ H

        return Q, R

    @staticmethod
    def givens_rotation(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用Givens旋转对矩阵A进行QR分解

        Args:
            A (np.ndarray): 输入矩阵

        Returns:
            Tuple[np.ndarray, np.ndarray]: Q和R矩阵
        """
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy()

        for j in range(n):
            for i in range(m-1, j, -1):
                # 计算Givens旋转矩阵的元素
                a, b = R[i-1, j], R[i, j]
                r = np.hypot(a, b)
                c = a / r
                s = -b / r

                # 应用Givens旋转
                G = np.eye(m)
                G[i-1, i-1] = c
                G[i, i] = c
                G[i-1, i] = s
                G[i, i-1] = -s

                R = G @ R
                Q = Q @ G.T

        return Q, R

def main():
    """
    主函数，用于示例QR分解
    """
    # 示例矩阵
    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ], dtype=float)

    # 使用Householder变换进行QR分解
    print("使用Householder变换进行QR分解:")
    Q_h, R_h = QRDecomposition.householder_reflection(A)
    print("Q矩阵:")
    print(Q_h)
    print("R矩阵:")
    print(R_h)
    print("验证Q @ R是否等于A:")
    print(np.allclose(Q_h @ R_h, A))

    # 使用Givens旋转进行QR分解
    print("\n使用Givens旋转进行QR分解:")
    Q_g, R_g = QRDecomposition.givens_rotation(A)
    print("Q矩阵:")
    print(Q_g)
    print("R矩阵:")
    print(R_g)
    print("验证Q @ R是否等于A:")
    print(np.allclose(Q_g @ R_g, A))

if __name__ == "__main__":
    main()
