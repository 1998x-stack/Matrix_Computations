# 00_2.5.1_Householder_and_Givens_Transformations

"""

Lecture: 2._Chapters/2.5_Orthogonalization_and_Least_Squares
Content: 00_2.5.1_Householder_and_Givens_Transformations

"""

import numpy as np
from typing import Tuple

class HouseholderTransformation:
    """
    实现Householder变换的类
    """

    @staticmethod
    def reflect(v: np.ndarray) -> np.ndarray:
        """
        计算Householder反射向量

        Args:
            v (np.ndarray): 输入向量

        Returns:
            np.ndarray: Householder反射向量
        """
        alpha = -np.sign(v[0]) * np.linalg.norm(v)
        v1 = v.copy()
        v1[0] -= alpha
        v1 = v1 / np.linalg.norm(v1)
        return v1

    @staticmethod
    def apply_to_matrix(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对矩阵A应用Householder变换，将其分解为QR形式

        Args:
            A (np.ndarray): 输入矩阵

        Returns:
            Tuple[np.ndarray, np.ndarray]: Q和R矩阵
        """
        m, n = A.shape
        Q = np.eye(m)
        R = A.copy()

        for i in range(n):
            H = np.eye(m)
            v = HouseholderTransformation.reflect(R[i:, i])
            H[i:, i:] -= 2.0 * np.outer(v, v)
            R = H @ R
            Q = Q @ H

        return Q, R


class GivensTransformation:
    """
    实现Givens变换的类
    """

    @staticmethod
    def rotation(a: float, b: float) -> Tuple[float, float]:
        """
        计算Givens旋转矩阵的元素

        Args:
            a (float): 元素a
            b (float): 元素b

        Returns:
            Tuple[float, float]: c和s，分别是cos和sin值
        """
        r = np.hypot(a, b)
        c = a / r
        s = -b / r
        return c, s

    @staticmethod
    def apply_to_matrix(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对矩阵A应用Givens变换，将其分解为QR形式

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
                c, s = GivensTransformation.rotation(R[i-1, j], R[i, j])
                G = np.eye(m)
                G[[i-1, i], [i-1, i]] = c
                G[i-1, i] = s
                G[i, i-1] = -s
                R = G @ R
                Q = Q @ G.T

        return Q, R


def main():
    """
    主函数，用于示例Householder和Givens变换
    """
    # 示例矩阵
    A = np.array([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ], dtype=float)

    # 使用Householder变换
    print("使用Householder变换:")
    Q_h, R_h = HouseholderTransformation.apply_to_matrix(A)
    print("Q矩阵:")
    print(Q_h)
    print("R矩阵:")
    print(R_h)
    print("验证Q @ R是否等于A:")
    print(np.allclose(Q_h @ R_h, A))

    # 使用Givens变换
    print("\n使用Givens变换:")
    Q_g, R_g = GivensTransformation.apply_to_matrix(A)
    print("Q矩阵:")
    print(Q_g)
    print("R矩阵:")
    print(R_g)
    print("验证Q @ R是否等于A:")
    print(np.allclose(Q_g @ R_g, A))


if __name__ == "__main__":
    main()