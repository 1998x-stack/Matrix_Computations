# 03_2.6.4_Subspace_Computations_with_the_SVD

"""

Lecture: 2._Chapters/2.6_Modified_Least_Squares_Problems_and_Methods
Content: 03_2.6.4_Subspace_Computations_with_the_SVD

"""

import numpy as np
from typing import Tuple

class SubspaceComputations:
    """
    使用奇异值分解（SVD）进行子空间计算的类
    """

    @staticmethod
    def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        计算两个子空间之间的最优旋转矩阵

        Args:
            A (np.ndarray): 第一个矩阵，形状为 (m, p)
            B (np.ndarray): 第二个矩阵，形状为 (m, p)

        Returns:
            np.ndarray: 最优旋转矩阵 Q，形状为 (p, p)
        """
        U, _, Vt = np.linalg.svd(np.dot(B.T, A))
        Q = np.dot(U, Vt)
        return Q

    @staticmethod
    def principal_angles(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算两个子空间之间的主要角度和向量

        Args:
            A (np.ndarray): 第一个子空间矩阵，形状为 (m, p)
            B (np.ndarray): 第二个子空间矩阵，形状为 (m, p)

        Returns:
            Tuple[np.ndarray, np.ndarray]: 主要角度和对应的主要向量
        """
        Q_A, _ = np.linalg.qr(A)
        Q_B, _ = np.linalg.qr(B)
        C = np.dot(Q_A.T, Q_B)
        U, Sigma, Vt = np.linalg.svd(C)
        return Sigma, (Q_A @ U, Q_B @ Vt.T)

    @staticmethod
    def subspace_intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        计算两个子空间的交集

        Args:
            A (np.ndarray): 第一个子空间矩阵，形状为 (m, p)
            B (np.ndarray): 第二个子空间矩阵，形状为 (m, p)

        Returns:
            np.ndarray: 交集子空间的基矩阵
        """
        _, Sigma, (U_A, U_B) = SubspaceComputations.principal_angles(A, B)
        intersection_mask = np.isclose(Sigma, 1)
        intersection = U_A[:, intersection_mask]
        return intersection

    @staticmethod
    def subspace_distance(A: np.ndarray, B: np.ndarray) -> float:
        """
        计算两个子空间之间的距离

        Args:
            A (np.ndarray): 第一个子空间矩阵，形状为 (m, p)
            B (np.ndarray): 第二个子空间矩阵，形状为 (m, p)

        Returns:
            float: 两个子空间之间的距离
        """
        Q_A, _ = np.linalg.qr(A)
        Q_B, _ = np.linalg.qr(B)
        P_A = np.dot(Q_A, Q_A.T)
        P_B = np.dot(Q_B, Q_B.T)
        dist = np.linalg.norm(P_A - P_B, ord='fro')
        return dist

def main():
    """
    主函数，用于示例子空间计算方法
    """
    # 示例矩阵
    A = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=float)
    B = np.array([
        [7, 8],
        [9, 10],
        [11, 12]
    ], dtype=float)

    # 计算最优旋转矩阵
    Q = SubspaceComputations.orthogonal_procrustes(A, B)
    print("最优旋转矩阵 Q:")
    print(Q)

    # 计算主要角度和向量
    angles, vectors = SubspaceComputations.principal_angles(A, B)
    print("\n主要角度的余弦值:")
    print(angles)
    print("\n主要向量:")
    print("子空间A中的主要向量:")
    print(vectors[0])
    print("子空间B中的主要向量:")
    print(vectors[1])

    # 计算子空间的交集
    intersection = SubspaceComputations.subspace_intersection(A, B)
    print("\n子空间的交集:")
    print(intersection)

    # 计算子空间距离
    distance = SubspaceComputations.subspace_distance(A, B)
    print("\n子空间之间的距离:")
    print(distance)

if __name__ == "__main__":
    main()
