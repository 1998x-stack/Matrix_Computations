# 04_2.7.5_The_Practical_QR_Algorithm

"""

Lecture: 2._Chapters/2.7_Unsymmetric_Eigenvalue_Problems
Content: 04_2.7.5_The_Practical_QR_Algorithm

"""

import numpy as np
from typing import Tuple


def householder_transform(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对矩阵进行Householder变换，将其转化为Hessenberg形式。

    Args:
        A (np.ndarray): 输入矩阵。

    Returns:
        H (np.ndarray): Hessenberg形式的矩阵。
        Q (np.ndarray): 正交矩阵。
    """
    m, n = A.shape
    Q = np.eye(m)
    H = A.copy()

    for i in range(n-2):
        x = H[i+1:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (1 if x[0] == 0 else np.sign(x[0]))
        u = x + e
        u = u / np.linalg.norm(u)

        H[i+1:, i:] -= 2 * np.outer(u, u @ H[i+1:, i:])
        H[:, i+1:] -= 2 * np.outer(H[:, i+1:] @ u, u)
        Q[i+1:] -= 2 * np.outer(u, u @ Q[i+1:])

    return H, Q.T


def qr_iteration(H: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    对Hessenberg矩阵进行QR迭代，逼近实Schur形式。

    Args:
        H (np.ndarray): Hessenberg形式的输入矩阵。
        max_iter (int): 最大迭代次数。
        tol (float): 收敛容差。

    Returns:
        T (np.ndarray): 上三角块状矩阵。
        Q (np.ndarray): 正交矩阵。
    """
    n = H.shape[0]
    Q_total = np.eye(n)

    for _ in range(max_iter):
        Q, R = np.linalg.qr(H)
        H = R @ Q
        Q_total = Q_total @ Q

        # 检查收敛性
        off_diagonal_norm = np.sum(np.abs(H[np.tril_indices(n, -1)]))
        if off_diagonal_norm < tol:
            break

    return H, Q_total


class PracticalQRAlgorithm:
    """
    实用QR算法类，用于计算矩阵的Schur形式。

    Attributes:
        A (np.ndarray): 输入矩阵。
        T (np.ndarray): 上三角块状矩阵。
        Q (np.ndarray): 正交矩阵。
    """

    def __init__(self, A: np.ndarray):
        """
        初始化PracticalQRAlgorithm类。

        Args:
            A (np.ndarray): 输入矩阵。
        """
        self.A = A
        self.T = None
        self.Q = None

    def compute_schur_form(self) -> None:
        """
        计算输入矩阵的Schur形式。
        """
        # 转换为Hessenberg形式
        H, Q = householder_transform(self.A)

        # QR迭代
        T, Q_schur = qr_iteration(H)

        # 最终的正交矩阵
        self.Q = Q @ Q_schur
        self.T = T

    def print_results(self) -> None:
        """
        打印计算结果。
        """
        if self.T is None or self.Q is None:
            print("请先计算Schur形式。")
        else:
            print("原始矩阵 A:")
            print(self.A)
            print("\nSchur形式的上三角块状矩阵 T:")
            print(self.T)
            print("\n正交矩阵 Q:")
            print(self.Q)
            print("\n验证 A = Q T Q^T:")
            print(np.allclose(self.A, self.Q @ self.T @ self.Q.T))


def main():
    """
    主函数，用于测试PracticalQRAlgorithm类。
    """
    A = np.array([[4, 1, 2],
                  [3, 4, 1],
                  [1, 1, 3]], dtype=float)

    qr_algo = PracticalQRAlgorithm(A)
    qr_algo.compute_schur_form()
    qr_algo.print_results()


if __name__ == "__main__":
    main()
